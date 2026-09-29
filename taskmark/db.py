"""SQLite storage. Written from day one because Claude Code prunes its own JSONL logs.

Schema is deliberately flat: one row per assistant turn, one row per task, plus the
files each task touched. No analysis here — stage 2 reads these tables.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterable

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    request_id        TEXT PRIMARY KEY,          -- api request id: makes ingest idempotent
    ts                REAL NOT NULL,             -- unix seconds, from the turn timestamp
    session_id        TEXT NOT NULL,
    cwd               TEXT,
    model             TEXT,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
    context_size      INTEGER NOT NULL DEFAULT 0, -- input + cache_read + cache_creation
    is_sidechain      INTEGER NOT NULL DEFAULT 0, -- 1 = subagent turn
    task_id           INTEGER,
    effort            TEXT,
    version           TEXT,
    message_id        TEXT,
    cost_usd          REAL
);
CREATE INDEX IF NOT EXISTS turns_ts ON turns(ts);
CREATE INDEX IF NOT EXISTS turns_task ON turns(task_id);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, ts);

CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    cwd           TEXT,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    title         TEXT,                  -- first prompt, trimmed
    slug          TEXT,
    boundary_reason TEXT,                -- why the PREVIOUS task ended
    merged_into   INTEGER,               -- set when a "return" proved it was the same task
    note_path     TEXT,
    prompts       INTEGER NOT NULL DEFAULT 0,
    calib_version INTEGER NOT NULL DEFAULT 0   -- criterion version: do not compare across versions
);
CREATE INDEX IF NOT EXISTS tasks_session ON tasks(session_id, started_at);

CREATE TABLE IF NOT EXISTS task_files (
    task_id  INTEGER NOT NULL,
    path     TEXT NOT NULL,
    hits     INTEGER NOT NULL DEFAULT 1,
    last_ts  REAL NOT NULL,
    PRIMARY KEY (task_id, path)
);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    session_id TEXT,
    task_id    INTEGER,
    kind       TEXT NOT NULL,            -- prompt | boundary | note | merge | hook-error
    payload    TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS signal_samples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    session_id TEXT,
    kind       TEXT NOT NULL,          -- pause | file_overlap | cosine
    value      REAL NOT NULL,
    was_boundary INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS samples_kind ON signal_samples(kind, id);

CREATE TABLE IF NOT EXISTS calibration (
    version    INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    params     TEXT NOT NULL,          -- json: thresholds actually used from now on
    samples    INTEGER NOT NULL,
    note       TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    session_id TEXT,
    task_id    INTEGER,
    kind       TEXT NOT NULL           -- same-task | boundary-here
);

-- Which repository state a task started and ended at. Filled when the task first touches a
-- repo and when it closes, so a diff/log of exactly that task is one command away.
CREATE TABLE IF NOT EXISTS task_git (
    task_id    INTEGER NOT NULL,
    repo       TEXT NOT NULL,           -- repository root on this machine
    branch     TEXT,
    head_start TEXT,
    head_end   TEXT,
    dirty_end  INTEGER NOT NULL DEFAULT 0,
    commits    INTEGER NOT NULL DEFAULT 0,
    first_ts   REAL NOT NULL,
    PRIMARY KEY (task_id, repo)
);
CREATE INDEX IF NOT EXISTS task_git_head ON task_git(head_end);

-- Decisions worth remembering: what was decided, why, and what was rejected. Written by the
-- one who decided, at the moment of deciding (see cli.cmd_decide) — never guessed from words.
CREATE TABLE IF NOT EXISTS decisions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    session_id TEXT,
    task_id    INTEGER,
    who        TEXT NOT NULL DEFAULT 'claude',   -- user | claude
    what       TEXT NOT NULL,
    why        TEXT,
    instead    TEXT,                             -- the alternative that was rejected
    repo       TEXT,
    head       TEXT                              -- where the code stood when it was decided
);
CREATE INDEX IF NOT EXISTS decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS decisions_task ON decisions(task_id);

CREATE TABLE IF NOT EXISTS ingest_state (
    transcript_path TEXT PRIMARY KEY,
    offset          INTEGER NOT NULL DEFAULT 0,
    updated_at      REAL NOT NULL
);

-- One row per prompt: every signal the detector measured, what it decided, and — once a
-- correction arrives — whether that decision was right. `signal_samples` keeps each signal
-- on its own row, which is all a percentile needs; learning needs the whole vector of one
-- decision together with its outcome, and the prompt that produced it.
CREATE TABLE IF NOT EXISTS prompt_signals (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    session_id TEXT,
    task_id    INTEGER,
    prompt     TEXT,                    -- truncated: enough to recognise, not to archive
    signals    TEXT,                    -- JSON: pause, file_overlap, cosine, target_overlap...
    reason     TEXT,                    -- what the rules answered ('' when they saw no boundary)
    rule_boundary INTEGER NOT NULL DEFAULT 0,
    label      TEXT,                    -- 'new' | 'same' — filled in by a later correction
    label_src  TEXT                     -- who corrected: model | user | merge
);
CREATE INDEX IF NOT EXISTS prompt_signals_ts ON prompt_signals(ts);
CREATE INDEX IF NOT EXISTS prompt_signals_label ON prompt_signals(label);
"""


# Columns added after the first release. CREATE TABLE IF NOT EXISTS leaves an existing table
# alone, so a database made by an older version keeps its old shape until it is widened here.
ADDED_COLUMNS = {
    "turns": [("message_id", "TEXT"), ("cost_usd", "REAL")],
    "tasks": [("calib_version", "INTEGER NOT NULL DEFAULT 0"), ("merged_into", "INTEGER"),
              ("note_path", "TEXT")],
}


def _migrate(con) -> None:
    for tbl, cols in ADDED_COLUMNS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({tbl})")}
        if not have:                                    # table not created yet — schema will
            continue
        for name, decl in cols:
            if name not in have:
                con.execute(f"ALTER TABLE {tbl} ADD COLUMN {name} {decl}")


@contextmanager
def connect():
    config.ensure_dirs()
    con = sqlite3.connect(config.DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        _migrate(con)
        yield con
        con.commit()
    finally:
        con.close()


def add_decision(con, *, session_id: str | None, task_id: int | None, who: str, what: str,
                 why: str | None, instead: str | None, repo: str | None, head: str | None,
                 ts: float) -> int:
    cur = con.execute("INSERT INTO decisions (ts, session_id, task_id, who, what, why, instead, "
                      "repo, head) VALUES (?,?,?,?,?,?,?,?,?)",
                      (ts, session_id, task_id, who, what, why, instead, repo, head))
    return int(cur.lastrowid)


def task_spans(con, session_id: str, task_id: int, since: float,
               until: float | None = None) -> tuple[list[tuple[float, float]], set[str]]:
    """The stretches of the session this task owned, from its turns.

    A run of the task's own turns starts right after the previous foreign turn (its prompt
    sits there) and ends at its last own turn. Empty when the task has no turns yet — the
    caller then falls back to the plain time window."""
    rows = con.execute(
        "SELECT ts, task_id, request_id FROM turns WHERE session_id=? AND is_sidechain=0 "
        "AND ts>=? ORDER BY ts", (session_id, since)).fetchall()
    spans: list[tuple[float, float]] = []
    own: set[str] = set()
    prev_other, start, last_own = since, None, None
    for ts, tid, rid in rows:
        if until and ts > until:
            break
        if tid == task_id:
            if start is None:
                start = prev_other
            last_own = ts
            if rid:
                own.add(rid)
        else:
            if start is not None:
                spans.append((start, last_own))
                start = None
            prev_other = ts
    if start is not None:
        spans.append((start, until or float("inf")))
    return spans, own


def decisions_of(con, task_id: int) -> list:
    return list(con.execute("SELECT * FROM decisions WHERE task_id=? ORDER BY ts", (task_id,)))


def git_start(con, task_id: int, repo: str, branch: str | None, head: str | None,
              ts: float) -> None:
    con.execute("INSERT OR IGNORE INTO task_git (task_id, repo, branch, head_start, first_ts) "
                "VALUES (?,?,?,?,?)", (task_id, repo, branch, head, ts))


def git_finish(con, task_id: int, repo: str, head: str | None, dirty: bool, commits: int) -> None:
    con.execute("UPDATE task_git SET head_end=?, dirty_end=?, commits=? WHERE task_id=? AND repo=?",
                (head, int(dirty), commits, task_id, repo))


def git_rows(con, task_id: int) -> list:
    return list(con.execute("SELECT * FROM task_git WHERE task_id=? ORDER BY repo", (task_id,)))


def task_for_commit(con, sha: str) -> list:
    """Reverse lookup: which task's work contains this commit."""
    return list(con.execute(
        "SELECT t.*, g.repo, g.head_start, g.head_end FROM task_git g JOIN tasks t ON t.id=g.task_id "
        "WHERE g.head_end LIKE ? OR g.head_start LIKE ? ORDER BY t.started_at DESC", (sha + "%", sha + "%")))


def log_event(con, kind: str, session_id: str | None = None, task_id: int | None = None,
              payload: str | None = None) -> None:
    con.execute("INSERT INTO events (ts, session_id, task_id, kind, payload) VALUES (?,?,?,?,?)",
                (time.time(), session_id, task_id, kind, payload))


def start_task(con, session_id: str, cwd: str, title: str, slug: str, reason: str,
               started_at: float | None = None, calib_version: int = 0) -> int:
    cur = con.execute(
        "INSERT INTO tasks (session_id, cwd, started_at, title, slug, boundary_reason, prompts, "
        "calib_version) VALUES (?,?,?,?,?,?,1,?)",   # the prompt that opened the task counts
        (session_id, cwd, started_at or time.time(), title[:300], slug, reason, calib_version))
    return int(cur.lastrowid)


def end_task(con, task_id: int, note_path: str | None = None,
             ended_at: float | None = None) -> None:
    con.execute("UPDATE tasks SET ended_at=COALESCE(ended_at, ?), note_path=COALESCE(?, note_path) "
                "WHERE id=?", (ended_at or time.time(), note_path, task_id))


def get_task(con, task_id: int) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()


def touch_files(con, task_id: int, paths: Iterable[str]) -> None:
    now = time.time()
    for p in paths:
        con.execute(
            "INSERT INTO task_files (task_id, path, hits, last_ts) VALUES (?,?,1,?) "
            "ON CONFLICT(task_id, path) DO UPDATE SET hits=hits+1, last_ts=excluded.last_ts",
            (task_id, p, now))


def task_file_set(con, task_id: int) -> set[str]:
    return {r["path"] for r in con.execute("SELECT path FROM task_files WHERE task_id=?", (task_id,))}


def find_task_by_files(con, session_id: str, paths: Iterable[str], not_task: int | None,
                       since_ts: float) -> int | None:
    """Which earlier task of this session owns these files? Used for the return-merge rule."""
    paths = list(paths)
    if not paths:
        return None
    qs = ",".join("?" * len(paths))
    sql = ("SELECT f.task_id, COUNT(*) n FROM task_files f JOIN tasks t ON t.id=f.task_id "
           f"WHERE t.session_id=? AND t.started_at>=? AND f.path IN ({qs}) ")
    args: list[Any] = [session_id, since_ts, *paths]
    if not_task:
        sql += "AND f.task_id<>? "
        args.append(not_task)
    sql += "GROUP BY f.task_id ORDER BY n DESC LIMIT 1"
    row = con.execute(sql, args).fetchone()
    return int(row["task_id"]) if row else None


def merge_task(con, src_task: int, dst_task: int) -> None:
    """The 'new' task turned out to be a return to an older one: move everything over."""
    con.execute("UPDATE turns SET task_id=? WHERE task_id=?", (dst_task, src_task))
    for r in con.execute("SELECT path, hits, last_ts FROM task_files WHERE task_id=?", (src_task,)):
        con.execute(
            "INSERT INTO task_files (task_id, path, hits, last_ts) VALUES (?,?,?,?) "
            "ON CONFLICT(task_id, path) DO UPDATE SET hits=hits+excluded.hits, "
            "last_ts=MAX(last_ts, excluded.last_ts)",
            (dst_task, r["path"], r["hits"], r["last_ts"]))
    con.execute("DELETE FROM task_files WHERE task_id=?", (src_task,))
    con.execute("UPDATE tasks SET merged_into=?, ended_at=COALESCE(ended_at, ?) WHERE id=?",
                (dst_task, time.time(), src_task))
    con.execute("UPDATE tasks SET ended_at=NULL WHERE id=?", (dst_task,))


def save_turns(con, rows: list[dict], task_id: int | None, since_ts: float | None = None) -> int:
    """Idempotent by request_id — re-reading a transcript never double-counts.

    `since_ts` is the moment the task began: turns older than that predate it and are stored
    unattributed. Without this, enabling the plugin inside a session that is already running
    hands the task every turn of the backlog — the first task ever created here collected
    8525 turns from 4 prompts that way, and then topped every report.
    """
    n = 0
    for r in rows:
        owner = task_id
        if owner is not None and since_ts is not None and float(r.get("ts") or 0) < since_ts:
            owner = None
        cur = con.execute(
            "INSERT OR IGNORE INTO turns (request_id, ts, session_id, cwd, model, input_tokens, "
            "output_tokens, cache_creation_tokens, cache_read_tokens, context_size, is_sidechain, "
            "task_id, effort, version, message_id, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["request_id"], r["ts"], r["session_id"], r.get("cwd"), r.get("model"),
             r.get("input_tokens", 0), r.get("output_tokens", 0),
             r.get("cache_creation_tokens", 0), r.get("cache_read_tokens", 0),
             r.get("context_size", 0), 1 if r.get("is_sidechain") else 0,
             owner, r.get("effort"), r.get("version"), r.get("message_id"),
             r.get("cost_usd")))
        n += cur.rowcount
    return n


def get_ingest_offset(con, path: str) -> int:
    row = con.execute("SELECT offset FROM ingest_state WHERE transcript_path=?", (path,)).fetchone()
    return int(row["offset"]) if row else 0


def set_ingest_offset(con, path: str, offset: int) -> None:
    con.execute("INSERT INTO ingest_state (transcript_path, offset, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(transcript_path) DO UPDATE SET offset=excluded.offset, "
                "updated_at=excluded.updated_at", (path, offset, time.time()))


def add_prompt_signals(con, *, session_id: str, task_id: int | None, prompt: str,
                       signals: dict, reason: str, boundary: bool, ts: float) -> int | None:
    """Record one detector decision with everything it was based on."""
    try:
        cur = con.execute(
            "INSERT INTO prompt_signals (ts, session_id, task_id, prompt, signals, reason, "
            "rule_boundary) VALUES (?,?,?,?,?,?,?)",
            (ts, session_id, task_id, (prompt or "")[:300],
             json.dumps(signals, ensure_ascii=False), reason or "", 1 if boundary else 0))
        return int(cur.lastrowid)
    except Exception:
        return None


def label_last_prompt(con, session_id: str, label: str, src: str = "model") -> None:
    """Attach a correction to the most recent decision of this session.

    Called when Claude or the user says `mark new` / `mark same`: that verdict is the
    ground truth the rules should have produced.
    """
    row = con.execute("SELECT id FROM prompt_signals WHERE session_id=? "
                      "ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
    if row:
        con.execute("UPDATE prompt_signals SET label=?, label_src=? WHERE id=?",
                    (label, src, row["id"] if hasattr(row, "keys") else row[0]))


def label_task_prompts(con, task_id: int, label: str, src: str = "merge") -> None:
    """A task merged back = every boundary that opened it was wrong."""
    con.execute("UPDATE prompt_signals SET label=?, label_src=? "
                "WHERE task_id=? AND label IS NULL", (label, src, task_id))


def trim_prompt_signals(con, keep: int) -> None:
    con.execute("DELETE FROM prompt_signals WHERE id < "
                "(SELECT MAX(id) - ? FROM prompt_signals)", (keep,))
