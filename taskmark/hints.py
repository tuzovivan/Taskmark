"""What the plugin says back to the session — kept small, and always optional.

Two things only:

1. **A hint at a task boundary.** The plugin never clears the context itself (hooks cannot run
   /clear, and it was never meant to). But when it has just closed a task and written its note
   while the context is large, it says so in one line: the previous task is saved, the context is
   N tokens, /clear is safe now. The decision stays with the user.

2. **The notes after a /clear.** A cleared or compacted session starts with nothing, and Claude
   has no way to know the research notes exist. On SessionStart with source clear / compact /
   startup, the plugin adds a short list of this project's recent notes — paths and titles, not
   contents — so the right one can be read when it becomes relevant. A few hundred tokens, not
   the notes themselves.

Settings live in ~/.claude/taskmark/settings.json, all optional:

    {"hint_min_context": 150000,   # below this the boundary hint stays silent
     "hint_color": null,           # e.g. "yellow" — ANSI colour for the hint, off by default
     "offer_notes": true,          # list recent notes after /clear, compaction and at startup
     "offer_max": 8, "offer_days": 14,
     "embeddings": false}         # the embedding layer; `taskmark fetch-model` sets it

Colour is off by default on purpose: the hint text is rendered by Claude Code, and whether it
passes ANSI colour through is not documented. Turn it on and look; if escape codes show up as
garbage, set it back to null.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config

DEFAULTS = {
    "hint_min_context": 150_000,
    "hint_color": None,
    "offer_notes": True,
    "offer_max": 8,
    "offer_days": 14,
    # ask Claude itself to confirm or correct each boundary decision (a short line of context
    # per prompt; Claude answers only when it disagrees, by running `taskmark mark ...`)
    "model_marks": True,
    # the embedding layer (second opinion on topic changes). Off by default: it needs
    # sentence-transformers, a ~450 MB model and ~1 GB of RAM for its worker, and on real
    # sessions it decided about one boundary in five days. `taskmark fetch-model` turns it on.
    "embeddings": False,
}

_ANSI = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "magenta": 35, "cyan": 36,
         "white": 37, "gray": 90, "grey": 90, "bright-yellow": 93, "bright-cyan": 96}

SETTINGS_PATH = config.HOME_DIR / "settings.json"

# Сообщения, которые нужно показать ЧЕЛОВЕКУ. Их кладёт `taskmark mark …` — обычный вызов
# CLI, чей stdout читает только модель. Ближайший хук UserPromptSubmit выгребает очередь
# и отдаёт её как systemMessage, поэтому смена задачи доходит с задержкой в один ход.
PENDING_PATH = config.HOME_DIR / "pending_messages.json"


def queue_message(text: str) -> None:
    """Отложить строку до ближайшего хука. Никогда не роняет вызывающего."""
    try:
        config.ensure_dirs()
        cur = []
        if PENDING_PATH.exists():
            cur = json.loads(PENDING_PATH.read_text(encoding="utf-8") or "[]")
        cur.append(text)
        PENDING_PATH.write_text(json.dumps(cur[-10:], ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def take_messages() -> list[str]:
    """Забрать и очистить очередь."""
    try:
        if not PENDING_PATH.exists():
            return []
        out = json.loads(PENDING_PATH.read_text(encoding="utf-8") or "[]")
        PENDING_PATH.unlink(missing_ok=True)
        return [x for x in out if x]
    except Exception:
        return []


def settings() -> dict:
    out = dict(DEFAULTS)
    try:
        user = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(user, dict):
            out.update({k: v for k, v in user.items() if k in DEFAULTS})
    except Exception:
        pass
    return out


def set_setting(key: str, value) -> None:
    """Write one key into settings.json, keeping everything else in it."""
    config.ensure_dirs()
    cur = {}
    try:
        cur = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(cur, dict):
            cur = {}
    except Exception:
        pass
    cur[key] = value
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SETTINGS_PATH)


def colorize(text: str, color: str | None) -> str:
    code = _ANSI.get((color or "").strip().lower())
    return f"\x1b[{code}m{text}\x1b[0m" if code else text


def last_context(con, session_id: str) -> int:
    """Context size of the session's latest main-chain turn (what the next request carries)."""
    row = con.execute("SELECT context_size FROM turns WHERE session_id=? AND is_sidechain=0 "
                      "ORDER BY ts DESC LIMIT 1", (session_id,)).fetchone()
    return int(row["context_size"]) if row else 0


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.0f}k" if n < 1_000_000 else f"{n / 1_000_000:.1f}M"


def boundary_hint(note: str | None, context_tokens: int, started_title: str | None = None) -> str | None:
    """One line for the user when a task was closed while the context is large.

    `started_title` is set when the boundary was placed one prompt back: the new task has
    already done a turn in this context. /clear is still safe — SessionEnd writes the open task
    its own note and SessionStart offers it — but the user has to know to pick it up."""
    s = settings()
    if not note or context_tokens < int(s["hint_min_context"] or 0):
        return None
    if started_title:
        text = (f"[taskmark] previous task closed and saved · context {_fmt_tokens(context_tokens)} "
                f"· the new task «{started_title[:50]}» began one message ago; /clear saves it "
                f"as its own note too — after clearing, ask to continue it")
    else:
        text = (f"[taskmark] previous task closed and saved · context {_fmt_tokens(context_tokens)} "
                f"· /clear is safe now, the research is kept")
    return colorize(text, s["hint_color"])


def recent_notes(cwd: str) -> list[tuple[Path, str]]:
    s = settings()
    base = Path(cwd or ".") / config.RESEARCH_SUBDIR
    if not base.is_dir():
        return []
    cutoff = time.time() - float(s["offer_days"]) * 86400
    notes = [p for p in base.glob("*.md") if p.stat().st_mtime >= cutoff]
    notes.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)   # name = date prefix
    out = []
    for p in notes[: int(s["offer_max"])]:
        title = ""
        try:
            with open(p, encoding="utf-8") as fh:
                first = fh.readline().strip()
            title = first.lstrip("#").strip()
        except Exception:
            pass
        out.append((p, title[:100]))
    return out


NOTE_RESTORE_MAX_CHARS = 12000     # ~3k tokens; a skeleton note is usually 3-6 KB


def current_note_context(note_path: str | None) -> str | None:
    """After a compaction: the full text of the current task's note, as written by pre-compact."""
    if not note_path or not settings()["offer_notes"]:
        return None
    try:
        text = Path(note_path).read_text(encoding="utf-8")
    except Exception:
        return None
    if len(text) > NOTE_RESTORE_MAX_CHARS:
        text = text[:NOTE_RESTORE_MAX_CHARS] + "\n… (truncated — the full note is at the path above)"
    return ("taskmark: the conversation was just compacted. This is the research note of the task "
            f"in progress, written right before the compaction ({note_path}). It keeps the files, "
            "commands and failures the summary may have dropped — use it, don't re-investigate:\n\n"
            + text)


def notes_context(cwd: str, source: str, skip: str | None = None) -> str | None:
    """Text added to a fresh context: where the earlier research of this project lives.
    `skip` is a note already restored in full — no point listing it again."""
    s = settings()
    if not s["offer_notes"] or source not in ("clear", "compact", "startup"):
        return None
    notes = [(p, t) for p, t in recent_notes(cwd) if not skip or str(p) != skip]
    if not notes:
        return None
    lines = ["taskmark: research notes from earlier work in this project (newest first). "
             "They record files, commands, what failed and what was decided. Read the relevant "
             "one before re-investigating something it may already cover; do not read them all "
             "up front."]
    for p, title in notes:
        lines.append(f"- {p}" + (f" — {title}" if title else ""))
    return "\n".join(lines)


# --- when it is worth re-tuning -------------------------------------------
# The plugin cannot call a model: hooks run outside the conversation. What it can do is
# notice that enough corrections have piled up and say so once, in the session that is
# already open — the model is right there, and /task-tune hands it the data.
TUNE_COOLDOWN_SECONDS = 3 * 24 * 3600


def tune_hint(con) -> str | None:
    """One line suggesting /task-tune, or None. Silent unless there is something to learn."""
    try:
        row = con.execute(
            "SELECT COUNT(*) n, SUM(label='new') news, SUM(label='same') sames "
            "FROM prompt_signals WHERE label IS NOT NULL").fetchone()
        if not row or not row["n"]:
            return None
        n, news, sames = int(row["n"]), int(row["news"] or 0), int(row["sames"] or 0)
        if n < config.SUPERVISED_MIN_LABELS:
            return None
        if min(news, sames) < config.SUPERVISED_MIN_PER_CLASS:
            return None          # all corrections point one way — nothing to trade off yet
        last = con.execute("SELECT MAX(ts) ts FROM events WHERE kind='tune-offer'").fetchone()
        if last and last["ts"] and (time.time() - float(last["ts"])) < TUNE_COOLDOWN_SECONDS:
            return None
        con.execute("INSERT INTO events (ts, session_id, task_id, kind, payload) "
                    "VALUES (?,?,?,?,?)", (time.time(), None, None, "tune-offer",
                                           json.dumps({"labelled": n})))
        wrong = con.execute("SELECT COUNT(*) n FROM prompt_signals "
                            "WHERE label='same' AND rule_boundary=1").fetchone()["n"]
        missed = con.execute("SELECT COUNT(*) n FROM prompt_signals "
                             "WHERE label='new' AND rule_boundary=0").fetchone()["n"]
        return (f"[taskmark] {n} corrections collected "
                f"({wrong} wrong splits, {missed} missed) — /task-tune re-fits the "
                f"thresholds to how you work")
    except Exception:
        return None
