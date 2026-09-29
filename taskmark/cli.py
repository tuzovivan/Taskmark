"""Entry point for hooks and for the handful of manual commands.

    python3 -m taskmark.cli hook <event>   # called by hooks.json, hook JSON on stdin
    python3 -m taskmark.cli split          # force a task boundary at the next prompt
    python3 -m taskmark.cli note           # write the research note for the open task now
    python3 -m taskmark.cli ingest         # import existing transcripts into the DB
    python3 -m taskmark.cli status         # what is recorded so far
    python3 -m taskmark.cli embed-daemon   # resident embedding worker (started automatically)
    python3 -m taskmark.cli ingest --mark   # also split imported history into tasks
    python3 -m taskmark.cli report         # metrics for a period + heaviest tasks
    python3 -m taskmark.cli cost           # cost by day, rates from the LiteLLM table
    python3 -m taskmark.cli same           # correction: that was the same task, merge it back
    python3 -m taskmark.cli git [<sha>]    # repo state per task; which task made a commit
    python3 -m taskmark.cli decide "<what>" --why "<why>" [--instead ...] [--who user|claude]
    python3 -m taskmark.cli decisions [--days=90] [text]   # the journal, newest first
    python3 -m taskmark.cli mark new|same|resume|name ["topic"] --session ID
                                           # Claude's own verdict on the prompt it answers
    python3 -m taskmark.cli calibrate      # show/recompute thresholds from your own data
    python3 -m taskmark.cli off / on       # kill switch: hooks become no-ops

Hooks must never break the session: every failure is swallowed and logged to
~/.claude/taskmark/taskmark.log.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from . import (boundary, calibrate, config, db, embed, hints, pricing, report, research,
               state as st, transcript, vcs)


LOG_MAX_BYTES = 1 << 20          # a plugin's error log must not quietly become a problem


def _log(msg: str) -> None:
    try:
        config.ensure_dirs()
        if config.LOG_PATH.exists() and config.LOG_PATH.stat().st_size > LOG_MAX_BYTES:
            config.LOG_PATH.replace(config.LOG_PATH.with_suffix(".log.1"))   # keep one old file
        with open(config.LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%F %T')} {msg}\n")
    except Exception:
        pass


def _read_hook_input() -> dict:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def _git_branch(cwd: str) -> str | None:
    try:
        import subprocess
        r = subprocess.run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
                           capture_output=True, text=True, timeout=3)
        out = (r.stdout or "").strip()
        return out or None
    except Exception:
        return None


def _title_of(prompt: str) -> str:
    line = (prompt or "").strip().splitlines()[0] if prompt.strip() else "task"
    return line[:200]


# --------------------------------------------------------------------------- ingest
def ingest_transcript(con, path: str, session_id: str, task_id: int | None) -> int:
    """Pull new turns out of a transcript into SQLite. Idempotent (offset + request_id)."""
    if not path:
        return 0
    offset = db.get_ingest_offset(con, path)
    act = transcript.read(path, offset=offset)
    rows = [t.__dict__ for t in transcript.dedupe(act.turns)]
    since = None
    if task_id:
        task = db.get_task(con, task_id)
        if task:
            since = float(task["started_at"])
    saved = db.save_turns(con, rows, task_id, since_ts=since)
    db.set_ingest_offset(con, path, act.offset)
    return saved


def close_task(con, state: dict, reason: str, write_note: bool = True,
               until: float | None = None) -> str | None:
    """Finish the open task: ingest its turns, write the research note, mark it ended.
    `until` ends it at an earlier moment — the boundary belongs to an earlier prompt."""
    task_id = state.get("task_id")
    if not task_id:
        return None
    task = db.get_task(con, task_id)
    if not task:
        return None
    path = state.get("transcript_path") or ""
    ingest_transcript(con, path, state.get("session_id", ""), task_id)
    git = close_git(con, task_id)
    recorded = [dict(r) for r in db.decisions_of(con, task_id)]
    note = None
    if write_note and path:
        since = float(task["started_at"]) - 1
        spans, own = db.task_spans(con, task["session_id"], task_id, since, until)
        act = transcript.read(path, offset=0, since_ts=since, until_ts=until,
                              spans=spans or None, own_requests=own if spans else None)
        if act.prompts or act.files or act.commands:
            try:
                note = str(research.write_note(
                    cwd=task["cwd"] or os.getcwd(), title=task["title"] or "task",
                    task_id=task_id, session_id=task["session_id"],
                    started_at=float(task["started_at"]), ended_at=until or time.time(),
                    act=act, reason=reason, git=git, recorded=recorded,
                    existing=task["note_path"]))
            except Exception as e:                      # noqa — a note is never worth a crash
                _log(f"note failed: {e}")
    db.end_task(con, task_id, note, ended_at=until)
    db.log_event(con, "boundary", state.get("session_id"), task_id,
                 json.dumps({"reason": reason, "note": note}, ensure_ascii=False))
    return note


def close_orphans(con, session_id: str, now: float) -> list[int]:
    """Close tasks left open by sessions that ended without SessionEnd (killed window,
    tmux, crash): write their note as session-end would have. A session counts as gone
    when it has been quiet for ORPHAN_IDLE_SECONDS and none of its hooks holds its lock."""
    rows = con.execute(
        "SELECT id, session_id, started_at FROM tasks WHERE ended_at IS NULL "
        "AND merged_into IS NULL AND session_id != ? AND started_at > ? ORDER BY started_at",
        (session_id, now - config.ORPHAN_MAX_AGE_DAYS * 86400)).fetchall()
    closed: list[int] = []
    for r in rows:
        if len(closed) >= config.ORPHAN_PER_START:
            break
        sid, tid = r["session_id"], int(r["id"])
        if not st.try_lock(sid):                      # a hook of that session is running
            continue
        other = st.load(sid)
        other["session_id"] = sid
        tpath = other.get("transcript_path") or ""
        last_turn = con.execute("SELECT MAX(ts) ts FROM turns WHERE task_id=?",
                                (tid,)).fetchone()["ts"]
        last = max(float(other.get("last_activity_ts") or 0), float(last_turn or 0),
                   float(r["started_at"]))
        seen = max(last, os.path.getmtime(tpath) if tpath and os.path.exists(tpath) else 0)
        if now - seen < config.ORPHAN_IDLE_SECONDS:
            continue                                  # may still be open in another window
        if other.get("task_id") == tid:
            close_task(con, other, "orphan", write_note=True, until=last)
            other["task_id"] = None                   # a later --resume starts a fresh task
            st.save(other)
        else:
            # the session had moved on but the task stayed open (a lost state write): its
            # note, if any, is already written — only end it where its turns stopped
            db.end_task(con, tid, None, ended_at=float(last_turn or r["started_at"]))
            db.log_event(con, "boundary", sid, tid,
                         json.dumps({"reason": "orphan-stale", "note": None}))
        closed.append(tid)
    return closed


def note_repos(con, state: dict, paths: list[str], task_id: int | None, now: float) -> None:
    """First time a task touches a repository, remember where that repository stood."""
    if not task_id:
        return
    seen = dict(state.get("task_repos") or {})
    for p in paths:
        root = transcript.project_root(p)               # cheap and cached; git is asked once
        if not root or root in seen or not os.path.isdir(os.path.join(root, ".git")):
            continue
        repo = vcs.repo_of(root) or root
        if repo in seen:
            continue
        seen[repo] = vcs.head(repo) or ""
        db.git_start(con, task_id, repo, vcs.branch(repo), seen[repo] or None, now)
    state["task_repos"] = seen


def close_git(con, task_id: int) -> list[dict]:
    """Where each repository of the task ended up, and what it gained meanwhile."""
    out: list[dict] = []
    for r in db.git_rows(con, task_id):
        repo = r["repo"]
        end = vcs.head(repo)
        commits = vcs.commits_between(repo, r["head_start"], end)
        db.git_finish(con, task_id, repo, end, vcs.is_dirty(repo), len(commits))
        out.append({"repo": repo, "branch": r["branch"], "start": r["head_start"],
                    "end": end, "commits": commits, "dirty": vcs.is_dirty(repo)})
    return out


def open_task(con, state: dict, *, cwd: str, title: str, reason: str, started_at: float,
              branch: str | None = None, calib_version: int = 0, prompts: int = 1,
              carry_files: list | None = None, carry_targets: list | None = None) -> int:
    task_id = db.start_task(con, state.get("session_id", ""), cwd, title,
                            research.slugify(title), reason, started_at=started_at,
                            calib_version=calib_version)
    if prompts != 1:
        con.execute("UPDATE tasks SET prompts=? WHERE id=?", (prompts, task_id))
    if carry_files:
        db.touch_files(con, task_id, carry_files)
    state.update({"task_id": task_id, "cwd": cwd, "prompts": prompts,
                  "task_files": list(carry_files or []), "window_files": [],
                  "task_targets": list(carry_targets or []), "window_targets": [],
                  "manual_split": False, "task_started": started_at, "task_vocab": [],
                  "task_repos": {},
                  "prompt_gaps": [], "git_branch": branch, "last_prompt_ts": None,
                  "prompt_vectors": []})
    return task_id


_SNAPSHOT_KEYS = ("cwd", "prompts", "task_files", "task_targets", "task_vocab", "task_started",
                  "prompt_gaps", "git_branch")


def move_turns(con, session_id: str, src: int, dst: int, since_ts: float) -> None:
    """Turns recorded under the old task after the boundary moment belong to the new one."""
    con.execute("UPDATE turns SET task_id=? WHERE session_id=? AND task_id=? AND ts>=?",
                (dst, session_id, src, since_ts - 1))


def _bin() -> str:
    root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    path = os.path.join(root, "bin", "taskmark") if root else os.path.abspath(sys.argv[0])
    return f'"{path}"' if " " in path else path


def _other_titles(con, session_id: str, task_id: int | None, n: int = 3) -> list[str]:
    rows = con.execute("SELECT id, title FROM tasks WHERE session_id=? AND id<>? AND "
                       "merged_into IS NULL ORDER BY id DESC LIMIT ?",
                       (session_id, task_id or 0, n)).fetchall()
    return [f"#{r['id']} «{(r['title'] or '')[:50]}»" for r in rows]


def model_context(con, state: dict, decision: dict) -> str | None:
    """The second opinion. taskmark tells Claude what it decided for this prompt; Claude, who
    sees the whole conversation, answers only when it disagrees — by running `taskmark mark`.
    Silence means agreement, so a continuation costs nothing."""
    if not hints.settings().get("model_marks"):
        return None
    sid = state.get("session_id", "")
    tid = state.get("task_id")
    row = db.get_task(con, tid) if tid else None
    title = (row["title"] if row else "")[:60]
    cmd = f"{_bin()} mark"
    ses = f"--session {sid}"
    dec = (f" If a decision was made here that changes behaviour or architecture, record it: "
           f"`{_bin()} decide \"<what>\" --why \"<why>\" {ses}` (not for small choices).")
    if decision.get("boundary") and not decision.get("retro"):
        prev = db.get_task(con, decision.get("prev_task") or 0)
        prev_t = (prev["title"] if prev else "")[:60]
        return (f"[taskmark] Task tracking: a NEW task was opened at this message "
                f"(signals: {decision.get('reason')}); previous task: «{prev_t}». "
                f"If this message actually continues it, run first: `{cmd} same {ses}`. "
                f"If you agree, name the new task in 3-6 words: `{cmd} name \"<topic>\" {ses}`. "
                f"Bookkeeping only — it needs no place in your reply.{dec}")
    others = _other_titles(con, sid, tid)
    back = (f" If it returns to an earlier task ({', '.join(others)}): "
            f"`{cmd} resume \"<topic or #id>\" {ses}`." if others else "")
    moved = ""
    if decision.get("retro"):
        prev = db.get_task(con, decision.get("prev_task") or 0)
        moved = (f" (opened at the previous message, after «{(prev['title'] if prev else '')[:50]}»; "
                 f"if that was still the same work, run `{cmd} same {ses}`; if not, name it: "
                 f"`{cmd} name \"<topic>\" {ses}`)")
    return (f"[taskmark] Task tracking: current task «{title}»{moved}. Silence = this message "
            f"continues it. If it starts a different task, run first: "
            f"`{cmd} new \"<topic in 3-6 words>\" {ses}`.{back} "
            f"Bookkeeping only — it needs no place in your reply.{dec}")


# --------------------------------------------------------------------------- hooks
def hook(event: str) -> int:
    if config.is_disabled():
        return 0
    data = _read_hook_input()
    session_id = data.get("session_id") or "unknown"
    # hook input "cwd" follows every `cd` in Bash; notes and the cwd signal belong to the
    # project the session was started in, so prefer CLAUDE_PROJECT_DIR
    shell_cwd = data.get("cwd") or os.getcwd()
    cwd = os.environ.get("CLAUDE_PROJECT_DIR") or shell_cwd
    tpath = data.get("transcript_path") or ""
    state = st.load(session_id)
    state["session_id"] = session_id
    if tpath:
        state["transcript_path"] = tpath
    now = time.time()
    out_lines: list[str] = []
    extra_context: str | None = None

    with db.connect() as con:
        if event == "session-start":
            state.setdefault("task_id", None)
            state["cwd"] = cwd
            source = data.get("source") or ""
            db.log_event(con, "session-start", session_id, None, source)
            try:
                close_orphans(con, session_id, now)
            except Exception as e:                      # noqa — never worth a broken start
                _log(f"close_orphans failed: {e}")
            # after a compaction the task goes on: put its own note (written by pre-compact)
            # straight back into the context, so the summary's losses are covered
            current, restored = None, None
            if source == "compact" and state.get("task_id"):
                task = db.get_task(con, state["task_id"])
                restored = task["note_path"] if task else None
                current = hints.current_note_context(restored)
                if current:
                    db.log_event(con, "note-restored", session_id, state["task_id"], restored)
            # after /clear or a compaction Claude knows nothing about the notes — tell it where
            # they are (paths and titles only), so clearing stops being a loss
            extra_context = hints.notes_context(cwd, source, skip=restored if current else None)
            if extra_context:
                db.log_event(con, "notes-offered", session_id, None,
                             json.dumps({"source": source,
                                         "count": extra_context.count("\n- ")}))
            if current:
                extra_context = current + ("\n\n" + extra_context if extra_context else "")

        elif event == "prompt":
            # Claude Code sends `user_prompt`; older builds/tests send `prompt`.
            prompt = data.get("user_prompt") or data.get("prompt") or ""
            branch = _git_branch(cwd)
            cal = calibrate.current(con)
            samples: dict = {}
            is_boundary, reason = boundary.decide(prompt=prompt, cwd=cwd, now=now, state=state,
                                                  git_branch=branch, thresholds=cal,
                                                  samples=samples)
            for kind, value in samples.items():       # feed the calibrator, labelled by outcome
                calibrate.add_sample(con, session_id, kind, value, is_boundary)
            # ...and the whole decision in one row, so a correction can be attached to it
            samples["prompt_terms"] = len(boundary.tokens(prompt))
            db.add_prompt_signals(con, session_id=session_id, task_id=state.get("task_id"),
                                  prompt=prompt, signals=samples, reason=reason,
                                  boundary=is_boundary, ts=now)
            prev_task = state.get("task_id")
            prev_ts = float(state.get("last_prompt_ts") or 0)
            # only last-turn signals fired (files/target) → the switch was at the previous
            # prompt: end the old task there and start the new one with that prompt
            retro = bool(is_boundary and prev_task and prev_ts and boundary.turn_based(reason))
            snapshot = {k: state.get(k) for k in _SNAPSHOT_KEYS} if is_boundary else None
            if is_boundary:
                if retro:
                    note = close_task(con, state, reason, until=prev_ts)
                    task_id = open_task(con, state, cwd=cwd, reason=reason, started_at=prev_ts,
                                        title=_title_of(state.get("last_prompt_text") or prompt),
                                        branch=branch, calib_version=int(cal.get("version") or 0),
                                        prompts=2, carry_files=state.get("window_files"),
                                        carry_targets=state.get("window_targets"))
                    move_turns(con, session_id, prev_task, task_id, prev_ts)
                    state["last_prompt_ts"] = prev_ts        # keeps the rhythm (gap) right
                else:
                    note = close_task(con, state, reason)
                    task_id = open_task(con, state, cwd=cwd, title=_title_of(prompt),
                                        reason=reason, started_at=now, branch=branch,
                                        calib_version=int(cal.get("version") or 0))
                if not os.environ.get("TASKMARK_QUIET"):
                    row = db.get_task(con, state["task_id"]) if retro else None
                    hint = hints.boundary_hint(note, hints.last_context(con, session_id),
                                               started_title=row["title"] if row else None)
                    if hint:
                        out_lines.append(hint)
            else:
                state["prompts"] = int(state.get("prompts") or 0) + 1
                state["window_files"] = []
                state["task_targets"] = sorted(set(state.get("task_targets") or [])
                                               | set(state.get("window_targets") or []))[:50]
                state["window_targets"] = []
                con.execute("UPDATE tasks SET prompts=prompts+1 WHERE id=?", (state["task_id"],))
            state["last_prompt_text"] = prompt[:300]
            state["last_decision"] = {"ts": now, "boundary": is_boundary, "retro": retro,
                                      "reason": reason, "prev_task": prev_task,
                                      "prev_state": snapshot}
            if not prompt.lstrip().startswith("<"):   # task notifications, command wrappers
                extra_context = model_context(con, state, state["last_decision"])
            st.note_prompt(state, prompt, now, boundary.tokens(prompt))
            state["tools_since_prompt"] = 0
            # keep the task centroid up to date (cheap: one short encode per prompt)
            if embed.available():
                v = embed.encode_one(prompt[:config.EMBED_MAX_PROMPT_CHARS])
                if v:
                    vecs = list(state.get("prompt_vectors") or [])
                    vecs.append(v)
                    state["prompt_vectors"] = vecs[-config.CENTROID_PROMPTS:]
            # calibration is recomputed rarely and on its own schedule
            if calibrate.compute(con) is not None:
                db.log_event(con, "calibration", session_id, None,
                             calibrate.describe(calibrate.current(con)))
            db.log_event(con, "prompt", session_id, state.get("task_id"),
                         json.dumps({"boundary": is_boundary, "reason": reason}, ensure_ascii=False))
            state["last_activity_ts"] = now

        elif event == "tool":
            tool_name = data.get("tool_name") or ""
            tool_input = data.get("tool_input") or {}
            if tool_name == "Bash" and re.search(r"bin/taskmark\S*\s+mark\b",
                                                 str(tool_input.get("command") or "")):
                # Claude's own verdict is bookkeeping, not work: it must not look like a
                # turn spent in the plugin's directory (that split the task again)
                st.save(state)
                return 0
            paths = transcript.paths_from_tool(tool_name, tool_input, shell_cwd)
            targets = transcript.targets_from_tool(tool_name, tool_input, shell_cwd)
            if targets:
                st.add_window_targets(state, targets)
            # the one-month test of the whole idea: are the notes ever read back?
            # (Claude reads them with Read as often as with cat/sed/grep in Bash — count both)
            if tool_name == "Read":
                fp = str(tool_input.get("file_path") or "").replace("\\", "/")
                if "/.claude/research/" in fp:
                    db.log_event(con, "note-read", session_id, state.get("task_id"), fp)
            elif tool_name == "Bash":
                cmd = str(tool_input.get("command") or "").replace("\\", "/")
                for fp in re.findall(r"[^\s'\"]*\.claude/research/[^\s'\";|&)]+\.md", cmd):
                    db.log_event(con, "note-read", session_id, state.get("task_id"), fp)
            state["tools_since_prompt"] = int(state.get("tools_since_prompt") or 0) + 1
            if paths:
                note_repos(con, state, paths, state.get("task_id"), now)
                st.add_window_files(state, paths)
                # file names are part of the task vocabulary: "parser.py" keeps "parser" alive
                vocab = set(state.get("task_vocab") or []) | boundary.path_tokens(paths)
                state["task_vocab"] = sorted(vocab)[:config.VOCAB_CAP]
                task_id = state.get("task_id")
                if task_id:
                    db.touch_files(con, task_id, paths)
                    owner = boundary.check_return(
                        con=con, session_id=session_id, task_id=task_id,
                        prompts=int(state.get("prompts") or 0), files=paths, now=now, db=db)
                    if owner:
                        db.merge_task(con, task_id, owner)
                        db.log_event(con, "merge", session_id, owner,
                                     json.dumps({"from": task_id}))
                        row = db.get_task(con, owner)
                        state["task_id"] = owner
                        state["cwd"] = (row["cwd"] if row else cwd)
                        state["task_started"] = float(row["started_at"]) if row else now
            state["last_activity_ts"] = now

        elif event in ("stop", "session-end", "pre-compact"):
            ingest_transcript(con, tpath, session_id, state.get("task_id"))
            state["last_activity_ts"] = now
            if event == "pre-compact":
                # the whole point of the plugin: never lose the research to a compaction
                note = close_task(con, state, "pre-compact", write_note=True)
                if note:
                    db.log_event(con, "note", session_id, state.get("task_id"), note)
                    if not os.environ.get("TASKMARK_QUIET"):
                        out_lines.append(hints.colorize(f"[taskmark] research saved to {note}",
                                                        hints.settings()["hint_color"]))
            tune = hints.tune_hint(con)      # enough corrections to learn from? say so once
            if tune:
                out_lines.append(hints.colorize(tune, hints.settings()["hint_color"]))
            # not `elif`: on the one session-end that showed the tune hint the task stayed open
            if event == "session-end":
                close_task(con, state, "session-end", write_note=True)
                state["task_id"] = None

    st.save(state)
    if event == "prompt":
        # Строки, отложенные `taskmark mark …`: их stdout читает модель, а не человек,
        # поэтому показать их можно только отсюда — из хука.
        out_lines = list(hints.take_messages()) + out_lines
    if extra_context and event == "session-start":
        # documented SessionStart contract: additionalContext is added to Claude's context
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart",
                                                 "additionalContext": extra_context},
                          "suppressOutput": True}, ensure_ascii=False))
    elif event == "prompt" and (extra_context or out_lines):
        out: dict = {}
        if extra_context:
            out["hookSpecificOutput"] = {"hookEventName": "UserPromptSubmit",
                                         "additionalContext": extra_context}
        if out_lines:
            out["systemMessage"] = " ".join(out_lines)
        # suppressOutput скрывает не только stdout: оставляем его лишь когда человеку
        # показывать нечего, иначе сообщение о смене задачи снова станет невидимым.
        if "systemMessage" not in out:
            out["suppressOutput"] = True
        print(json.dumps(out, ensure_ascii=False))
    elif out_lines and event in ("prompt", "pre-compact"):
        # documented hook output contract: JSON with systemMessage
        print(json.dumps({"systemMessage": " ".join(out_lines), "suppressOutput": True},
                         ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------- commands
def cmd_split(argv: list[str]) -> int:
    sid = argv[0] if argv else _latest_session()
    state = st.load(sid)
    state["session_id"] = sid
    state["manual_split"] = True
    st.save(state)
    print("taskmark: the next prompt starts a new task")
    return 0


def _find_task(con, session_id: str, not_task: int | None, topic: str) -> int | None:
    """An earlier task of this session by '#id' or by what its title is about."""
    m = re.fullmatch(r"#?(\d+)", topic.strip())
    rows = con.execute("SELECT id, title FROM tasks WHERE session_id=? AND id<>? AND "
                       "merged_into IS NULL ORDER BY id DESC LIMIT 30",
                       (session_id, not_task or 0)).fetchall()
    if m:
        return next((int(r["id"]) for r in rows if int(r["id"]) == int(m.group(1))), None)
    # words compared by their first letters: "нагрузка" and "нагрузку", "parser" and "parsers"
    stems = lambda t: {w[:5] for w in boundary.tokens(t) if len(w) > 2}
    words = stems(topic)
    tv = embed.encode_one(topic) if embed.available() else None
    best, best_score = None, 0.0
    for r in rows:
        title = r["title"] or ""
        score = len(words & stems(title)) / len(words) if words else 0.0
        if tv:
            score = max(score, embed.cosine(tv, embed.encode_one(title)) or 0.0)
        if score > best_score:
            best, best_score = int(r["id"]), score
    return best if best_score >= config.RESUME_MIN_SCORE else None


def cmd_mark(argv: list[str]) -> int:
    """Claude's verdict on the prompt it is answering: `new "<topic>"`, `same`, `resume "<topic>"`.
    taskmark already decided on its own (see model_context); this confirms or corrects it."""
    verdict = argv[0] if argv else ""
    if verdict not in ("new", "same", "resume", "name"):
        print('usage: taskmark mark new|same|resume|name ["topic"] [--session ID]')
        return 2
    rest, sid = argv[1:], None
    if "--session" in rest:
        i = rest.index("--session")
        sid = rest[i + 1] if i + 1 < len(rest) else None
        rest = rest[:i] + rest[i + 2:]
    topic = " ".join(rest).strip().strip('"')[:120]
    sid = sid or _latest_session()
    state = st.load(sid)
    state["session_id"] = sid
    dec = state.get("last_decision") or {}
    tid = state.get("task_id")
    ts = float(state.get("last_prompt_ts") or time.time())      # the prompt being answered
    local_new = bool(dec.get("boundary")) and not dec.get("retro")   # opened AT this prompt
    local_any = bool(dec.get("boundary"))                            # ...or at the previous one
    agree = ((verdict == "new") == local_new if verdict == "new"
             else (verdict == "same") != local_any if verdict == "same" else False)
    msg = ""
    with db.connect() as con:
        if verdict == "name":                              # agreed — only a better title
            if tid and topic:
                con.execute("UPDATE tasks SET title=?, slug=? WHERE id=?",
                            (topic, research.slugify(topic), tid))
            db.log_event(con, "model-mark", sid, tid, json.dumps(
                {"verdict": "name", "topic": topic, "local": local_any, "agree": True},
                ensure_ascii=False))
            print(f"taskmark: task #{tid} «{topic}»")
            return 0
        target = _find_task(con, sid, tid, topic) if verdict == "resume" and topic else None
        if verdict == "resume" and target is None:
            verdict = "new"                               # nothing to return to: a new task
        if verdict == "new":
            if local_new and tid:                          # agreed — Claude only names it
                if topic:
                    con.execute("UPDATE tasks SET title=?, slug=? WHERE id=?",
                                (topic, research.slugify(topic), tid))
                msg = f"taskmark: task #{tid} «{topic}»"
            else:
                close_task(con, state, "model", until=ts)
                new = open_task(con, state, cwd=state.get("cwd") or os.getcwd(),
                                title=topic or _title_of(state.get("last_prompt_text") or ""),
                                reason="model", started_at=ts,
                                carry_files=state.get("window_files"),
                                carry_targets=state.get("window_targets"))
                if tid:
                    move_turns(con, sid, tid, new, ts)
                state["last_prompt_ts"] = ts
                state["task_vocab"] = sorted(boundary.tokens(state.get("last_prompt_text") or ""))
                db.label_last_prompt(con, sid, "new", "model")
                msg = f"taskmark: new task #{new} «{topic}»"
        elif verdict == "same":
            prev = dec.get("prev_task")
            if local_any and prev and tid and prev != tid:
                db.merge_task(con, tid, int(prev))
                calibrate.feedback(con, "same-task", tid, sid)
                db.label_task_prompts(con, int(tid), "same", "merge")
                db.label_last_prompt(con, sid, "same", "model")
                # what the wrongly split part touched is now part of the task: a second turn
                # on the same server must not look like a new place again
                carried_t = set(state.get("task_targets") or []) | set(state.get("window_targets") or [])
                carried_f = set(state.get("task_files") or [])
                state.update({k: v for k, v in (dec.get("prev_state") or {}).items()})
                state["task_targets"] = sorted(set(state.get("task_targets") or []) | carried_t)
                state["task_files"] = sorted(set(state.get("task_files") or []) | carried_f)
                state["task_id"] = int(prev)
                state["prompts"] = int(state.get("prompts") or 0) + 1
                con.execute("UPDATE tasks SET prompts=prompts+1 WHERE id=?", (int(prev),))
                msg = f"taskmark: kept in task #{prev}"
            else:
                msg = "taskmark: ok, same task"
        else:                                              # resume an earlier task
            if local_new and tid:
                db.merge_task(con, tid, target)
            else:
                close_task(con, state, "model-resume", until=ts)
                con.execute("UPDATE tasks SET ended_at=NULL, prompts=prompts+1 WHERE id=?",
                            (target,))
                if tid:
                    move_turns(con, sid, tid, target, ts)
            row = db.get_task(con, target)
            state.update({"task_id": target, "task_started": float(row["started_at"]),
                          "prompts": int(row["prompts"] or 1), "window_files": [],
                          "window_targets": [], "task_files": sorted(db.task_file_set(con, target)),
                          "task_targets": [], "task_vocab": sorted(boundary.tokens(row["title"] or "")),
                          "prompt_vectors": []})
            msg = f"taskmark: back to task #{target} «{row['title']}»"
        db.log_event(con, "model-mark", sid, state.get("task_id"),
                     json.dumps({"verdict": verdict, "topic": topic, "local": local_new,
                                 "local_retro": bool(dec.get("retro")),
                                 "local_reason": dec.get("reason"), "agree": agree},
                                ensure_ascii=False))
    # the verdict is spent: a second `mark` for the same prompt must not act on stale data
    state["last_decision"] = {**dec, "boundary": state.get("task_id") != tid or local_new,
                              "retro": False, "prev_task": dec.get("prev_task")}
    st.save(state)
    # Эта строка печатается в stdout, который читает модель, а не человек. Чтобы смена
    # задачи доходила до пользователя, кладём её в очередь — ближайший хук UserPromptSubmit
    # отдаст её как systemMessage.
    if not os.environ.get("TASKMARK_QUIET") and verdict in ("new", "resume"):
        hints.queue_message(hints.colorize("[taskmark] " + msg.replace("taskmark: ", ""),
                                           hints.settings()["hint_color"]))
    print(msg)
    return 0


def _opt(argv: list[str], name: str) -> tuple[list[str], str | None]:
    if name not in argv:
        return argv, None
    i = argv.index(name)
    val = argv[i + 1] if i + 1 < len(argv) else None
    return argv[:i] + argv[i + 2:], val


def cmd_decide(argv: list[str]) -> int:
    """`taskmark decide "<what>" --why "<why>" [--instead "<rejected>"] [--who user|claude]`.

    For decisions that change behaviour or architecture — not for every small choice. Written
    by whoever decided, at the moment of deciding, so the journal stays true and short."""
    argv, sid = _opt(argv, "--session")
    argv, why = _opt(argv, "--why")
    argv, instead = _opt(argv, "--instead")
    argv, who = _opt(argv, "--who")
    what = " ".join(argv).strip().strip('"')
    if not what:
        print('usage: taskmark decide "<what was decided>" --why "<why>" '
              '[--instead "<rejected option>"] [--who user|claude] [--session ID]')
        return 2
    sid = sid or _latest_session()
    state = st.load(sid)
    repos = list((state.get("task_repos") or {}).keys())
    repo = repos[0] if repos else None
    with db.connect() as con:
        did = db.add_decision(con, session_id=sid, task_id=state.get("task_id"),
                              who=(who or "claude"), what=what[:500], why=(why or "")[:1000] or None,
                              instead=(instead or "")[:500] or None, repo=repo,
                              head=vcs.head(repo) if repo else None, ts=time.time())
    print(f"taskmark: decision #{did} recorded")
    return 0


def cmd_decisions(argv: list[str]) -> int:
    days, grep = 90, None
    for a in argv:
        if a.startswith("--days="):
            days = max(1, int(a.split("=", 1)[1]))
        elif not a.startswith("-"):
            grep = a
    since = time.time() - days * 86400
    with db.connect() as con:
        rows = con.execute("SELECT d.*, t.title, t.note_path FROM decisions d "
                           "LEFT JOIN tasks t ON t.id=d.task_id WHERE d.ts>=? ORDER BY d.ts DESC",
                           (since,)).fetchall()
    if grep:
        g = grep.lower()
        rows = [r for r in rows if g in ((r["what"] or "") + " " + (r["why"] or "")
                                         + " " + (r["title"] or "")).lower()]
    if not rows:
        print("taskmark: no decisions recorded for that period")
        return 0
    for r in rows:
        print(f"{time.strftime('%F %H:%M', time.localtime(r['ts']))} [{r['who']}] {r['what']}")
        if r["why"]:
            print(f"    why: {r['why']}")
        if r["instead"]:
            print(f"    instead of: {r['instead']}")
        tail = f"    task #{r['task_id']} «{(r['title'] or '')[:40]}»" if r["task_id"] else ""
        if r["head"]:
            tail += f" · {r['repo']}@{r['head'][:8]}"
        if tail:
            print(tail)
    return 0


def cmd_git(argv: list[str]) -> int:
    """`taskmark git <sha>` — which task produced that commit; without a sha, recent ranges."""
    with db.connect() as con:
        if argv and not argv[0].startswith("-"):
            sha = argv[0]
            hits = []
            for r in con.execute("SELECT g.*, t.title, t.note_path, t.started_at FROM task_git g "
                                 "JOIN tasks t ON t.id=g.task_id ORDER BY g.first_ts DESC LIMIT 400"):
                if vcs.made_during(r["repo"], sha, r["head_start"], r["head_end"]):
                    hits.append(r)
            if not hits:
                print(f"taskmark: no task recorded around {sha}")
                return 0
            for r in hits[:5]:
                when = time.strftime("%F %H:%M", time.localtime(r["started_at"]))
                print(f"task #{r['task_id']} «{r['title']}» · {when} · {r['repo']}"
                      + (f"\n  note: {r['note_path']}" if r["note_path"] else ""))
            return 0
        rows = con.execute("SELECT g.*, t.title FROM task_git g JOIN tasks t ON t.id=g.task_id "
                           "ORDER BY g.first_ts DESC LIMIT 20").fetchall()
        if not rows:
            print("taskmark: no repository work recorded yet")
        for r in rows:
            print(f"#{r['task_id']:>4} {(r['title'] or '')[:40]:40} {r['repo']} "
                  f"{(r['head_start'] or '?')[:8]}..{(r['head_end'] or 'open')[:8]} "
                  f"{r['commits']} commits" + (" (dirty)" if r["dirty_end"] else ""))
    return 0


def cmd_note(argv: list[str]) -> int:
    sid = argv[0] if argv else _latest_session()
    state = st.load(sid)
    with db.connect() as con:
        note = close_task(con, state, "manual-note", write_note=True)
        task = db.get_task(con, state["task_id"]) if state.get("task_id") else None
        if task:
            # keep working in the same task after a manual note
            db.end_task(con, task["id"])
    print(note or "taskmark: nothing to write yet")
    return 0


def cmd_same(argv: list[str]) -> int:
    """Manual correction: 'this was the same task' — merge the open task into the previous one."""
    sid = argv[0] if argv else _latest_session()
    state = st.load(sid)
    task_id = state.get("task_id")
    with db.connect() as con:
        if not task_id:
            print("taskmark: no open task")
            return 0
        row = con.execute(
            "SELECT id FROM tasks WHERE session_id=? AND id<>? AND merged_into IS NULL "
            "ORDER BY id DESC LIMIT 1", (sid, task_id)).fetchone()
        if not row:
            print("taskmark: nothing to merge into")
            return 0
        db.merge_task(con, task_id, int(row["id"]))
        calibrate.feedback(con, "same-task", task_id, sid)
        state["task_id"] = int(row["id"])
    st.save(state)
    print(f"taskmark: merged into task #{row['id']} — thanks, this nudges the calibration")
    return 0


def cmd_calibrate(argv: list[str]) -> int:
    force = "--force" in argv
    sets = [a for a in argv if a.startswith("--set")]
    if sets:
        return _calibrate_set(argv)
    with db.connect() as con:
        new = calibrate.compute(con, force=force)
        cal = calibrate.current(con)
        counts = {k: con.execute("SELECT COUNT(*) n FROM signal_samples WHERE kind=?",
                                 (k,)).fetchone()["n"] for k in ("pause", "file_overlap", "cosine")}
        q = calibrate.quality(con)
    print(calibrate.describe(cal))
    print("samples:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    if q["labelled"]:
        print(f"against corrections: {q['labelled']} labelled · "
              f"right splits {q['true_split']} · wrong splits {q['false_split']} · "
              f"missed {q['missed_split']}"
              + (f" · F1 {q['f1']:.2f}" if q["f1"] is not None else ""))
    if new is None:
        need = max(0, config.CALIBRATION_MIN_SAMPLES - counts["pause"])
        print(f"not recalculated (need ~{need} more prompts, or run with --force)")
    return 0


def _calibrate_set(argv: list[str]) -> int:
    """`calibrate --set key=value ...` — apply thresholds a tuning pass came up with."""
    params: dict = {}
    for a in argv:
        if a.startswith("--set="):
            a = a.split("=", 1)[1]
        elif a == "--set":
            continue
        if "=" not in a or a.startswith("--"):
            continue
        k, v = a.split("=", 1)
        k = k.strip()
        if k not in config.DEFAULT_THRESHOLDS:
            print(f"unknown threshold: {k} (known: {', '.join(config.DEFAULT_THRESHOLDS)})")
            return 2
        try:
            params[k] = float(v)
        except ValueError:
            print(f"not a number: {k}={v}")
            return 2
    if not params:
        print("nothing to set")
        return 2
    with db.connect() as con:
        cal = calibrate.set_thresholds(con, params, note="model")
    print(calibrate.describe(cal))
    print("applied:", ", ".join(f"{k}={v}" for k, v in params.items()))
    return 0


def cmd_tune(argv: list[str]) -> int:
    """Print everything a tuning pass needs: the current thresholds, how they score against
    the corrections so far, and the decisions that were corrected — with the prompt that
    produced each one. Meant to be read by Claude, not by a human."""
    limit = 60
    for a in argv:
        if a.startswith("--limit="):
            limit = max(5, int(a.split("=", 1)[1]))
    with db.connect() as con:
        cal = calibrate.current(con)
        q = calibrate.quality(con)
        rows = con.execute(
            "SELECT ts, task_id, prompt, signals, reason, rule_boundary, label, label_src "
            "FROM prompt_signals WHERE label IS NOT NULL ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        recent = con.execute(
            "SELECT reason, COUNT(*) n FROM prompt_signals WHERE rule_boundary=1 "
            "GROUP BY reason ORDER BY n DESC LIMIT 15").fetchall()
    print("# taskmark tuning data")
    print()
    print("## current thresholds")
    for k in sorted(config.DEFAULT_THRESHOLDS):
        print(f"  {k} = {cal.get(k)}")
    print(f"  (calibration v{cal.get('version', 0)}, samples {cal.get('samples', 0)})")
    print()
    print("## how they score against corrections")
    print(f"  labelled decisions: {q['labelled']}")
    print(f"  correct splits    : {q['true_split']}")
    print(f"  wrong splits      : {q['false_split']}   <- rules cut, it was the same task")
    print(f"  missed splits     : {q['missed_split']}   <- rules continued, it was a new task")
    print(f"  correct continues : {q['true_continue']}")
    if q["f1"] is not None:
        print(f"  precision {q['precision']:.2f} · recall {q['recall']:.2f} · F1 {q['f1']:.2f}")
    print()
    print("## what fires most often")
    for r in recent:
        print(f"  {r['n']:>3}  {r['reason'] or '(none)'}")
    print()
    print("## corrected decisions (newest first)")
    print("  label = what it SHOULD have been; rules = what the detector answered")
    for r in rows:
        try:
            sig = json.loads(r["signals"] or "{}")
        except Exception:
            sig = {}
        shown = {k: (round(v, 3) if isinstance(v, (int, float)) else v)
                 for k, v in sig.items()}
        when = time.strftime("%d.%m %H:%M", time.localtime(r["ts"]))
        verdict = "SPLIT" if r["rule_boundary"] else "continue"
        print(f"\n  [{when}] rules={verdict} ({r['reason'] or '-'}) "
              f"label={r['label']} ({r['label_src']})")
        print(f"    prompt: {(r['prompt'] or '')[:160]}")
        print(f"    signals: {shown}")
    return 0


def cmd_report(argv: list[str]) -> int:
    days = 7
    for a in argv:
        if a.startswith("--days="):
            days = max(1, int(a.split("=", 1)[1]))
    with db.connect() as con:
        print(report.period_report(con, days=days))
        print()
        print(report.tasks_report(con, days=days))
    return 0


def cmd_cost(argv: list[str]) -> int:
    days = 30
    for a in argv:
        if a.startswith("--days="):
            days = max(1, int(a.split("=", 1)[1]))
    eur = pricing.eur_per_usd()
    with db.connect() as con:
        rows = con.execute(
            "SELECT date(ts,'unixepoch','localtime') d, model, SUM(input_tokens) i, "
            "SUM(output_tokens) o, SUM(cache_creation_tokens) cc, SUM(cache_read_tokens) cr "
            "FROM turns WHERE ts >= ? GROUP BY d, model ORDER BY d DESC",
            (time.time() - days * 86400,)).fetchall()
    per_day: dict[str, float] = {}
    unknown = set()
    for r in rows:
        c = pricing.cost_usd(r["model"], {"input": r["i"], "output": r["o"],
                                          "cache_creation": r["cc"], "cache_read": r["cr"]})
        if c is None:
            if r["i"] + r["o"] + r["cc"] + r["cr"] > 0:     # skip zero-token placeholders
                unknown.add(r["model"] or "unknown")
            continue
        per_day[r["d"]] = per_day.get(r["d"], 0.0) + c
    cur = "EUR" if eur else "USD"
    k = eur or 1.0
    total = sum(per_day.values()) * k
    print(f"Cost, last {days} days (rates: LiteLLM table"
          + (f", USD→EUR {eur}" if eur else ", no FX rate") + ")")
    for d in sorted(per_day, reverse=True)[:days]:
        print(f"  {d}  {per_day[d] * k:8.2f} {cur}")
    print(f"  {'total':10s} {total:8.2f} {cur}")
    if unknown:
        print("  ⚠ no price for:", ", ".join(sorted(unknown)))
    return 0


def mark_transcript(con, path: str, session_id: str, use_embeddings: bool) -> int:
    """Replay an old transcript through the same boundary rules and record the tasks.

    Without this, `ingest` gives turns but no tasks, and every per-task metric in the report
    stays empty until the plugin has run live for weeks. The split is done with the rules only
    unless --embed is passed: the embedding layer needs the model loaded in-process, which is
    fine for one import but pointless to force on someone who only wants the history.
    """
    events = transcript.replay(path)
    if not events:
        return 0
    cal = calibrate.current(con)
    state: dict = {"session_id": session_id}
    tasks = 0
    open_id: int | None = None
    for ts, kind, p in events:
        if kind == "t":
            state["tools_since_prompt"] = int(state.get("tools_since_prompt") or 0) + 1
            st.add_window_files(state, p["paths"])
            vocab = set(state.get("task_vocab") or []) | boundary.path_tokens(p["paths"])
            state["task_vocab"] = sorted(vocab)[:config.VOCAB_CAP]
            if open_id:
                db.touch_files(con, open_id, p["paths"])
            continue
        cwd = p.get("cwd") or os.getcwd()
        samples: dict = {}
        is_boundary, reason = boundary.decide(prompt=p["text"], cwd=cwd, now=ts, state=state,
                                              thresholds=cal, samples=samples)
        for k, v in samples.items():
            calibrate.add_sample(con, session_id, k, v, is_boundary)
        if is_boundary or open_id is None:
            if open_id:
                db.end_task(con, open_id, None, ended_at=ts)
            title = _title_of(p["text"])
            open_id = db.start_task(con, session_id, cwd, title, research.slugify(title),
                                    reason or "import", started_at=ts,
                                    calib_version=int(cal.get("version") or 0))
            tasks += 1
            state.update({"task_id": open_id, "cwd": cwd, "prompts": 1, "task_files": [],
                          "window_files": [], "manual_split": False, "task_started": ts,
                          "task_vocab": [], "prompt_gaps": [], "last_prompt_ts": None,
                          "prompt_vectors": []})
        else:
            state["prompts"] = int(state.get("prompts") or 0) + 1
            state["window_files"] = []
            con.execute("UPDATE tasks SET prompts=prompts+1 WHERE id=?", (open_id,))
        st.note_prompt(state, p["text"], ts, boundary.tokens(p["text"]))
        state["tools_since_prompt"] = 0
        if use_embeddings and embed.available():
            v = embed.encode_one(p["text"][:config.EMBED_MAX_PROMPT_CHARS])
            if v:
                vecs = list(state.get("prompt_vectors") or [])
                vecs.append(v)
                state["prompt_vectors"] = vecs[-config.CENTROID_PROMPTS:]
    if open_id:
        db.end_task(con, open_id, None, ended_at=events[-1][0])
    # attach the imported turns to the tasks they fall into, by time within this session
    con.execute(
        "UPDATE turns SET task_id = (SELECT t.id FROM tasks t WHERE t.session_id = turns.session_id "
        "  AND turns.ts >= t.started_at AND (t.ended_at IS NULL OR turns.ts <= t.ended_at) "
        "  ORDER BY t.started_at DESC LIMIT 1) "
        "WHERE session_id = ? AND task_id IS NULL", (session_id,))
    return tasks


def cmd_ingest(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    mark = "--mark" in argv
    use_embed = "--embed" in argv
    if use_embed:
        os.environ["TASKMARK_EMBED_INLINE"] = "1"      # one process, no daemon, for a one-off import
    root = args[0] if args else str(Path.home() / ".claude" / "projects")
    total = tasks = 0
    with db.connect() as con:
        for path in transcript.iter_sessions(root):
            sid = Path(path).stem
            total += ingest_transcript(con, path, sid, None)
            if mark:
                tasks += mark_transcript(con, path, sid, use_embed)
    print(f"taskmark: imported {total} new turns from {root}"
          + (f", marked {tasks} tasks" if mark else ""))
    return 0


def cmd_fetch_model(_argv: list[str]) -> int:
    """Download the embedding model deliberately — the plugin never does it on its own."""
    name = embed.model_name()
    if embed.model_present():
        hints.set_setting("embeddings", True)
        print(f"taskmark: {name} is already on disk — the embedding layer is on")
        return 0
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        print("taskmark: sentence-transformers is not installed — `pip install sentence-transformers`")
        return 1
    print(f"taskmark: downloading {name} (~450 MB, one time)…")
    SentenceTransformer(name, device="cpu")
    hints.set_setting("embeddings", True)
    print("taskmark: done — the embedding layer is now active")
    return 0


def cmd_off(_argv: list[str]) -> int:
    config.ensure_dirs()
    config.DISABLED_FLAG.write_text(time.strftime("%F %T") + "\n", encoding="utf-8")
    print("taskmark: off — hooks do nothing until `taskmark on`. Data and notes are kept.")
    return 0


def cmd_on(_argv: list[str]) -> int:
    config.DISABLED_FLAG.unlink(missing_ok=True)
    print("taskmark: on")
    return 0


def cmd_status(_argv: list[str]) -> int:
    with db.connect() as con:
        t = con.execute("SELECT COUNT(*) n, MIN(ts) a, MAX(ts) b FROM turns").fetchone()
        tasks = con.execute("SELECT COUNT(*) n FROM tasks").fetchone()["n"]
        notes = con.execute("SELECT COUNT(*) n FROM tasks WHERE note_path IS NOT NULL").fetchone()["n"]
        tok = con.execute(
            "SELECT SUM(input_tokens) i, SUM(output_tokens) o, SUM(cache_creation_tokens) cc, "
            "SUM(cache_read_tokens) cr FROM turns").fetchone()
        marks = [json.loads(r["payload"] or "{}") for r in con.execute(
            "SELECT payload FROM events WHERE kind='model-mark'")]
    span = ""
    if t["a"]:
        span = (f"{time.strftime('%F', time.localtime(t['a']))} … "
                f"{time.strftime('%F', time.localtime(t['b']))}")
    with db.connect() as con2:
        cal = calibrate.current(con2)
    print(f"state:  {'OFF' if config.is_disabled() else 'on'} · embeddings: "
          f"{'on (' + embed.model_name().split('/')[-1] + ')' if embed.available() else 'off'}")
    print(f"calib:  {calibrate.describe(cal)}")
    print(f"db:     {config.DB_PATH}")
    print(f"turns:  {t['n']} ({span})")
    print(f"tasks:  {tasks} (with notes: {notes})")
    print(f"tokens: in={tok['i'] or 0} out={tok['o'] or 0} "
          f"cache_write={tok['cc'] or 0} cache_read={tok['cr'] or 0}")
    if marks:
        # Claude's second opinion: silence is agreement, so only these moments are recorded
        named = sum(1 for m in marks if m.get("verdict") == "name")
        own = sum(1 for m in marks if m.get("verdict") == "new" and not m.get("agree"))
        undo = sum(1 for m in marks if m.get("verdict") == "same" and not m.get("agree"))
        back = sum(1 for m in marks if m.get("verdict") == "resume")
        print(f"claude: confirmed+named {named} · split the rules missed {own} · "
              f"undid a rule split {undo} · returned to an earlier task {back}")
    return 0


def _latest_session() -> str:
    files = sorted(config.STATE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0].stem if files else "unknown"


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "hook":
            return hook(rest[0] if rest else "")
        if cmd == "split":
            return cmd_split(rest)
        if cmd == "note":
            return cmd_note(rest)
        if cmd == "ingest":
            return cmd_ingest(rest)
        if cmd == "status":
            return cmd_status(rest)
        if cmd == "embed-daemon":
            return embed.serve()
        if cmd == "fetch-model":
            return cmd_fetch_model(rest)
        if cmd == "report":
            return cmd_report(rest)
        if cmd == "cost":
            return cmd_cost(rest)
        if cmd == "same":
            return cmd_same(rest)
        if cmd == "mark":
            return cmd_mark(rest)
        if cmd == "git":
            return cmd_git(rest)
        if cmd == "decide":
            return cmd_decide(rest)
        if cmd == "decisions":
            return cmd_decisions(rest)
        if cmd == "tune":
            return cmd_tune(rest)
        if cmd == "calibrate":
            return cmd_calibrate(rest)
        if cmd == "off":
            return cmd_off(rest)
        if cmd == "on":
            return cmd_on(rest)
        print(__doc__)
        return 1
    except Exception:                                   # noqa — never break the session
        _log("error in %s: %s" % (cmd, traceback.format_exc()))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
