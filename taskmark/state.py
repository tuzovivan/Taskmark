"""Per-session scratch state (which task is open, what files the current window touched).

One small JSON per session in ~/.claude/taskmark/state/. Hooks run as separate
processes, so this is how they hand facts to each other.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from pathlib import Path

from . import config


def _path(session_id: str) -> Path:
    safe = "".join(c for c in (session_id or "unknown") if c.isalnum() or c in "-_")[:80]
    return config.STATE_DIR / f"{safe}.json"


# fd of every session lock this process holds; the OS drops them when the process exits
_held: dict[str, int] = {}


def try_lock(session_id: str, wait: float = 0.0) -> bool:
    """Take the session's lock and keep it until this process exits.

    Hooks and `taskmark mark` are separate processes that read-modify-write the same
    state file. Without a lock a prompt hook that read the state before `mark resume`
    saved it wrote the old task back over the new one (23.09: #75 stayed open forever).
    Every command is a short process, so "until exit" is the whole read-modify-write."""
    config.ensure_dirs()
    key = str(_path(session_id).with_suffix(".lock"))
    if key in _held:
        return True
    fd = os.open(key, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _held[key] = fd
            return True
        except BlockingIOError:
            if time.time() >= deadline:
                os.close(fd)
                return False
            time.sleep(0.05)


def load(session_id: str) -> dict:
    config.ensure_dirs()
    # a stuck holder must not hang the session: after LOCK_WAIT go on unlocked, as before
    try_lock(session_id, wait=config.LOCK_WAIT_SECONDS)
    p = _path(session_id)
    if not p.exists():
        return {"session_id": session_id}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"session_id": session_id}


def save(state: dict) -> None:
    config.ensure_dirs()
    p = _path(state.get("session_id", "unknown"))
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tm-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)
    os.replace(tmp, p)


def note_prompt(state: dict, prompt: str, now: float, vocab_terms: set[str]) -> None:
    """Remember the rhythm (gaps between prompts) and the task's vocabulary."""
    last = float(state.get("last_prompt_ts") or 0)
    if last:
        gaps = list(state.get("prompt_gaps") or [])
        gaps.append(round(now - last, 1))
        state["prompt_gaps"] = gaps[-config.GAP_HISTORY:]
    state["last_prompt_ts"] = now
    vocab = set(state.get("task_vocab") or []) | set(vocab_terms)
    state["task_vocab"] = sorted(vocab)[: config.VOCAB_CAP]


def add_window_targets(state: dict, targets: list[str], cap: int = 50) -> None:
    """Targets of the current turn only; they join task_targets when the next prompt is
    judged — so the target signal compares the last turn with everything before it."""
    win = set(state.get("window_targets") or []) | set(targets)
    state["window_targets"] = sorted(win)[:cap]


def add_window_files(state: dict, paths: list[str], cap: int = 400) -> None:
    win = set(state.get("window_files") or [])
    allf = set(state.get("task_files") or [])
    for p in paths:
        win.add(p)
        allf.add(p)
    state["window_files"] = sorted(win)[:cap]
    state["task_files"] = sorted(allf)[: cap * 3]
