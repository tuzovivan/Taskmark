"""Git facts about a task: where the work started, where it ended, what was committed.

Only local repositories on this machine — no network, no GitHub API. Work done over ssh on a
server without git leaves no trace here, and that is fine: the task still has its files.
Every call is guarded: a missing git, a broken repo or a submodule must never break a hook.
"""
from __future__ import annotations

import os
import subprocess

TIMEOUT = 3


def _run(repo: str, *args: str) -> str | None:
    try:
        r = subprocess.run(("git", "-C", repo, *args), capture_output=True, text=True,
                           timeout=TIMEOUT)
        return (r.stdout or "").strip() if r.returncode == 0 else None
    except Exception:
        return None


def repo_of(path: str) -> str | None:
    """Repository root containing `path` (a file or a directory), or None."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if not d or not os.path.isdir(d):
        return None
    root = _run(d, "rev-parse", "--show-toplevel")
    return root or None


def head(repo: str) -> str | None:
    return _run(repo, "rev-parse", "HEAD")


def branch(repo: str) -> str | None:
    b = _run(repo, "rev-parse", "--abbrev-ref", "HEAD")
    return b or None


def is_dirty(repo: str) -> bool:
    out = _run(repo, "status", "--porcelain")
    return bool(out)


def made_during(repo: str, sha: str, start: str | None, end: str | None) -> bool:
    """True when `sha` appeared in this repo between the task's start and its end."""
    if not end:
        return False
    try:
        inside = subprocess.run(("git", "-C", repo, "merge-base", "--is-ancestor", sha, end),
                                capture_output=True, timeout=TIMEOUT).returncode == 0
        if not inside:
            return False
        if not start:
            return True
        return subprocess.run(("git", "-C", repo, "merge-base", "--is-ancestor", sha, start),
                              capture_output=True, timeout=TIMEOUT).returncode != 0
    except Exception:
        return False


def commits_between(repo: str, start: str | None, end: str | None) -> list[str]:
    """Subjects of the commits a task added, oldest first ('<short sha> <subject>')."""
    if not start or not end or start == end:
        return []
    out = _run(repo, "log", "--reverse", "--format=%h %s", f"{start}..{end}")
    return [ln for ln in (out or "").splitlines() if ln][:50]
