"""Task boundary detection — rules only (stage 1), and deliberately language-agnostic.

Phrase lists were the first attempt and were dropped: every person words a switch
differently, and in three languages at once. What is left works the same whatever the
user types, because it looks at *structure*, not vocabulary:

  hard signals (any one of them ends the task)
    · manual mark (`/task-split`)
    · git branch changed
    · a very long pause (PAUSE_HARD_SECONDS, an hour by default)

  strong signals (two of them, or one plus a weak one, end the task)
    · the working directory moved to a different tree (`cd` inside the project is normal work)
    · the files of the last prompt window do not overlap the task's files
    · lexical continuity: the words of the new prompt do not appear in the task's
      vocabulary (previous prompts + names of the files touched). Plain term overlap,
      no word lists, no model — works for Russian, English, Estonian or code alike.

  weak signals
    · an unusual pause for this session's own rhythm (people go for coffee mid-task)
    · the previous turn ended without touching any file — the task looked finished

An optional user file (~/.claude/taskmark/markers.txt, one phrase per line) can add
explicit switch phrases. It is empty by default.
"""
from __future__ import annotations

import re
import statistics
from typing import Iterable

from . import config, embed

_TOKEN_RE = re.compile(r"[^\W\d_]{3,}|\d{4,}", re.UNICODE)


def tokens(text: str) -> set[str]:
    """Words of 3+ letters in any script, plus numbers of 4+ digits (SKUs, ids, ports)."""
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def path_tokens(paths: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for p in paths or []:
        out |= tokens(re.sub(r"[/._-]+", " ", p.rsplit("/", 2)[-1]))
    return out


def _shared_dir(d: str) -> bool:
    """Directories every task passes through say nothing about the topic. Measured on a real
    session: /root, /srv, /home and ~/.claude/.../memory made a taskmark task and a translator
    task look 0.62 alike, so the file signal never fired."""
    if d.count("/") < 2:                       # /root, /srv, /home, /opt, /etc ...
        return True
    return "/.claude" in d                     # memory, research notes, plugin state


def dirs_of(paths: Iterable[str]) -> set[str]:
    """Directory of each path, plus its parent — a topic lives in a couple of directories."""
    out: set[str] = set()
    for p in paths or []:
        d = p.rsplit("/", 1)[0] if "/" in p else p
        if len(d) > 1 and not _shared_dir(d):
            out.add(d)
            parent = d.rsplit("/", 1)[0]
            if len(parent) > 1 and not _shared_dir(parent):
                out.add(parent)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def coverage(new: set[str], known: set[str]) -> float:
    """Share of the new prompt's terms that the task has already seen."""
    if not new:
        return 1.0
    return len(new & known) / len(new)


def user_markers() -> list[str]:
    try:
        raw = config.MARKERS_FILE.read_text(encoding="utf-8")
    except Exception:
        return []
    return [ln.strip().lower() for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]


def adaptive_pause_limit(gaps: list[float], fallback: float | None = None,
                         floor: float | None = None) -> float:
    """A pause is 'long' relative to how this session normally breathes.

    `floor` is the calibrated `pause_floor` — the gap this particular person takes without
    it meaning anything. Someone who answers chat between prompts and someone who types
    without pause need different floors, so it is data, not a constant.
    """
    gaps = [g for g in gaps if g > 0][-config.GAP_HISTORY:]
    lo = float(floor if floor is not None else config.PAUSE_FLOOR_SECONDS)
    if len(gaps) < 3:
        return float(fallback if fallback is not None
                     else config.DEFAULT_THRESHOLDS["pause_hard"])
    med = statistics.median(gaps)
    return max(lo, config.PAUSE_RHYTHM_FACTOR * med)


def decide(*, prompt: str, cwd: str | None, now: float, state: dict,
           git_branch: str | None = None, thresholds: dict | None = None,
           samples: dict | None = None) -> tuple[bool, str]:
    """Should this prompt start a new task? Returns (boundary, reason).

    `thresholds` comes from calibrate.current() — the user's own percentiles.
    `samples` is filled in with the raw signal values so they can be stored for calibration.
    """
    th = {**config.DEFAULT_THRESHOLDS, **(thresholds or {})}
    out = samples if samples is not None else {}
    if state.get("manual_split"):
        return True, "manual"
    if not state.get("task_id"):
        return True, "start"

    if git_branch and state.get("git_branch") and git_branch != state["git_branch"]:
        return True, "branch-change"

    gap = now - float(state.get("last_prompt_ts") or state.get("last_activity_ts") or now)
    if gap > 0:
        out["pause"] = gap
    if gap >= float(th["pause_hard"]):
        return True, f"pause:{int(gap // 60)}min"

    young = int(state.get("prompts") or 0) < config.MIN_TURNS_BEFORE_SPLIT
    if young:
        return False, ""

    strong: list[str] = []
    weak: list[str] = []

    # where the work happens: the last turn went somewhere the task never went (another
    # server, container, project). It used to end the task on its own; measured over a day
    # it was the single largest source of false boundaries (8 splits, at least 4 wrong).
    # One task routinely spans localhost and a couple of servers — `ssh prod` inside work
    # that started locally is ordinary, not a new topic. So it now votes like any other
    # strong signal and needs company. It judges the previous turn — see turn_based().
    win_t = set(state.get("window_targets") or [])
    task_t = set(state.get("task_targets") or [])
    if win_t and task_t:
        out["target_overlap"] = jaccard(win_t, task_t)
        if not (win_t & task_t):
            strong.append("target")

    # `cd` inside the same tree is normal work, not a new task — only a different tree counts
    old_cwd = state.get("cwd") or ""
    if cwd and old_cwd and cwd != old_cwd:
        same_tree = cwd.startswith(old_cwd.rstrip("/") + "/") or old_cwd.startswith(cwd.rstrip("/") + "/")
        if not same_tree:
            strong.append("cwd")

    # an unusual pause alone is not a new task — people go for coffee mid-task.
    # The limit is whichever is stricter: this user's calibrated pause, or this session's rhythm.
    limit = max(float(th.get("pause_floor") or config.PAUSE_FLOOR_SECONDS),
                min(float(th["pause_soft"]),
                    adaptive_pause_limit([float(g) for g in (state.get("prompt_gaps") or [])],
                                         fallback=float(th["pause_hard"]),
                                         floor=th.get("pause_floor"))))
    if gap > limit:
        weak.append(f"pause:{int(gap // 60)}min>{int(limit // 60)}min")

    # Compare *directories*, not exact paths: one prompt rarely reopens the very same file,
    # but work on a topic stays inside the same few directories. Measured on real logs:
    # exact-path overlap was 0.00 at the 85th percentile — it fired almost every prompt.
    win_files = set(state.get("window_files") or [])
    earlier_files = set(state.get("task_files") or []) - win_files   # subtract FILES...
    window = dirs_of(win_files)                                      # ...then compare DIRS
    task_files = dirs_of(earlier_files)
    if (len(win_files) >= config.MIN_FILES_FOR_SAMPLE
            and len(earlier_files) >= config.MIN_FILES_FOR_SAMPLE
            and window and task_files):
        ov = jaccard(window, task_files)
        out["file_overlap"] = ov                      # always measured: feeds the calibrator
        enough = (len(win_files) >= config.MIN_FILES_FOR_OVERLAP
                  and len(earlier_files) >= config.MIN_FILES_FOR_OVERLAP)
        if enough and ov <= float(th["file_overlap"]):
            strong.append(f"files:{ov:.2f}")          # ...but only acted on when solid

    vocab = set(state.get("task_vocab") or [])
    new_terms = tokens(prompt)
    min_terms = int(th.get("min_prompt_terms") or config.MIN_PROMPT_TERMS)
    long_enough = len(new_terms) >= min_terms
    if long_enough and len(vocab) >= config.MIN_VOCAB_TERMS:
        cov = coverage(new_terms, vocab)
        if cov <= config.CONTINUITY_MIN:
            strong.append(f"words:{cov:.2f}")

    # --- second layer: meaning, not words ---------------------------------
    # Same floor as the lexical signal: a centroid has nothing to say about "да сделай".
    cos = prompt_similarity(prompt, state, min_terms=min_terms) if long_enough else None
    if cos is not None:
        out["cosine"] = cos
        if cos <= float(th["cosine_min"]):
            strong.append(f"topic:{cos:.2f}")

    if not state.get("tools_since_prompt"):
        weak.append("idle-turn")

    markers = user_markers()
    if markers:
        low = " ".join((prompt or "").lower().split())
        for m in markers:
            if m in low:
                strong.append("marker")
                break

    if score(strong, weak) >= float(th.get("boundary_score") or config.BOUNDARY_SCORE):
        return True, "+".join(strong + weak)
    return False, ""


def score(strong: Iterable[str], weak: Iterable[str]) -> float:
    """Total evidence for a boundary. `words` and `topic` share one vote between them:
    both read the same sentence, so a reworded follow-up trips both and used to look like
    two independent findings."""
    total = 0.0
    text_counted = False
    for sig in strong:
        name = sig.split(":")[0]
        if name in config.TEXT_SIGNALS:
            if text_counted:
                continue
            text_counted = True
        total += config.SIGNAL_WEIGHT_STRONG
    total += config.SIGNAL_WEIGHT_WEAK * len(list(weak))
    return total


# Signals that describe the LAST TURN (what was touched after the previous prompt), not the
# prompt just typed. When only these fired, the switch happened at the previous prompt.
_TURN_SIGNALS = ("files", "target")
_PROMPT_SIGNALS = ("words", "topic", "marker", "cwd", "manual", "start", "branch-change")


def turn_based(reason: str) -> bool:
    """True when the boundary belongs to the previous prompt (see _TURN_SIGNALS)."""
    parts = [p.split(":")[0] for p in (reason or "").split("+") if p]
    return (any(p in _TURN_SIGNALS for p in parts)
            and not any(p in _PROMPT_SIGNALS for p in parts))


def prompt_similarity(prompt: str, state: dict, min_terms: int | None = None) -> float | None:
    """Cosine between this prompt and the centroid of the prompts already in the task.

    Returns None when embeddings are unavailable or the task has too few prompts to have
    a centroid yet — in that case the detector simply runs on the rule layer.
    """
    vecs = state.get("prompt_vectors") or []
    if len(vecs) < config.EMBED_MIN_PROMPTS or not embed.available():
        return None
    if len(tokens(prompt)) < int(min_terms or config.EMBED_MIN_PROMPT_TERMS):
        return None
    v = embed.encode_one(prompt[:config.EMBED_MAX_PROMPT_CHARS])
    if not v:
        return None
    return embed.cosine(v, embed.centroid(vecs))


def check_return(*, con, session_id: str, task_id: int, prompts: int, files: Iterable[str],
                 now: float, db) -> int | None:
    """Returning to the previous task? Then the split was wrong — merge it back.

    Only young tasks are merged, so a genuine new task that happens to touch one shared
    file is not swallowed by its predecessor.
    """
    files = [f for f in files if f]
    if not files or prompts > config.MIN_TURNS_BEFORE_SPLIT + 1:
        return None
    owner = db.find_task_by_files(con, session_id, files, not_task=task_id,
                                  since_ts=now - config.RETURN_WINDOW_SECONDS)
    if not owner:
        return None
    own_files = db.task_file_set(con, owner)
    if jaccard(set(files), own_files) < config.RETURN_FILE_OVERLAP:
        return None
    return owner
