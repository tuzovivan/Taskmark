"""Research notes: what is worth keeping from a task once the context is thrown away.

The note is built from the transcript itself — files, commands, failures, decisions —
not from a retelling of the dialogue. Nothing is sent anywhere; no model is called.
If you want a richer, model-written note, run the `/research-save` command: it asks
Claude to fill the same file in while the context is still there.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from . import config, transcript


def slugify(text: str, max_len: int = 40) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[`*_>#\[\]()]+", " ", t)
    translit = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
        "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
        "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh",
        "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
        "ä": "a", "ö": "o", "ü": "u", "õ": "o", "š": "s", "ž": "z",
    }
    t = "".join(translit.get(ch, ch) for ch in t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return (t[:max_len].rstrip("-") or "task")


# A note is assembled from real shell history, and real shell history contains passwords.
# Everything written to a note goes through this first. Patterns only — no cleverness, and
# when in doubt it redacts: a masked note is still useful, a leaked one is not.
_SECRET_RE = [
    re.compile(r"(--(?:password|token|api-key|secret)[= ])(\S+)", re.I),
    re.compile(r"(?<![\w-])(-p)(?=[^\s-])(\S+)"),                        # mysql -pSecret
    re.compile(r"((?:password|passwd|pwd|secret|token|api[_-]?key|apikey|auth)\s*[=:]\s*)"
               r"([\"\']?)([^\s\"\']{4,})", re.I),
    re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{12,})"),
    re.compile(r"\b((?:sk|rpa|xai|hf|ghp|gho|github_pat|glpat|xoxb|xoxp|AIza)[-_])"
               r"([A-Za-z0-9_\-]{12,})"),
    re.compile(r"(ssh-rsa |ssh-ed25519 )(\S{20,})"),
    # keys passed in a URL query (?key=… &access_token=…) and in custom headers (X-Api-Key:)
    re.compile(r"([?&](?:key|apikey|api_key|access_token|token|secret|sig|signature|pass|password)=)"
               r"([^&\s\"\'#]{6,})", re.I),
    re.compile(r"(X-[\w-]*(?:Key|Token|Secret)\s*:\s*)([\"\']?)([^\s\"\']{8,})", re.I),
]


def redact(text: str) -> str:
    """Mask credentials that leak into shell commands and error output."""
    out = text or ""
    for rx in _SECRET_RE:
        out = rx.sub(lambda m: "".join(m.groups()[:-1]) + "***", out)
    return out


def _one_line(cmd: str, limit: int = 220) -> str:
    """A note is for reading: a 40-line heredoc pasted verbatim helps nobody."""
    flat = re.sub(r"\s+", " ", (cmd or "").strip())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + " …"


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text or "")
    return [p.strip(" -•\t") for p in parts if p and len(p.strip()) > 15]


def _pick(lines: list[str], markers: list[str], limit: int) -> list[str]:
    out, seen = [], set()
    for ln in lines:
        low = ln.lower()
        if any(m in low for m in markers):
            key = re.sub(r"\W+", "", low)[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append(re.sub(r"\s+", " ", ln)[:300])
            if len(out) >= limit:
                break
    return out


def _note_worthy(path: str) -> bool:
    """Paths scraped from shell text include `cd` targets, globs cut at a `*`, slash
    commands quoted in heredocs. Keep files; keep missing paths only when they look
    like one (a deleted or moved file still belongs in the note)."""
    if path.endswith(("/", "-", ".")) or os.path.isdir(path):
        return False
    if os.path.exists(path):
        return True
    return path.count("/") >= 2 and "." in os.path.basename(path)


def _source_pointer(act: "transcript.Activity") -> list[str]:
    """Where the raw dialogue of this task sits.

    The note keeps facts; the reasoning, the wording and everything the extractor did not
    recognise stay in the transcript. Without an exact pointer that transcript is a haystack —
    hundreds of megabytes of JSONL across sessions — so the note carries the file, the line
    range and a command that prints exactly that slice.
    """
    if not getattr(act, "path", ""):
        return []
    ranges = getattr(act, "ranges", None) or []
    if len(ranges) > 1:            # the task was left and resumed: only its own stretches
        human = ", ".join(f"{a}–{b}" for a, b in ranges)
        script = ";".join(f"{a},{b}p" for a, b in ranges)
        return [f"- Transcript: `{act.path}` lines {human} (the gaps belong to other tasks) — "
                f"`sed -n '{script}' {act.path}`"]
    if len(ranges) == 1:
        a, b = ranges[0]
        return [f"- Transcript: `{act.path}` lines {a}–{b} — `sed -n '{a},{b}p' {act.path}`"]
    if act.start_line and act.end_line:
        return [f"- Transcript: `{act.path}` lines {act.start_line}–{act.end_line} — "
                f"`sed -n '{act.start_line},{act.end_line}p' {act.path}`"]
    return [f"- Transcript: `{act.path}`"]


# A closing answer is prose, and prose is where the conclusion lives; anything shorter than
# this is an acknowledgement ("done", "running it now"), not a finding.
_CLOSING_MIN = 200
_CLOSING_MAX = 1600


def _closing_answer(act: "transcript.Activity") -> str:
    """The last substantial thing that was said before the task ended.

    The extractor below hunts for files, commands and failures — facts. What it cannot see is
    the assessment they add up to, and that is usually the final answer of the task. Copying it
    verbatim costs nothing and keeps the one part of a task that cannot be reconstructed from
    the shell history.
    """
    texts = [(a.get("text") or "").strip() for a in getattr(act, "assistant_text", [])]
    for text in reversed(texts[-4:]):          # walk back past short sign-offs
        if len(text) < _CLOSING_MIN:
            continue
        if len(text) <= _CLOSING_MAX:
            return text
        cut = text[:_CLOSING_MAX]
        brk = max(cut.rfind("\n\n"), cut.rfind(". "), cut.rfind("\n"))
        return (cut[:brk + 1] if brk > _CLOSING_MAX // 2 else cut).rstrip() + "\n\n…(cut — the rest is in the transcript)"
    return ""


def build_note(*, title: str, task_id: int, session_id: str, cwd: str, started_at: float,
               ended_at: float, act: transcript.Activity, reason: str,
               git: list | None = None, recorded: list | None = None) -> str:
    started = datetime.fromtimestamp(started_at).strftime("%Y-%m-%d %H:%M")
    ended = datetime.fromtimestamp(ended_at or started_at).strftime("%H:%M")

    parents = {os.path.dirname(p) for p in act.files}   # a gone path that had children was a dir
    files = sorted(((p, n) for p, n in act.files.items() if p not in parents and _note_worthy(p)),
                   key=lambda kv: (-kv[1], kv[0]))[:config.MAX_NOTE_ITEMS]
    cmds, seen_cmd = [], set()
    for c in act.commands:
        key = re.sub(r"\s+", " ", c["cmd"])[:120]
        if key in seen_cmd:
            continue
        seen_cmd.add(key)
        cmds.append(c)
        if len(cmds) >= config.MAX_NOTE_ITEMS:
            break

    assistant_lines: list[str] = []
    for a in act.assistant_text:
        assistant_lines += _sentences(a["text"])
    user_lines: list[str] = []
    for p in act.prompts:
        user_lines += _sentences(p["text"])

    decisions = _pick(assistant_lines + user_lines, config.DECISION_MARKERS, 15)
    tried = _pick(assistant_lines + user_lines, config.FAILURE_MARKERS, 15)
    questions = [re.sub(r"\s+", " ", p["text"])[:200] for p in act.prompts if "?" in p["text"]][-8:]

    out = [
        f"# {title.strip()[:120] or 'Task'}",
        "",
        f"- Date: {started} – {ended}",
        f"- Directory: `{cwd}`",
        f"- Session: `{session_id}` · task #{task_id} · boundary: {reason or 'n/a'}",
        f"- Volume: {len(act.prompts)} prompts, {len(act.turns)} turns, "
        f"{len(act.files)} files, {len(act.commands)} shell commands",
    ]
    out += _source_pointer(act)
    out += [
        "",
        "## What this task was about",
        "",
    ]
    for p in act.prompts[:6]:
        first = re.sub(r"\s+", " ", p["text"])[:220]
        out.append(f"- {first}")
    if len(act.prompts) > 6:
        out.append(f"- …and {len(act.prompts) - 6} more prompts")

    closing = _closing_answer(act)
    if closing:
        out += ["", "## Conclusions (the closing answer, verbatim)", "", closing]

    for g in (git or []):
        # where each repository stood when the task first touched it, and where it ended
        out += ["", f"## Git — {g['repo']}" + (f" ({g['branch']})" if g.get("branch") else ""), ""]
        rng = f"{(g.get('start') or '?')[:8]}..{(g.get('end') or '?')[:8]}"
        out.append(f"- Range: `{rng}`" + ("  ⚠️ uncommitted changes left" if g.get("dirty") else ""))
        if g.get("commits"):
            out.append(f"- Commits ({len(g['commits'])}):")
            out += [f"  - `{c}`" for c in g["commits"]]
        else:
            out.append("- No commits in this task")
    out += ["", "## Files touched", ""]
    if files:
        for path, hits in files:
            out.append(f"- `{path}` ({hits})")
    else:
        out.append("- (none recorded)")

    out += ["", "## Commands worth remembering", ""]
    if cmds:
        for c in cmds:
            desc = f" — {c['description']}" if c.get("description") else ""
            out.append(f"- `{_one_line(c['cmd'])}`{desc}")
    else:
        out.append("- (none recorded)")

    out += ["", "## What did not work", ""]
    if act.failures or tried:
        for f in act.failures[:config.MAX_NOTE_ITEMS]:
            head = f["description"] or f["cmd"][:120]
            out.append(f"- `{head}` → {f['error'][:200]}")
        for t in tried:
            out.append(f"- {t}")
    else:
        out.append("- (nothing recorded — fill in by hand if it matters)")

    out += ["", "## Decisions", ""]
    if recorded:                       # written by whoever decided, with the reason
        for d in recorded:
            out.append(f"- **{d['what']}** — {d['who']}")
            if d.get("why"):
                out.append(f"  - why: {d['why']}")
            if d.get("instead"):
                out.append(f"  - instead of: {d['instead']}")
    elif decisions:                    # fallback: sentences that sounded like a decision
        out += [f"- {d}" for d in decisions]
    else:
        out.append("- (nothing recorded — `taskmark decide` while it is fresh)")

    if questions:
        out += ["", "## Open questions", ""] + [f"- {q}" for q in questions]

    out += ["", "---", "",
            "Written automatically by taskmark when the task boundary was detected.",
            "It is a skeleton built from facts in the transcript, not a summary of the dialogue —",
            "edit it, or regenerate a fuller one with `/research-save` while the context is alive.",
            "When the skeleton is not enough, the full dialogue is still in the transcript listed",
            "at the top: read those lines instead of guessing.",
            ""]
    return "\n".join(out)


# The notes live next to the project, because that is where they are useful — but a note is
# a private working file, not part of the repository. A `.gitignore` containing `*` inside the
# directory hides the notes *and itself* from git, whatever the repo's own rules say, and does
# it for every repo without touching the project's .gitignore. Opt out with TASKMARK_NOTES_GIT=1.
GITIGNORE_BODY = ("# Private working notes written by the taskmark plugin.\n"
                  "# They are local research dumps, not project files — nothing here is committed.\n"
                  "# Delete this file (or set TASKMARK_NOTES_GIT=1) if you do want them in git.\n"
                  "*\n")


def _shield_from_git(base: Path) -> None:
    if os.environ.get("TASKMARK_NOTES_GIT"):
        return
    gi = base / ".gitignore"
    try:
        if not gi.exists():
            gi.write_text(GITIGNORE_BODY, encoding="utf-8")
    except Exception:
        pass                                   # a note is never worth failing over


def _reuse_or_new(existing: str | None, cwd: str, slug: str, when: float) -> Path:
    """Overwrite this task's own note when it still exists; otherwise start a fresh file."""
    if existing:
        p = Path(existing)
        try:
            if p.is_file() and p.parent.is_dir():
                return p
        except OSError:
            pass                               # unreadable path → fall through to a new file
    return note_path(cwd, slug, when)


def note_path(cwd: str, slug: str, when: float) -> Path:
    base = Path(cwd or os.getcwd()) / config.RESEARCH_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    _shield_from_git(base)
    day = datetime.fromtimestamp(when).strftime("%Y-%m-%d")
    p = base / f"{day}-{slug}.md"
    n = 2
    while p.exists():
        p = base / f"{day}-{slug}-{n}.md"
        n += 1
    return p


def write_note(*, cwd: str, title: str, task_id: int, session_id: str, started_at: float,
               ended_at: float, act: transcript.Activity, reason: str,
               git: list | None = None, recorded: list | None = None,
               existing: str | None = None) -> Path:
    """`existing` is the note this task wrote earlier, if any.

    One task can be closed more than once — a note is written when the session goes idle, and
    the task then carries on, or a compaction writes one mid-task. Each of those used to leave
    another `-2` file describing the same work, and a reader could not tell which was current.
    The later, fuller note now replaces the earlier one in place.
    """
    slug = slugify(title)
    path = _reuse_or_new(existing, cwd, slug, started_at)
    body = build_note(title=title, task_id=task_id, session_id=session_id, cwd=cwd,
                      started_at=started_at, ended_at=ended_at, act=act, reason=reason, git=git,
                      recorded=recorded)
    path.write_text(redact(body), encoding="utf-8")
    return path
