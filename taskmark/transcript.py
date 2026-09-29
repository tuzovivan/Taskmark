"""Reading Claude Code's own session log (~/.claude/projects/<slug>/<session>.jsonl).

Verified against real transcripts (v2.1.x). What we rely on:

  assistant turn : type=assistant, requestId, timestamp (ISO-Z), cwd, version, isSidechain,
                   message.model, message.usage.{input_tokens, output_tokens,
                   cache_creation_input_tokens, cache_read_input_tokens}
  human prompt   : type=user, origin.kind='human', message.content (str | list of blocks)
  tool call      : assistant message content block type='tool_use' → name + input
                   (file_path / path / notebook_path / pattern / command)
  tool result    : type=user with toolUseResult {stdout, stderr, interrupted, ...}

Everything else in the file (attachment, mode, ai-title, file-history-*) is ignored.
The reader is incremental: it remembers a byte offset per transcript.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

USAGE_MAP = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_creation_input_tokens": "cache_creation_tokens",
    "cache_read_input_tokens": "cache_read_tokens",
}

PATH_KEYS = ("file_path", "path", "notebook_path", "output_path")
# paths that appear inside a shell command line
_SHELL_PATH_RE = re.compile(r"(?:^|[\s=:'\"])(/[\w./@+-]{3,}|\./[\w./@+-]{2,}|~/[\w./@+-]{2,})")
# Paths that say nothing about the task. Claude Code's own scratch dir is skipped unless
# TASKMARK_KEEP_SCRATCH=1 (handy when testing the plugin itself).
# POSIX system paths that say nothing about the task (harmless no-ops on Windows)
_NOISE_PREFIXES = ("/dev/", "/proc/", "/sys/", "/usr/bin/", "/bin/", "/etc/ssl/")
if not os.environ.get("TASKMARK_KEEP_SCRATCH"):
    _NOISE_PREFIXES += ("/tmp/claude-",)


def iso_to_ts(s: str | None) -> float:
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def _text_of(content: Any) -> str:
    """message.content is either a plain string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                out.append(b["text"])
        return "\n".join(out)
    return ""


def clean_path(p: str, cwd: str | None = None) -> str | None:
    p = p.strip().strip("'\"")
    if "\\" in p and ":" in p[:3]:            # Windows path → same separators as everything else
        p = p.replace("\\", "/")
    if not p or p.startswith(_NOISE_PREFIXES) or p.startswith("//"):   # "//host/..." = tail of a URL
        return None
    if p.startswith("~/"):
        p = os.path.expanduser(p)
    if not (p.startswith("/") or (len(p) > 2 and p[1] == ":")) and cwd:
        p = os.path.normpath(os.path.join(cwd, p))
    if len(p) < 4 or p.count("/") == 0:
        return None
    return p


def paths_from_command(cmd: str, cwd: str | None) -> list[str]:
    out: list[str] = []
    for m in _SHELL_PATH_RE.finditer(cmd or ""):
        c = clean_path(m.group(1), cwd)
        if c:
            out.append(c)
    return out


def paths_from_tool(name: str, tool_input: dict, cwd: str | None) -> list[str]:
    out: list[str] = []
    for k in PATH_KEYS:
        v = tool_input.get(k)
        if isinstance(v, str):
            c = clean_path(v, cwd)
            if c:
                out.append(c)
    if name == "Bash" and isinstance(tool_input.get("command"), str):
        out += paths_from_command(tool_input["command"], cwd)
    if name in ("Grep", "Glob"):
        for k in ("path", "glob", "pattern"):
            v = tool_input.get(k)
            if isinstance(v, str) and ("/" in v):
                c = clean_path(v, cwd)
                if c:
                    out.append(c)
    return out



# --- work targets: WHERE the work happens, not which files ------------------------------
# A switch from one server / container / cluster / project to another is the clearest sign
# of a new task, and it needs no language at all. Kept generic on purpose: any ssh alias,
# any container, any repo — nothing specific to one user's machines.
# only in command position ("echo ssh is down" is not a connection)
_REMOTE_RE = re.compile(r"(?:^|[;&|(`\n]|\bsudo|\bexec|\btimeout\s+\S+)\s*(ssh|scp|rsync|sftp|mosh)\s+([^;&|`\n]*)")
_SSH_ARG_OPTS = set("bcDEeFIiJLlmOoPpQRSWw")        # options of ssh/scp/sftp that take a value
_DOCKER_RE = re.compile(r"\bdocker\s+(exec|attach|logs|restart|cp)\s+([^;&|`\n]*)")
_DOCKER_ARG_OPTS = {"-e", "--env", "-u", "--user", "-w", "--workdir", "--env-file", "--since",
                    "--until", "-n", "--tail", "-t", "--time", "--detach-keys"}
_KUBE_RE = re.compile(r"\bkubectl\b.*?--context[= ](\S+)")
_DBHOST_RE = re.compile(r"\b(?:mysql|mariadb|psql|mongosh?|redis-cli|clickhouse-client)\b[^;&|\n]*?\s-h\s*([\w.-]+)")
_LOCALHOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_ROOT_MARKERS = (".git", "pyproject.toml", "package.json", "composer.json", "go.mod",
                 "Cargo.toml", ".claude-plugin", "pom.xml", "Gemfile")


def _remote_host(tool: str, args: str) -> str | None:
    toks = args.replace("'", " ").replace('"', " ").split()
    skip = False
    for t in toks:
        if skip:
            skip = False
            continue
        if t.startswith("-"):
            if tool != "rsync" and len(t) == 2 and t[1] in _SSH_ARG_OPTS:
                skip = True
            continue
        if tool in ("ssh", "mosh", "sftp"):
            host = t.split("@")[-1].split(":")[0]
        elif ":" in t and not t.startswith("/"):
            host = t.split(":")[0].split("@")[-1]
        else:
            continue
        if host and re.fullmatch(r"[\w.-]+", host) and host not in _LOCALHOSTS:
            return host
        return None
    return None


def _docker_container(sub: str, args: str) -> str | None:
    toks, skip = args.split(), False
    for t in toks:
        if skip:
            skip = False
            continue
        if t.startswith("-"):
            # "-t" is --time for restart but --tty for exec
            skip = t in _DOCKER_ARG_OPTS and not (t == "-t" and sub == "exec")
            continue
        name = t.split(":")[0]                      # docker cp <container>:<path>
        return name if re.fullmatch(r"[\w.-]+", name) else None
    return None


_root_cache: dict[str, str | None] = {}


def project_root(path: str) -> str | None:
    """Nearest ancestor that looks like a project; else the path's two top directories."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if not d or not d.startswith("/") or "/.claude" in d:
        return None
    if d in _root_cache:
        return _root_cache[d]
    cur, root = d, None
    for _ in range(8):
        if any(os.path.exists(os.path.join(cur, m)) for m in _ROOT_MARKERS):
            root = cur
            break
        up = os.path.dirname(cur)
        if up == cur:
            break
        cur = up
    if root is None or root.count("/") < 2:          # "/" or "/root" is not a project
        parts = [x for x in d.split("/") if x]
        root = "/" + "/".join(parts[:2]) if len(parts) >= 2 else None
    _root_cache[d] = root
    return root


_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\b", re.S)


def targets_from_tool(name: str, tool_input: dict, cwd: str | None) -> list[str]:
    """'ssh:<host>', 'docker:<container>', 'k8s:<context>', 'db:<host>', 'local:<project>'."""
    out: list[str] = []
    cmd = tool_input.get("command") if name == "Bash" else None
    if isinstance(cmd, str):
        # a heredoc body is data (a script, docs, a commit message), not commands run here
        cmd = _HEREDOC_RE.sub("<<heredoc", cmd)
        for m in _REMOTE_RE.finditer(cmd):
            h = _remote_host(m.group(1), m.group(2))
            if h:
                out.append("ssh:" + h)
        for m in _DOCKER_RE.finditer(cmd):
            c = _docker_container(m.group(1), m.group(2))
            if c:
                out.append("docker:" + c)
        out += ["k8s:" + m.group(1) for m in _KUBE_RE.finditer(cmd)]
        out += ["db:" + h for h in _DBHOST_RE.findall(cmd) if h not in _LOCALHOSTS]
    # local project: only for paths that exist here. Kept next to a remote host on purpose:
    # `scp ./app/x host:` touches both, so a deploy step still overlaps the project's task.
    for p in paths_from_tool(name, tool_input, cwd):
        if os.path.exists(p):
            r = project_root(p)
            if r:
                out.append("local:" + r)
    return sorted(set(out))

@dataclass
class Turn:
    request_id: str        # dedupe key is (message_id, request_id, session_id) — see ccusage
    ts: float
    session_id: str
    cwd: str | None
    model: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    context_size: int = 0
    is_sidechain: bool = False
    effort: str | None = None
    version: str | None = None
    message_id: str | None = None
    cost_usd: float | None = None      # some records carry costUSD — keep it if present

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_creation_tokens + self.cache_read_tokens)

    def dedupe_key(self) -> tuple:
        return (self.message_id or self.request_id, self.request_id, self.session_id)


@dataclass
class Activity:
    """Everything one pass over a transcript slice gives us."""
    turns: list[Turn] = field(default_factory=list)
    prompts: list[dict] = field(default_factory=list)      # {ts, text}
    files: dict[str, int] = field(default_factory=dict)    # path -> hits
    commands: list[dict] = field(default_factory=list)     # {ts, cmd, description}
    failures: list[dict] = field(default_factory=list)     # {ts, cmd, error}
    assistant_text: list[dict] = field(default_factory=list)  # {ts, text}
    last_ts: float = 0.0
    offset: int = 0
    # where this slice sits in the transcript, so a note can point back at the raw dialogue.
    # Line numbers are absolute only when the pass started at the top of the file (offset 0);
    # a resumed incremental read leaves them at 0 rather than reporting a number that lies.
    path: str = ""
    start_line: int = 0
    end_line: int = 0
    # with `spans`, a task that was left and resumed owns several stretches of the file
    ranges: list[list[int]] = field(default_factory=list)


def read(path: str, offset: int = 0, since_ts: float = 0.0, until_ts: float | None = None,
         spans: list[tuple[float, float]] | None = None,
         own_requests: set[str] | None = None) -> Activity:
    """Parse a transcript from a byte offset (and/or a time window). Never raises on junk lines.

    `spans` narrows the window to the stretches a task actually owned: when work switched to
    another task and came back, the records in between belong to that other task and must not
    end up in this one's note. `own_requests` settles assistant records exactly by requestId,
    so the tail of the other task's last answer does not slip in at a span edge."""
    act = Activity(offset=offset)
    if not path or not os.path.isfile(path):
        return act
    act.path = path
    track_lines = (offset == 0)            # counting only means something from the first line
    gap = False                            # a foreign record was skipped since the last kept one
    lineno = 0
    pending_tool: dict[str, dict] = {}     # tool_use_id -> {cmd, ts}
    with open(path, "rb") as fh:
        fh.seek(offset)
        for raw in fh:
            lineno += 1
            act.offset = fh.tell()
            try:
                d = json.loads(raw.decode("utf-8", "ignore"))
            except Exception:
                continue
            ts = iso_to_ts(d.get("timestamp"))
            if ts and since_ts and ts < since_ts:
                continue
            if ts and until_ts and ts > until_ts:
                continue
            if spans and ts:
                rid_ = d.get("requestId")
                if d.get("type") == "assistant" and own_requests is not None and rid_:
                    mine = rid_ in own_requests
                else:
                    mine = any(lo <= ts <= hi for lo, hi in spans)
                if not mine:
                    gap = True
                    continue
            if track_lines and ts and spans:
                if not act.ranges or gap:
                    act.ranges.append([lineno, lineno])
                else:
                    act.ranges[-1][1] = lineno
                gap = False
            if ts:
                act.last_ts = max(act.last_ts, ts)
                if track_lines:        # first and last *dated* record of the slice, 1-based.
                    if not act.start_line:     # undated records (summaries, file-history) pass
                        act.start_line = lineno  # the time filter untouched and would widen the
                    act.end_line = lineno        # range to the whole file.
            t = d.get("type")
            cwd = d.get("cwd")

            if t == "assistant":
                msg = d.get("message") or {}
                usage = msg.get("usage") or {}
                rid = d.get("requestId")
                if (rid or msg.get("id")) and usage:
                    turn = Turn(
                        request_id=rid, ts=ts, session_id=d.get("sessionId") or d.get("session_id") or "",
                        cwd=cwd, model=msg.get("model"), is_sidechain=bool(d.get("isSidechain")),
                        effort=d.get("effort"), version=d.get("version"))
                    for src, dst in USAGE_MAP.items():
                        setattr(turn, dst, int(usage.get(src) or 0))
                    # newer builds split cache writes into ephemeral buckets
                    cc = usage.get("cache_creation")
                    if isinstance(cc, dict):
                        bucket = sum(int(v or 0) for k, v in cc.items()
                                     if k.endswith("input_tokens"))
                        turn.cache_creation_tokens = max(turn.cache_creation_tokens, bucket)
                    turn.message_id = msg.get("id")
                    if isinstance(d.get("costUSD"), (int, float)):
                        turn.cost_usd = float(d["costUSD"])
                    turn.context_size = (turn.input_tokens + turn.cache_read_tokens
                                         + turn.cache_creation_tokens)
                    act.turns.append(turn)
                text = _text_of(msg.get("content"))
                if text.strip():
                    act.assistant_text.append({"ts": ts, "text": text})
                for b in (msg.get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        ti = b.get("input") or {}
                        for p in paths_from_tool(b.get("name") or "", ti, cwd):
                            act.files[p] = act.files.get(p, 0) + 1
                        if b.get("name") == "Bash" and isinstance(ti.get("command"), str):
                            entry = {"ts": ts, "cmd": ti["command"][:400],
                                     "description": (ti.get("description") or "")[:200]}
                            act.commands.append(entry)
                            if b.get("id"):
                                pending_tool[b["id"]] = entry

            elif t == "user":
                origin = d.get("origin") or {}
                msg = d.get("message") or {}
                if origin.get("kind") == "human":
                    text = _text_of(msg.get("content")).strip()
                    if text and not text.startswith("<command-message>"):
                        act.prompts.append({"ts": ts, "text": text})
                res = d.get("toolUseResult")
                if isinstance(res, dict):
                    err = (res.get("stderr") or "").strip()
                    interrupted = bool(res.get("interrupted"))
                    if err or interrupted:
                        src = None
                        for b in (msg.get("content") or []):
                            if isinstance(b, dict) and b.get("tool_use_id") in pending_tool:
                                src = pending_tool.pop(b["tool_use_id"])
                                break
                        act.failures.append({
                            "ts": ts,
                            "cmd": (src or {}).get("cmd", ""),
                            "description": (src or {}).get("description", ""),
                            "error": ("interrupted" if interrupted else err)[:300],
                        })
    return act


def dedupe(turns: list[Turn]) -> list[Turn]:
    """One API call can appear several times in the log (streaming updates, sidechain replays).

    Same rule as ccusage (MIT, ccusage/ccusage — rust/adapters/claude/src/lib.rs): key on
    (message id, request id, session id) and keep the copy with the largest token total,
    preferring the main chain over a sidechain replay.
    """
    best: dict[tuple, Turn] = {}
    for t in turns:
        k = t.dedupe_key()
        cur = best.get(k)
        if cur is None:
            best[k] = t
            continue
        if cur.is_sidechain != t.is_sidechain:
            if cur.is_sidechain:
                best[k] = t
            continue
        if t.total_tokens > cur.total_tokens:
            best[k] = t
    return list(best.values())


def iter_sessions(projects_dir: str) -> Iterator[str]:
    for root, _dirs, files in os.walk(projects_dir):
        for fn in files:
            if fn.endswith(".jsonl"):
                yield os.path.join(root, fn)


def replay(path: str) -> list[tuple[float, str, dict]]:
    """Chronological (ts, kind, payload) for offline task marking of an old transcript.

    kind 'p' — a human prompt {text, cwd}; kind 't' — a tool call that touched files {paths}.
    Same two inputs the live hooks get, so `ingest --mark` and the hooks split identically.
    """
    out: list[tuple[float, str, dict]] = []
    if not path or not os.path.isfile(path):
        return out
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                d = json.loads(raw.decode("utf-8", "ignore"))
            except Exception:
                continue
            ts = iso_to_ts(d.get("timestamp"))
            if not ts:
                continue
            t = d.get("type")
            if t == "user" and (d.get("origin") or {}).get("kind") == "human":
                text = _text_of((d.get("message") or {}).get("content")).strip()
                if text and not text.startswith("<command-message>"):
                    out.append((ts, "p", {"text": text[:2000], "cwd": d.get("cwd")}))
            elif t == "assistant":
                for b in ((d.get("message") or {}).get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        paths = paths_from_tool(b.get("name") or "", b.get("input") or {},
                                                d.get("cwd"))
                        if paths:
                            out.append((ts, "t", {"paths": paths}))
    return out
