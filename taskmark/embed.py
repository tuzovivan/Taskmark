"""Embedding layer of the boundary detector (1.2, second layer).

A prompt is compared with the *centroid* of the prompts already seen in the open task.
Low similarity = the user is talking about something else.

Model: multilingual by necessity. The obvious choice from the spec, `all-MiniLM-L6-v2`,
was measured here first and fails outright on this user's languages: the same sentence in
Russian and Estonian scored 0.26 while a Russian sentence about a *different* topic scored
0.69 — it separates alphabets, not meanings. So the default is a multilingual model
(the same family Vilow uses in production: intfloat/multilingual-e5-*).

Notes that matter:
  · e5 wants the "query: " prefix — that is how it was trained.
  · e5 cosines are inflated: unrelated everyday sentences sit at 0.70-0.80, so a threshold
    can never be picked intuitively. It comes from the user's own distribution (calibrate.py).
  · everything is local and lazy: the model loads on first use, in a subprocess-friendly way,
    and if sentence-transformers is not installed the plugin simply runs on rules alone.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

from . import config

_model = None
_unavailable = False


def model_name() -> str:
    return os.environ.get("TASKMARK_EMBED_MODEL", config.EMBED_MODEL)


def model_present() -> bool:
    """Is the model already on disk? Nothing is ever downloaded behind the user's back.

    A first run must not silently pull ~450 MB over someone's connection, so the embedding
    layer stays off until the model is in the local cache. `taskmark fetch-model` (or any
    other way of putting it there) turns the layer on for good.
    """
    name = model_name()
    if os.path.isdir(name):                                # a local directory works too
        return True
    hub = Path(os.environ.get("HF_HUB_CACHE")
               or (Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
                   / "hub"))
    return (hub / ("models--" + name.replace("/", "--"))).is_dir()


def available() -> bool:
    """True when embeddings can be used at all (switched on, library present, model on disk).

    Checked with find_spec, never with a real import: importing sentence_transformers pulls
    in torch and costs ~3.5 s, and this runs in every hook process. Measured — that single
    import was the whole cost of the prompt hook (4.2 s → 0.1 s).
    """
    if _unavailable or os.environ.get("TASKMARK_NO_EMBED"):
        return False
    from . import hints
    if not hints.settings().get("embeddings"):
        return False
    try:
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is None:
            return False
    except Exception:
        return False
    return model_present()


def _get():
    global _model, _unavailable
    if _model is None and not _unavailable:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_name(), device="cpu")
        except Exception:
            _unavailable = True
            return None
    return _model


def _prefix(texts: Sequence[str]) -> list[str]:
    # e5 models expect a task prefix; other models are used as-is
    return [("query: " + t if "e5" in model_name() else t) for t in texts]


# --- resident worker ------------------------------------------------------
# Loading the model costs ~24 s, and every hook is a fresh process. So the model lives in a
# small daemon; hooks ask it over a unix socket with a short timeout and simply skip the
# embedding signal if the daemon is still warming up. Nothing leaves the machine.

# Unix sockets on Linux/macOS, a loopback TCP port on Windows (Python's AF_UNIX support
# there is not dependable). Same protocol either way.
_WINDOWS = sys.platform.startswith("win")


def _sock_path() -> str:
    config.ensure_dirs()
    return str(config.STATE_DIR / "embed.sock")


def _port_file():
    config.ensure_dirs()
    return config.STATE_DIR / "embed.port"


def _connect(timeout: float):
    if _WINDOWS:
        port = int(_port_file().read_text().strip())
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.settimeout(timeout)
        c.connect(("127.0.0.1", port))
        return c
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.settimeout(timeout)
    c.connect(_sock_path())
    return c


def _ask_daemon(text: str, timeout: float = 1.5) -> list[float] | None:
    # if the worker is not up yet, don't sit on a connect timeout — skip the signal this time
    if not _WINDOWS and not os.path.exists(_sock_path()):
        return None
    try:
        c = _connect(timeout)
        c.sendall(json.dumps({"text": text}).encode("utf-8") + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = c.recv(65536)
            if not chunk:
                break
            buf += chunk
        c.close()
        return json.loads(buf.decode("utf-8")).get("vector")
    except Exception:
        return None


IDLE_TIMEOUT = 1800.0           # the worker holds ~1.4 GB RSS; give it back after 30 min idle
SPAWN_COOLDOWN = 90.0           # loading takes ~25 s; don't spawn again while one is starting


def _lock_path():
    config.ensure_dirs()
    return config.STATE_DIR / "embed.lock"


def _try_lock():
    """Exclusive, non-blocking lock: one worker per data directory, ever.

    Measured failure without it: every hook that got no answer spawned another worker, and each
    new worker took the socket over from the previous one — 14 copies, ~19 GB of RAM, each
    waiting out its idle timeout. The lock is released by the OS when the process dies, so a
    crashed worker never blocks the next one.
    """
    fh = open(_lock_path(), "a+")
    try:
        if _WINDOWS:
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except OSError:
        fh.close()
        return None


def start_daemon() -> None:
    """Fire-and-forget: the next prompt will find the model ready."""
    try:
        marker = config.STATE_DIR / "embed.starting"
        if marker.exists() and time.time() - marker.stat().st_mtime < SPAWN_COOLDOWN:
            return                                  # one is already on its way
        probe = _try_lock()
        if probe is None:
            return                                  # a worker is alive (maybe still loading)
        probe.close()                               # free it for the worker we start now
        marker.write_text(str(time.time()), encoding="utf-8")
        subprocess.Popen([sys.executable, "-m", "taskmark.cli", "embed-daemon"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    except Exception:
        pass


def serve(idle_timeout: float = IDLE_TIMEOUT) -> int:
    """Run the worker: load the model once, answer until idle. At most one per data dir."""
    lock = _try_lock()
    if lock is None:
        return 0                                    # another worker owns this directory
    # load the model FIRST, then open the socket: while loading there is no socket, so hooks
    # skip the signal instantly instead of hanging on a worker that cannot answer yet
    if _get() is None:
        return 1
    path = _sock_path()
    if _WINDOWS:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))                 # loopback only
        _port_file().write_text(str(srv.getsockname()[1]), encoding="utf-8")
    else:
        try:
            os.unlink(path)                        # safe: we hold the lock, so it is stale
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        os.chmod(path, 0o600)
    srv.listen(4)
    srv.settimeout(idle_timeout)
    try:
        (config.STATE_DIR / "embed.starting").unlink()
    except FileNotFoundError:
        pass
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            break
        try:
            conn.settimeout(10)
            data = conn.recv(1 << 20).decode("utf-8", "ignore").strip()
            text = (json.loads(data) or {}).get("text", "") if data else ""
            vec = encode_one_local(text) if text else None
            conn.sendall(json.dumps({"vector": vec}).encode("utf-8") + b"\n")
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
    try:
        if _WINDOWS:
            _port_file().unlink(missing_ok=True)
        else:
            os.unlink(path)
    except Exception:
        pass
    return 0


def encode(texts: Sequence[str]) -> list[list[float]] | None:
    """L2-normalised vectors (so cosine == dot product), or None if unavailable."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return None
    m = _get()
    if m is None:
        return None
    try:
        vecs = m.encode(_prefix(texts), normalize_embeddings=True,
                        batch_size=8, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]
    except Exception:
        return None


def encode_one_local(text: str) -> list[float] | None:
    out = encode([text])
    return out[0] if out else None


def encode_one(text: str) -> list[float] | None:
    """Ask the resident worker; start it in the background if it is not running yet."""
    if not available():
        return None
    if os.environ.get("TASKMARK_EMBED_INLINE"):        # tests and one-off scripts
        return encode_one_local(text)
    v = _ask_daemon(text)
    if v is None:
        start_daemon()
    return v


def centroid(vectors: Iterable[Sequence[float]]) -> list[float] | None:
    """Mean vector, re-normalised — the 'what this task is about' point."""
    vs = [v for v in vectors if v]
    if not vs:
        return None
    dim = len(vs[0])
    acc = [0.0] * dim
    for v in vs:
        for i in range(dim):
            acc[i] += v[i]
    norm = sum(x * x for x in acc) ** 0.5
    if norm == 0:
        return None
    return [x / norm for x in acc]


def cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    return float(sum(x * y for x, y in zip(a, b)))
