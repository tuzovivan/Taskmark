"""Prices for the four token categories, from an external table — never hard-coded.

Source: LiteLLM's `model_prices_and_context_window.json`, the same table ccusage uses.
It carries the four rates we need per model:

    input_cost_per_token
    output_cost_per_token
    cache_creation_input_token_cost     (writing to the prompt cache)
    cache_read_input_token_cost         (reading from it — an order of magnitude cheaper)

The table is fetched once and cached under ~/.claude/taskmark/, with a TTL. If the network
is unavailable and there is no cache, cost is simply reported as unknown — a wrong number
would be worse than none. USD→EUR comes from Frankfurter (ECB reference rates), cached the
same way; a rate can also be pinned with TASKMARK_EUR_PER_USD.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from . import config

PRICES_URL = ("https://raw.githubusercontent.com/BerriAI/litellm/main/"
              "model_prices_and_context_window.json")
FX_URL = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=EUR"

PRICES_CACHE = config.HOME_DIR / "prices.json"
FX_CACHE = config.HOME_DIR / "fx.json"
PRICES_TTL = 7 * 24 * 3600
FX_TTL = 24 * 3600

RATE_KEYS = {
    "input": "input_cost_per_token",
    "output": "output_cost_per_token",
    "cache_creation": "cache_creation_input_token_cost",
    "cache_read": "cache_read_input_token_cost",
}


def _cached(path: Path, ttl: int, url: str) -> dict | None:
    config.ensure_dirs()
    if path.exists() and time.time() - path.stat().st_mtime < ttl:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        # a bare Python-urllib User-Agent gets 403 from some APIs (Frankfurter does this)
        req = urllib.request.Request(url, headers={"User-Agent": "taskmark/0.1 (+local)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception:
        if path.exists():                      # stale is better than nothing
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None


_TABLE: tuple[float, dict] | None = None        # (mtime of the cache file, parsed table)


def table() -> dict:
    """The price table, parsed once per process.

    _cached() re-reads and re-parses a 2.2 MB JSON file on every call, and rates_for() is
    called once per turn: a week's report parsed it ~3500 times and spent 90 s of CPU doing
    it. Keyed by mtime, so a refreshed cache is still picked up.
    """
    global _TABLE
    try:
        mtime = PRICES_CACHE.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _TABLE is not None and _TABLE[0] == mtime:
        return _TABLE[1]
    data = _cached(PRICES_CACHE, PRICES_TTL, PRICES_URL) or {}
    try:
        mtime = PRICES_CACHE.stat().st_mtime    # a fetch may have just rewritten it
    except OSError:
        pass
    _TABLE = (mtime, data)
    _RATES.clear()
    return data


_RATES: dict[str, dict | None] = {}             # model -> rates, memoised per process


def eur_per_usd() -> float | None:
    pinned = os.environ.get("TASKMARK_EUR_PER_USD")
    if pinned:
        try:
            return float(pinned)
        except ValueError:
            pass
    fx = _cached(FX_CACHE, FX_TTL, FX_URL)
    try:
        return float(fx["rates"]["EUR"])
    except Exception:
        return None


def rates_for(model: str | None) -> dict | None:
    """The four per-token rates for a model, or None when the table doesn't know it."""
    if not model:
        return None
    t = table()                                  # also clears _RATES when the table changed
    if model in _RATES:
        return _RATES[model]
    out = _rates_uncached(model, t)
    _RATES[model] = out
    return out


def _rates_uncached(model: str, t: dict) -> dict | None:
    entry = t.get(model)
    if entry is None:                          # try the provider-prefixed spellings
        for key in (f"anthropic/{model}", f"claude-3-5-{model}"):
            if key in t:
                entry = t[key]
                break
    if entry is None:                          # longest known key that this model starts with
        best = None
        for k in t:
            if model.startswith(k) and (best is None or len(k) > len(best)):
                best = k
        entry = t.get(best) if best else None
    if not isinstance(entry, dict):
        return None
    out = {}
    for name, key in RATE_KEYS.items():
        v = entry.get(key)
        if v is None and name == "cache_creation":
            v = entry.get("input_cost_per_token")        # some models bill writes as input
        if v is None and name == "cache_read":
            v = entry.get("input_cost_per_token")
        out[name] = float(v) if v is not None else None
    return out if out.get("input") is not None else None


def cost_usd(model: str | None, tokens: dict) -> float | None:
    """tokens: {'input':n, 'output':n, 'cache_creation':n, 'cache_read':n}"""
    r = rates_for(model)
    if not r:
        return None
    total = 0.0
    for name, rate in r.items():
        if rate is None:
            continue
        total += float(tokens.get(name, 0) or 0) * rate
    return total
