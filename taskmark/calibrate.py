"""Threshold calibration — the thresholds come from the user's own data, not from constants.

Why: a person with five-minute prompts and a person with hour-long ones need different
"long pause"; a person working in one repo and a person hopping between servers need
different file-overlap limits. Hand-tuned numbers would only fit whoever tuned them.

How: every prompt writes its signal values into `signal_samples` (pause, file overlap,
cosine to the task centroid). Once there are enough samples, thresholds are taken as
percentiles of those distributions — a pause is "long" if it is in the top 10% of that
user's pauses, similarity is "lost" if it is in the bottom 10% of that user's similarities.
No labelling is required.

Stability matters more than freshness: a criterion that moves every day makes periods
incomparable. So calibration is computed once when enough data has accumulated, written
with a version number, and then re-computed only on demand (`taskmark calibrate`) or when
the sample count has grown several-fold. Every task records the calibration version it was
detected under; periods across versions are not comparable and stage 2 must not mix them.
"""
from __future__ import annotations

import json
import statistics
import time
from typing import Iterable

from . import config


def add_sample(con, session_id: str, kind: str, value: float, boundary: bool) -> None:
    con.execute("INSERT INTO signal_samples (ts, session_id, kind, value, was_boundary) "
                "VALUES (?,?,?,?,?)", (time.time(), session_id, kind, float(value),
                                       1 if boundary else 0))


def _percentile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        raise ValueError("no samples")
    k = (len(vals) - 1) * (q / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def current(con) -> dict:
    """Active calibration, or the built-in defaults when there isn't one yet."""
    row = con.execute("SELECT * FROM calibration ORDER BY version DESC LIMIT 1").fetchone()
    if not row:
        return {"version": 0, **config.DEFAULT_THRESHOLDS}
    params = json.loads(row["params"])
    # defaults first: a calibration written by an older version knows nothing about
    # thresholds added since, and a missing key must not read as None.
    return {"version": int(row["version"]), **config.DEFAULT_THRESHOLDS, **params}


def _samples(con, kind: str, limit: int = 5000) -> list[float]:
    return [float(r["value"]) for r in con.execute(
        "SELECT value FROM signal_samples WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit))]


def quality(con, thresholds: dict | None = None) -> dict:
    """How well the current rules match the corrections that arrived afterwards.

    Only labelled decisions count. `rule_boundary` is what the rules answered; `label` is
    what Claude or the user said it should have been. No model is run here — this is the
    scoreboard a tuning pass is judged against, before and after.
    """
    rows = con.execute("SELECT rule_boundary, label FROM prompt_signals "
                       "WHERE label IS NOT NULL").fetchall()
    tp = sum(1 for r in rows if r["rule_boundary"] and r["label"] == "new")
    fp = sum(1 for r in rows if r["rule_boundary"] and r["label"] == "same")
    fn = sum(1 for r in rows if not r["rule_boundary"] and r["label"] == "new")
    tn = sum(1 for r in rows if not r["rule_boundary"] and r["label"] == "same")
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {"labelled": len(rows), "true_split": tp, "false_split": fp,
            "missed_split": fn, "true_continue": tn,
            "precision": prec, "recall": rec, "f1": f1}


def set_thresholds(con, params: dict, note: str = "model") -> dict:
    """Write a new calibration version. Used by a tuning pass; every version is kept, so
    a bad one is undone by writing the previous numbers again."""
    cur = current(con)
    merged = {k: v for k, v in cur.items() if k != "version"}
    for k, v in params.items():
        if k in config.DEFAULT_THRESHOLDS or k in ("model",):
            merged[k] = v
    version = int(cur.get("version") or 0) + 1
    con.execute("INSERT INTO calibration (version, created_at, params, samples, note) "
                "VALUES (?,?,?,?,?)",
                (version, time.time(), json.dumps(merged, ensure_ascii=False),
                 int(merged.get("samples") or 0), note))
    merged["version"] = version
    return merged


def compute(con, *, force: bool = False) -> dict | None:
    """Recompute thresholds from the collected samples. Returns the new calibration or None."""
    pauses = _samples(con, "pause")
    overlaps = _samples(con, "file_overlap")
    cosines = _samples(con, "cosine")

    if len(pauses) < config.CALIBRATION_MIN_SAMPLES and not force:
        return None

    cur = current(con)
    if not force:
        prev_n = int(cur.get("samples") or 0)
        # don't move the criterion for every new prompt: only when the data has grown a lot
        if prev_n and len(pauses) < prev_n * config.CALIBRATION_GROWTH:
            return None

    params = dict(config.DEFAULT_THRESHOLDS)
    if len(pauses) >= config.CALIBRATION_MIN_SAMPLES:
        params["pause_soft"] = max(config.PAUSE_FLOOR_SECONDS,
                                   _percentile(pauses, config.PCT_PAUSE_SOFT))
        params["pause_hard"] = max(params["pause_soft"] * 2,
                                   _percentile(pauses, config.PCT_PAUSE_HARD))
    if len(overlaps) >= config.CALIBRATION_MIN_SAMPLES:
        params["file_overlap"] = min(0.35, _percentile(overlaps, config.PCT_FILE_OVERLAP))
    if len(cosines) >= config.CALIBRATION_MIN_SAMPLES:
        params["cosine_min"] = _percentile(cosines, config.PCT_COSINE)
    if len(pauses) >= config.CALIBRATION_MIN_SAMPLES:
        # "a gap this person takes routinely" — the floor below which a pause proves nothing.
        params["pause_floor"] = max(config.PAUSE_FLOOR_SECONDS,
                                    _percentile(pauses, config.PCT_PAUSE_FLOOR))
    terms = _samples(con, "prompt_terms")
    if len(terms) >= config.CALIBRATION_MIN_SAMPLES:
        # "a short reply" is relative too: some people type paragraphs, some three words.
        params["min_prompt_terms"] = max(3.0, _percentile(terms, config.PCT_PROMPT_TERMS))
    params["samples"] = len(pauses)
    params["model"] = config.EMBED_MODEL

    version = int(cur.get("version") or 0) + 1
    con.execute("INSERT INTO calibration (version, created_at, params, samples, note) "
                "VALUES (?,?,?,?,?)",
                (version, time.time(), json.dumps(params, ensure_ascii=False), len(pauses),
                 "forced" if force else "auto"))
    params["version"] = version
    return params


def feedback(con, kind: str, task_id: int | None, session_id: str | None) -> None:
    """Optional human correction: 'same task' / 'boundary here'. Never required."""
    con.execute("INSERT INTO feedback (ts, session_id, task_id, kind) VALUES (?,?,?,?)",
                (time.time(), session_id, task_id, kind))


def describe(cal: dict) -> str:
    pause_soft = cal.get("pause_soft", 0)
    pause_hard = cal.get("pause_hard", 0)
    return (f"calibration v{cal.get('version', 0)} "
            f"(samples: {cal.get('samples', 0)}): "
            f"pause soft {pause_soft / 60:.0f}min / hard {pause_hard / 60:.0f}min, "
            f"file overlap ≤{cal.get('file_overlap', 0):.2f}, "
            f"cosine ≤{cal.get('cosine_min', 0):.3f}")
