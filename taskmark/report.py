"""Stage 2: the numbers. Built now so the code is ready; it only becomes meaningful with history.

Metrics per task and per period:
  · tokens by the four categories, and cost in EUR (rates from pricing.py, never hard-coded)
  · cache-hit share  = cache_read / (cache_read + cache_creation + input)
  · output/input ratio = output / (input + cache_read + cache_creation)
  · subagent share   = tokens spent in sidechain turns
  · turns per task, prompts per task, average context size

Comparison with the trailing 2–4 weeks, and a plain-words decomposition of why the cost per
task moved — the four causes we agreed on:
  model change        output/input ratio moved while turns per task did not
  prompt/flow change  turns per task moved
  broken caching      cache-hit share fell
  context pile-up     average context size grew

Two rules the report obeys, because breaking them produces confident nonsense:
  1. periods detected under different calibration versions are never compared (the criterion
     changed, so "tokens per task" means different things on the two sides);
  2. a period with too few tasks is reported as "not enough data", not as a trend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import pricing

MIN_TASKS_FOR_TREND = 8


@dataclass
class Totals:
    turns: int = 0
    tasks: int = 0
    prompts: int = 0
    input: int = 0
    output: int = 0
    cache_creation: int = 0
    cache_read: int = 0
    sidechain_tokens: int = 0
    context_sum: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    unpriced: set = field(default_factory=set)
    models: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input + self.output + self.cache_creation + self.cache_read

    @property
    def cache_hit(self) -> float | None:
        denom = self.cache_read + self.cache_creation + self.input
        return self.cache_read / denom if denom else None

    @property
    def out_in(self) -> float | None:
        denom = self.input + self.cache_read + self.cache_creation
        return self.output / denom if denom else None

    @property
    def subagent_share(self) -> float | None:
        return self.sidechain_tokens / self.total_tokens if self.total_tokens else None

    @property
    def avg_context(self) -> float | None:
        return self.context_sum / self.turns if self.turns else None

    @property
    def turns_per_task(self) -> float | None:
        return self.turns / self.tasks if self.tasks else None

    @property
    def cost_per_task(self) -> float | None:
        return self.cost_usd / self.tasks if self.tasks else None


def collect(con, since: float, until: float, calib_version: int | None = None) -> Totals:
    t = Totals()
    sql = ("SELECT model, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, "
           "context_size, is_sidechain, cost_usd FROM turns WHERE ts >= ? AND ts < ?")
    args = [since, until]
    for r in con.execute(sql, args):
        t.turns += 1
        t.input += r["input_tokens"]
        t.output += r["output_tokens"]
        t.cache_creation += r["cache_creation_tokens"]
        t.cache_read += r["cache_read_tokens"]
        t.context_sum += r["context_size"]
        if r["is_sidechain"]:
            t.sidechain_tokens += (r["input_tokens"] + r["output_tokens"]
                                   + r["cache_creation_tokens"] + r["cache_read_tokens"])
        m = r["model"] or "unknown"
        t.models[m] = t.models.get(m, 0) + 1
        own = r["cost_usd"]
        if own is not None:
            t.cost_usd += float(own)
        else:
            c = pricing.cost_usd(m, {"input": r["input_tokens"], "output": r["output_tokens"],
                                     "cache_creation": r["cache_creation_tokens"],
                                     "cache_read": r["cache_read_tokens"]})
            if c is None:
                # placeholder models ("<synthetic>") carry no real tokens — ignore them,
                # otherwise one such row makes the whole period's cost "unknown"
                if (r["input_tokens"] + r["output_tokens"] + r["cache_creation_tokens"]
                        + r["cache_read_tokens"]) > 0:
                    t.cost_known = False
                    t.unpriced.add(m)
            else:
                t.cost_usd += c
    q = "SELECT COUNT(*) n, COALESCE(SUM(prompts),0) p FROM tasks WHERE started_at >= ? AND started_at < ? AND merged_into IS NULL"
    a2 = [since, until]
    if calib_version is not None:
        q += " AND calib_version = ?"
        a2.append(calib_version)
    row = con.execute(q, a2).fetchone()
    t.tasks, t.prompts = int(row["n"]), int(row["p"])
    return t


def calib_versions(con, since: float, until: float) -> list[int]:
    return [int(r["calib_version"]) for r in con.execute(
        "SELECT DISTINCT calib_version FROM tasks WHERE started_at >= ? AND started_at < ? "
        "AND merged_into IS NULL ORDER BY calib_version", (since, until))]


def _pct(cur: float | None, prev: float | None) -> str:
    if cur is None or prev in (None, 0):
        return "—"
    d = (cur - prev) / prev * 100
    return f"{d:+.0f}%"


def explain(cur: Totals, prev: Totals) -> list[str]:
    """Why did cost per task move? Only claims what the numbers support."""
    out: list[str] = []
    if not (cur.tasks and prev.tasks):
        return ["not enough tasks to compare"]

    def moved(a, b, thr=0.15):
        if a is None or not b:
            return 0.0
        return (a - b) / b if abs((a - b) / b) >= thr else 0.0

    d_ratio = moved(cur.out_in, prev.out_in)
    d_turns = moved(cur.turns_per_task, prev.turns_per_task)
    d_cache = moved(cur.cache_hit, prev.cache_hit, 0.10)
    d_ctx = moved(cur.avg_context, prev.avg_context)

    if d_ratio and not d_turns:
        out.append(f"output/input ratio moved {d_ratio*100:+.0f}% with turns per task unchanged "
                   f"— looks like a model or effort change")
    if d_turns:
        out.append(f"turns per task moved {d_turns*100:+.0f}% — the way tasks are being worked "
                   f"changed (prompts, tools, scope)")
    if d_cache < 0:
        out.append(f"cache-hit share fell {d_cache*100:.0f}% — caching is being broken "
                   f"(context edited early, or sessions restarted often)")
    if d_ctx > 0:
        out.append(f"average context grew {d_ctx*100:+.0f}% — context is piling up, "
                   f"clearing it earlier would help")
    if not out:
        out.append("no single cause stands out — the differences are within 10-15%")
    return out


def period_report(con, days: int = 7, compare_days: int | None = None) -> str:
    now = time.time()
    since = now - days * 86400
    cur = collect(con, since, now)
    lines = [f"Period: last {days} days"]

    versions = calib_versions(con, since, now)
    if len(versions) > 1:
        lines.append(f"⚠ tasks in this period were detected under calibration versions "
                     f"{versions} — task-level numbers are not comparable across them")

    eur = pricing.eur_per_usd()
    cost = (f"{cur.cost_usd * eur:.2f} EUR" if eur
            else f"{cur.cost_usd:.2f} USD")
    if cur.unpriced:
        cost += " (partial: no price for " + ", ".join(sorted(cur.unpriced)) + ")"
    lines += [
        f"  tasks {cur.tasks}, turns {cur.turns}, prompts {cur.prompts}",
        f"  tokens: in {cur.input:,} · out {cur.output:,} · cache write {cur.cache_creation:,} "
        f"· cache read {cur.cache_read:,}",
        f"  cost: {cost}" + (f" · {cur.cost_usd / cur.tasks * (eur or 1):.2f} per task"
                             if cur.tasks else ""),
        f"  cache-hit {cur.cache_hit:.1%}" if cur.cache_hit is not None else "  cache-hit —",
        f"  output/input {cur.out_in:.3f}" if cur.out_in is not None else "  output/input —",
        f"  subagents {cur.subagent_share:.1%}" if cur.subagent_share is not None else "  subagents —",
        f"  avg context {cur.avg_context:,.0f} tokens" if cur.avg_context else "  avg context —",
        f"  turns per task {cur.turns_per_task:.1f}" if cur.turns_per_task else "  turns per task —",
        f"  models: " + ", ".join(f"{m} ({n})" for m, n in
                                  sorted(cur.models.items(), key=lambda kv: -kv[1])[:4]),
    ]

    # the one-month test of the whole idea: notes that are written but never read are waste
    ev = {r["kind"]: int(r["n"]) for r in con.execute(
        "SELECT kind, COUNT(*) n FROM events WHERE ts >= ? AND kind IN "
        "('notes-offered','note-read') GROUP BY kind", (since,))}
    written = con.execute("SELECT COUNT(*) FROM tasks WHERE started_at >= ? AND note_path IS NOT NULL",
                          (since,)).fetchone()[0]
    lines.append(f"  notes: written {written} · offered after /clear or compaction "
                 f"{ev.get('notes-offered', 0)} times · read back {ev.get('note-read', 0)} times")

    dec = con.execute("SELECT who, what, why FROM decisions WHERE ts>=? ORDER BY ts", (since,)).fetchall()
    if dec:
        lines += ["", f"Decisions recorded ({len(dec)}):"]
        for d in dec[:15]:
            lines.append(f"  · [{d['who']}] {d['what']}" + (f" — {d['why']}" if d["why"] else ""))

    cmp_days = compare_days or days * 3
    prev = collect(con, since - cmp_days * 86400, since)
    lines += ["", f"Compared with the preceding {cmp_days} days:"]
    if prev.tasks < MIN_TASKS_FOR_TREND or cur.tasks < MIN_TASKS_FOR_TREND:
        lines.append(f"  not enough data yet ({cur.tasks} vs {prev.tasks} tasks; "
                     f"need {MIN_TASKS_FOR_TREND} on each side)")
        return "\n".join(lines)

    lines += [
        f"  cost per task   {_pct(cur.cost_per_task, prev.cost_per_task)}",
        f"  turns per task  {_pct(cur.turns_per_task, prev.turns_per_task)}",
        f"  cache-hit       {_pct(cur.cache_hit, prev.cache_hit)}",
        f"  output/input    {_pct(cur.out_in, prev.out_in)}",
        f"  avg context     {_pct(cur.avg_context, prev.avg_context)}",
        "",
    ] + [f"  · {x}" for x in explain(cur, prev)]
    return "\n".join(lines)


def tasks_report(con, days: int = 7, limit: int = 15) -> str:
    since = time.time() - days * 86400
    rows = con.execute(
        "SELECT t.id, t.title, t.started_at, t.prompts, t.calib_version, t.note_path, "
        "  (SELECT COUNT(*) FROM turns u WHERE u.task_id=t.id) turns, "
        "  (SELECT COALESCE(SUM(u.input_tokens+u.output_tokens+u.cache_creation_tokens+"
        "     u.cache_read_tokens),0) FROM turns u WHERE u.task_id=t.id) tok "
        "FROM tasks t WHERE t.started_at >= ? AND t.merged_into IS NULL "
        "ORDER BY tok DESC LIMIT ?", (since, limit)).fetchall()
    if not rows:
        return "no tasks recorded yet"
    out = [f"Heaviest tasks of the last {days} days:"]
    for r in rows:
        when = time.strftime("%d.%m %H:%M", time.localtime(r["started_at"]))
        out.append(f"  {when}  {r['tok']:>12,} tok  {r['turns']:>4} turns  {r['prompts']:>3} prompts"
                   f"  {'note' if r['note_path'] else '    '}  {(r['title'] or '')[:52]}")
    return "\n".join(out)
