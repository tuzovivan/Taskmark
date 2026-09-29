# taskmark

A small Claude Code plugin that notices when you move to another task, writes down what the
finished task established, and keeps a local history of token usage.

It exists for one reason: not clearing the context is expensive, and clearing it loses the
research — which file does what, how the pieces connect, what was already tried and failed.
If that lands in a file, clearing stops being a loss.

**What it did for me** (one heavy user, Claude Code on Opus, ~550 requests a day, 13 days before
vs 11 days after): average context per request **504k → 270k tokens (−46%)**, cost per request
**about 40% lower** at the same model prices. Your numbers will depend on how you work — the
plugin measures them for you (`taskmark report`).

## Prior art

The JSONL parsing and token accounting follow **ccusage** (MIT, `ccusage/ccusage`), which has
solved this properly: the dedupe key is (message id, request id, session id), keeping the copy
with the largest token total; cache writes can arrive as ephemeral buckets under
`usage.cache_creation`; some records carry their own `costUSD`; prices come from LiteLLM's model
table rather than a hard-coded list. **claude-powerline** (MIT) was the reference for context
size — the last main-chain turn's `input + cache_read + cache_creation`, sidechain turns
excluded. **claude-devtools** (MIT) covers session, tool and subagent inspection.

No code was copied — those projects are Rust and TypeScript, this one is Python — only the
approach; see NOTICE. If you want usage statistics alone, use ccusage: it is better at that and
always will be. This plugin exists for the part none of them do — marking task boundaries and
dumping the research before the context is cleared.

## What it does

- **Marks task boundaries** — rules first, Claude as the second opinion.
  *Rules:* a pause (long in absolute terms, or long for this session's own rhythm), a move to a
  different directory tree, a changed git branch, work moving to another target (ssh host,
  container, project), directories that no longer overlap the ones the task was working in, and
  lexical continuity — how many of the new prompt's words this task has already seen.
  A boundary needs a score of two: one strong signal is not enough on its own.
  *Claude:* every prompt carries one line of context saying what the rules decided. Silence is
  agreement; when Claude disagrees it runs `taskmark mark new|same|resume` itself. In practice
  this is what does most of the work (see the numbers below). `model_marks: false` turns it off.
  *Embeddings (optional, off by default):* cosine between the prompt and the centroid of the
  task's prompts, multilingual model on CPU in a small resident worker.
- **Learns from corrections.** Every decision is stored with all its signals; each `mark same`,
  `mark new` or `/task-split` labels it. `taskmark tune` scores the thresholds against those
  labels and `/task-tune` lets Claude read the dataset and propose new ones
  (`taskmark calibrate --set pause_floor=900 …`). Each calibration is a version; tasks keep the
  version they were detected under, so periods stay comparable.
- **Writes a research note** to `.claude/research/YYYY-MM-DD-slug.md` when a task ends and before
  every compaction: files touched, commands worth keeping, what failed (with the error text),
  decisions, open questions, the repositories and commits involved. Built from facts in the
  transcript, not from a retelling. `/research-save` asks Claude for a fuller one while the
  context is still there. After `/clear`, compaction or a fresh start, the session gets a short
  list of recent notes (paths and titles, a few hundred tokens), so Claude can read the right one.
  Notes never reach git — a `.gitignore` with `*` is dropped into that directory on first write
  (`TASKMARK_NOTES_GIT=1` opts out) — and credentials that leak into shell history (`-pSecret`,
  `--password=`, `token=`, `?key=` in URLs, `Bearer …`, `sk-/hf_/ghp_` keys, `X-*-Key:` headers)
  are masked before anything is written.
- **Keeps a decision log.** `taskmark decide "what" --why "why"` — written by whoever decided,
  printed in the note and in the weekly report; `taskmark decisions [text]` searches it.
- **Records every turn** into SQLite at `~/.claude/taskmark/taskmark.db`: model, all four token
  categories, context size, subagent flag, task. `taskmark report --days=7` gives cost per task,
  cache hit rate, average context and a comparison with the preceding period (prices from
  LiteLLM's model table, EUR via frankfurter.dev — the only two network calls, both cached).

## Platforms

Written and tested on Linux. macOS should behave identically (untested) — same paths, same unix socket
for the embedding worker. On Windows the plugin runs but two things differ: the worker falls
back to a loopback TCP port (Python's unix-socket support there is not dependable), and hooks
are executed through `cmd.exe`, so they use `bin\taskmark.cmd`, which needs `python` on PATH.
Neither macOS nor Windows has been tried — if it misbehaves, run `taskmark off` and open an issue or just fork.

## Install

Requires Python 3.10+ and nothing else.

```
/plugin marketplace add tuzovivan/Taskmark
/plugin install taskmark@taskmark
```

Restart Claude Code. From then on it records turns, marks tasks and writes notes by itself.
Optional:

```bash
T=$(ls -d ~/.claude/plugins/cache/taskmark/taskmark/*/bin | tail -1)/taskmark
$T ingest --mark            # import and split your past sessions, so reports are not empty
pip install sentence-transformers && $T fetch-model   # embedding layer: ~450 MB model, ~1 GB RAM
```

## Commands

In a session: `/task-split`, `/research-save`, `/task-tune`. From a shell (`bin/taskmark`):

```bash
taskmark status                   # what has been recorded, how often Claude corrected the rules
taskmark report --days=7          # tokens, cost per task, cache hits, compared with before
taskmark note                     # write the research note for the open task now
taskmark same                     # this task is the previous one continued — merge them
taskmark decide "…" --why "…"     # record a decision
taskmark tune / calibrate         # thresholds against your own corrections
taskmark off / on                 # kill switch: hooks become no-ops, data is kept
```

## Settings

`~/.claude/taskmark/settings.json`, all optional:

```json
{"model_marks": true, "embeddings": false, "offer_notes": true,
 "hint_min_context": 150000, "hint_color": null}
```

## Turning it off

`taskmark off` (or `TASKMARK_OFF=1`) makes every hook return immediately — nothing is read,
nothing is written. Uninstalling the plugin stops it completely; the database and the notes are
plain files and stay where they are until you delete them.

## What it does not do

No web UI, no dashboard, no cloud, no local LLM, no support for other clients. It never clears
your context by itself — at a boundary with a large context it says `/clear` is safe now, that is all.

## How well it works

**The point of it.** Before the plugin I rarely cleared the context, for fear of losing what a
long session had established: 4 sessions in 13 days. With notes written automatically I clear it
several times a day: 33 sessions in the 11 days after. Average context per request went from 504k
to 270k tokens. Priced at one model's rates for both periods (the model got cheaper in between, and
that must not be counted), a request costs 37–41% less; output per request did not change, so the
work did not get smaller. This is one person and a before/after comparison, not a controlled test.

**Boundaries, honestly.** Five days of my own use (≈460 prompts): the rules on their
own mostly cut on pauses, and nearly half of those cuts were wrong. Claude, prompted by the
one-line hint, added 31 boundaries the rules missed, undid 10 wrong ones and sent 11 prompts back
to an earlier task. Embeddings decided about one boundary in those five days, which is why they
are off by default. Short conversational prompts ("ok, do it", "and the other server?") carry no
topic, and neither word counts nor sentence embeddings separate them reliably.

Two things were measured and rejected on the way: `all-MiniLM-L6-v2` (the obvious 90 MB choice)
is English-only — the same sentence in Russian and Estonian scored 0.26 while an unrelated Russian
sentence scored 0.69; and comparing exact file paths was useless in shell-heavy work (overlap was
0.00 at the 85th percentile). Directories fixed the latter.

## Status

A personal tool, published as is. Written for my own use, tested on my own sessions on Linux;
macOS and Windows are untested and there is no automated test suite. No support and no roadmap;
issues and forks welcome. Documentation of the internals, in Russian: `DOCS.ru.md`.

Licence: Apache 2.0.
