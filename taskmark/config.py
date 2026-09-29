"""Paths and tunables. Everything lives under ~/.claude/taskmark, nothing leaves the machine."""
from __future__ import annotations

import os
from pathlib import Path

HOME_DIR = Path(os.environ.get("TASKMARK_HOME") or (Path.home() / ".claude" / "taskmark"))
DB_PATH = HOME_DIR / "taskmark.db"
LOG_PATH = HOME_DIR / "taskmark.log"
STATE_DIR = HOME_DIR / "state"          # per-session scratch (current task, manual split flag)

# Where research notes are written, relative to the project directory.
RESEARCH_SUBDIR = Path(".claude") / "research"

# --- boundary rules (1.2): structure + embeddings, thresholds calibrated per user ----
# Starting values measured on the author's own logs (532 prompts, 12 days). They are only
# defaults: after CALIBRATION_MIN_SAMPLES prompts each user gets their own percentiles.
DEFAULT_THRESHOLDS = {
    "pause_hard": 3600.0,      # gap that ends a task on its own
    "pause_soft": 1500.0,      # gap that counts as one weak signal
    "pause_floor": 600.0,      # below this a gap is never "long" FOR THIS USER (calibrated)
    "file_overlap": 0.05,      # Jaccard between the last window's files and the task's files
    "cosine_min": 0.25,        # similarity of the prompt to the task centroid (model below)
    "min_prompt_terms": 5.0,   # shorter replies carry no topic — calibrated per user
    "boundary_score": 2.0,     # evidence needed to end a task (see SIGNAL_WEIGHT_*)
    "samples": 0,
}

# Percentiles used when calibrating on the user's own distribution.
PCT_PAUSE_SOFT = 85           # a pause in the longest 15% of this user's pauses
PCT_PAUSE_HARD = 97
PCT_FILE_OVERLAP = 10         # the 10% least overlapping windows
PCT_COSINE = 10               # the 10% least similar prompts
PCT_PAUSE_FLOOR = 60          # a gap this user takes routinely is not evidence of anything
PCT_PROMPT_TERMS = 30         # "short reply" = shorter than 70% of this user's prompts
PAUSE_FLOOR_SECONDS = 300     # absolute sanity floor only — a 3-minute gap is never "long"
                              # for anybody. The floor that actually matters is `pause_floor`
                              # in the thresholds: it is calibrated from this user's own
                              # pauses (see PCT_PAUSE_FLOOR), because "a long break" means
                              # ten minutes to one person and two hours to another.

CALIBRATION_MIN_SAMPLES = 200   # ~2 weeks of normal use before the first calibration
CALIBRATION_GROWTH = 3.0        # re-calibrate only when samples grew this many times

MIN_TURNS_BEFORE_SPLIT = 3      # don't cut a task that has barely started
MIN_FILES_FOR_OVERLAP = 4       # judge file overlap only once both sides touched enough files
MIN_FILES_FOR_SAMPLE = 2        # ...but already MEASURE it from two files a side, so the
                                # calibrator is not starved: acting on a noisy 2-file window
                                # would split wrongly, recording it costs nothing.

CONTINUITY_MIN = 0.10           # share of the prompt's words already seen in this task
MIN_PROMPT_TERMS = 5            # default floor; `min_prompt_terms` in the thresholds wins.
                                # Ignore short replies — too little text to judge a topic by.
                                # Was 4: "я так понял это не обязательные поля" clears four
                                # terms easily, yet says nothing about the subject — it is a
                                # follow-up inside the same work. Someone who writes long
                                # prompts needs a higher floor, someone terse a lower one,
                                # hence the calibrated value. Also guards the embedding
                                # layer (see EMBED_MIN_PROMPT_TERMS). 8 was tried and was
                                # too deaf: "посмотри почту user@example.com на сервере mail"
                                # is seven tokens and a genuine switch of subject. 5 still
                                # silences "да сделай" / "дай ссылку на фид".
MIN_VOCAB_TERMS = 40            # ...and don't judge until the task has a vocabulary
VOCAB_CAP = 800
GAP_HISTORY = 12

RETURN_WINDOW_SECONDS = 2 * 3600  # touching the old files again within this = same task
RETURN_FILE_OVERLAP = 0.30        # ...and at least this share of files shared, so a new task
                                  # that happens to touch one common file is not swallowed.
                                  # Deliberately NOT the calibrated `file_overlap`: that one
                                  # answers the opposite question (little in common = split).
RESUME_MIN_SCORE = 0.5            # `mark resume "<topic>"`: share of its words in the title, or cosine

# --- how the signals add up ------------------------------------------------
# Each signal is worth a score; a task ends when the total reaches BOUNDARY_SCORE.
# Replaces the older "2 strong, or 1 strong + 1 weak" rule, which treated `words` and
# `topic` as independent evidence — they are not: both read the same sentence, and a
# reworded follow-up trips both at once. They now share one group with a single vote.
SIGNAL_WEIGHT_STRONG = 1.0
SIGNAL_WEIGHT_WEAK = 0.5
BOUNDARY_SCORE = 2.0                # default only; `boundary_score` in the thresholds wins
TEXT_SIGNALS = ("words", "topic")   # one vote between them, not two

# --- learning from corrections ---------------------------------------------
# Percentiles need no labels, which is why they were the first layer — but they assume a
# fixed SHARE of prompts are boundaries (top 15% of pauses, bottom 10% of similarities),
# and that share is a property of the person, not of the data. Whenever enough corrections
# have accumulated (`mark same` / `mark new` / a merged-back task), the thresholds are
# chosen to match those corrections instead — that is what makes the plugin fit a way of
# working it has never seen.
SUPERVISED_MIN_LABELS = 40      # below this there is nothing to learn from; percentiles stay
SUPERVISED_MIN_PER_CLASS = 8    # ...and both answers must be represented
PROMPT_SIGNALS_KEEP = 5000      # rows of the labelled table to keep

# Embeddings (second layer). all-MiniLM-L6-v2 was measured first and rejected: on Russian /
# Estonian prompts it scored an unrelated topic higher than the same sentence translated.
# Multilingual model, CPU, ~450 MB, loaded lazily; without sentence-transformers the plugin
# falls back to rules only.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_MAX_PROMPT_CHARS = 600
EMBED_MIN_PROMPTS = 2           # need a centroid of at least this many prompts
EMBED_MIN_PROMPT_TERMS = MIN_PROMPT_TERMS   # ...and enough words in the prompt itself.
                                # A centroid cannot judge "да сделай" or "ок": such a reply
                                # carries no topic, so its cosine is low by construction and
                                # the signal fires on continuations. Measured: of 8 false
                                # boundaries in one day, 5 came from short follow-ups.

# Optional, empty by default: one explicit switch phrase per line.
MARKERS_FILE = HOME_DIR / "markers.txt"

# --- research notes (1.3) ---------------------------------------------------
# These word lists only decide what gets *quoted* in a note — never where a task ends.
# The hard evidence in a note (failed commands, files, shell history) needs no words at all.
FAILURE_MARKERS = [
    "не сработал", "не работает", "не помогло", "ошибка", "упал", "сломал", "откатил",
    "does not work", "didn't work", "failed", "error", "broken", "rolled back", "reverted",
    "ei tööta", "viga",
]
DECISION_MARKERS = [
    "решили", "решил", "договорились", "вывод", "итог", "выбрали", "оставляем",
    "decided", "conclusion", "we will", "chose", "keep as is", "otsustasime",
]
# Add your own in ~/.claude/taskmark/note-words.txt (one per line, "fail:" or "decide:" prefix).

MAX_NOTE_ITEMS = 40                     # cap per section in a research note

LOCK_WAIT_SECONDS = 30.0                # longest a hook waits for another one on the same session
ORPHAN_IDLE_SECONDS = 2 * 3600          # a session this quiet whose task is still open was closed
                                        # without SessionEnd (window killed): close the task for it
ORPHAN_MAX_AGE_DAYS = 14                # older open tasks are left alone (imported history)
ORPHAN_PER_START = 3                    # each close reads a transcript; keep session start quick


DISABLED_FLAG = HOME_DIR / "disabled"


def is_disabled() -> bool:
    """Kill switch. Either the env var or the flag file makes every hook a no-op;
    nothing is recorded, nothing is written, the data already collected stays put."""
    if os.environ.get("TASKMARK_OFF") or os.environ.get("TASKMARK_DISABLED"):
        return True
    return DISABLED_FLAG.exists()


def ensure_dirs() -> None:
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
CENTROID_PROMPTS = 12           # centroid is built from the task's recent prompts

PAUSE_RHYTHM_FACTOR = 5         # "long for this session" = this many times the median gap
