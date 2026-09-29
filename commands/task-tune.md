---
description: Re-calibrate task-boundary detection from the corrections collected so far
---

Tune the boundary detector to how **this** person works. The numbers in the plugin are
defaults measured on someone else; the corrections in the database are evidence about the
person in front of you.

1. Run `"${CLAUDE_PLUGIN_ROOT}/bin/taskmark" tune` and read all of it. It prints:
   - the thresholds in force now,
   - how they score against every correction so far (wrong splits, missed splits, F1),
   - which signal combinations fire most often,
   - each corrected decision: the prompt, the signal values, what the rules answered and
     what it should have been.

2. Look for the *pattern*, not the average. Useful questions:
   - Do the wrong splits share a shape? (short replies, a pause of a certain length, work
     that moved to another server mid-task, one signal that fires on its own)
   - Do the missed splits share one? (a new topic that reuses the same files, a switch
     announced in few words)
   - Which signal is doing the damage — and would moving its threshold fix the wrong
     splits without losing the right ones? Check that against the listed decisions; do not
     assume.

3. Apply what you concluded:

   ```
   "${CLAUDE_PLUGIN_ROOT}/bin/taskmark" calibrate --set pause_floor=900 cosine_min=0.18
   ```

   Thresholds you may set: `pause_hard`, `pause_soft`, `pause_floor`, `file_overlap`,
   `cosine_min`, `min_prompt_terms`, `boundary_score`. Every call writes a new calibration
   version; nothing is overwritten, so a bad guess is undone by setting the old numbers back.

4. Tell the person, in three or four lines: what was wrong, what you changed, what you
   expect to improve. If the data does not support a change, say so and change nothing —
   fewer than ~40 corrections is usually not enough to conclude anything.

Never edit the plugin's source from this command. Thresholds only.
