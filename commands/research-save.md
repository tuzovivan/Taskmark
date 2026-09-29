---
description: Write a research note for the current task while the context is still alive
---

Write what this session actually established, into `.claude/research/` in the project.

1. Run `"${CLAUDE_PLUGIN_ROOT}/bin/taskmark" note` — it writes the factual skeleton (files, commands,
   failures, decisions detected in the transcript) and prints the path of the file.
2. Open that file and fill it in from what you still have in context. Keep the structure,
   replace the placeholder bullets. Aim for what a person returning in a month needs:

   - **What this task was about** — one paragraph, plain words.
   - **Files touched** — keep the generated list, add one line per file: what it does and
     what was changed in it.
   - **How it works** — the mechanism you had to reconstruct: which function calls which,
     where the data comes from, which cron/service runs it. This is the part that costs
     the most to rediscover.
   - **What did not work** — every attempt that failed and why. Keep the exact error text.
     Do not omit the things that looked obvious and still failed.
   - **Decisions** — what was decided and the reason. Name the alternative that was rejected.
   - **Open questions** — what is still unverified.

3. Do not invent anything. If something was never established in this session, write
   "not verified" instead of guessing.
4. Answer with one line: the path of the file and how many sections you filled.
