---
description: Mark a task boundary — the next prompt starts a new task
---

Run:

```bash
"${CLAUDE_PLUGIN_ROOT}/bin/taskmark" split
```

Then tell the user in one line that the next prompt will be recorded as a new task,
and that the research note for the task just closed is written to `.claude/research/`.
