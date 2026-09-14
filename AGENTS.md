# Repository and controller rules

- Execute DeepSeek tasks only when the user explicitly requests `deepseek_subagent` for the current task. Mentioning the tool, reviewing code or editing configuration does not authorize model calls.
- Within an authorized task, the controller may divide bounded independent work, submit background jobs and collect results without asking again for each subtask. Do not extend authorization to unrelated tasks.
- Default DeepSeek Flash effort is `max`; do not lower it unless requested. The controller retains architecture, scientific/model decisions, difficult root causes and final acceptance.
- Prefer `submit` or `batch`, continue useful independent work, and use bounded `wait` only when results are needed. Do not duplicate delegated work or end a task merely because it was submitted.
- Supply an absolute cwd, stable task IDs, clear objectives and acceptance criteria. Keep read-only tasks `allow_write=false`, `allow_shell=false`. Writes and shell require explicit scope and contracts.
- Dependencies on scientific judgment or key interface decisions require controller review before submitting the next task; queue `completed` is not that review.
- Treat `needs_host` as a handoff: claim once, execute an authorized real host tool, then resume using the same task ID and genuine evidence. Do not copy credentials or hidden reasoning. Do not recursively delegate from a subagent.
- Inspect uncertain results before retrying. Cancellation does not roll back existing effects. Retain project-specific resource controls.
- Run offline tests for code changes. Do not call paid models merely to test configuration.
- Never commit API keys, personal paths, user conversation data, runtime state, project-specific data or credentials. Keep examples synthetic and runtime artifacts ignored.
