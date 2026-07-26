# Phase 1 — Core coder loop

[← Back to README](../../README.md) · [Docs index](../README.md) · [Build plan](../build-plans/PHASE-1.md)

> The riskiest core, proven in isolation: an agent that edits files, runs commands in a sandbox, and checks its own work with real tests — before any orchestration exists.

## What was built

- **Tool layer** (`app/tools/`) — a typed `Tool` protocol, `ToolResult` / `ToolContext`, a
  registry, and a `to_langchain_tool` adapter. Every call flows through a central
  **authorization pipeline** (`execute_tool`): schema-validate → path-jail → command
  allow/deny → approval hook → execute → truncate → trace.
- **Tools** — `read_file` / `write_file` / `edit_file` / `list_dir`, `search_code` (ripgrep
  with a pure-Python fallback), `run_command`, git (`status` / `diff` / `add` / `commit`),
  and `finish_task`.
- **Sandbox** (`app/tools/sandbox.py`, `infra/sandbox/Dockerfile`) — `DockerSandbox`
  (`--network=none`, mem/CPU/pids limits, timeouts — the security boundary per
  [ADR-0007](../adr/0007-sandboxed-execution.md)) with a config-flagged `SubprocessSandbox`
  fallback for machines without Docker.
- **Workspace lifecycle** (`app/workspace/`) — git-backed project dirs, a `base_commit`,
  `agent/run-<id>` branches, commit-per-task, and a cumulative diff.
- **Coder ReAct loop** (`app/agents/coder.py`) — grounds via tools, edits files, runs
  commands in the sandbox, and self-corrects until `finish_task` or a budget/loop guard trips.
- **Deterministic verify** (`app/verify/runner.py`) — auto-detects and runs checks
  (`compileall`, `pytest`, …) in the sandbox; a timeout counts as a failure. No LLM involved.
- **Budgets** (`app/agents/budget.py`) — step / wall-clock / token caps + no-progress
  detection, so a stuck loop always terminates.

## How it was verified

**End to end (live model):** the coder builds a running, test-passing Python project entirely
inside the Docker sandbox (live `qwen2.5-coder:7b`, converged in 4 steps), confirmed by the
independent verify runner — not by the coder's own say-so.

## Key decisions

- [ADR-0001 — Capability-bounded agents, not role-based multi-agent](../adr/0001-capability-bounded-agents.md)
- [ADR-0005 — Deterministic (non-LLM) verify node](../adr/0005-deterministic-verify.md)
- [ADR-0007 — Sandboxed execution](../adr/0007-sandboxed-execution.md)

---

[← Phase 0 — Foundations](phase-0-foundations.md) · Next: [Phase 2 — Orchestration →](phase-2-orchestration.md)
