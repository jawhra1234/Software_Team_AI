# Phase 1 — Core coder loop

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 0 — Foundations](phase-0-foundations.md) · [Next: Phase 2 — Orchestration →](phase-2-orchestration.md)

> *This is an **as-built** writeup. For the original forward-looking specification, see the [Phase 1 build plan](../build-plans/PHASE-1.md).*

The riskiest core, proven in isolation: an agent that edits files, runs commands in a sandbox,
and checks its own work with real tests — before any orchestration exists.

## What this phase adds

A tool layer with a central authorization pipeline, a hardened execution sandbox, a git-backed
workspace, the coder ReAct loop, a deterministic (non-LLM) verify runner, and budget/loop guards.

## Why it was needed

The hard, risky parts of a coding agent are tools, safe execution, and closed-loop verification —
not orchestration. Proving them standalone means Phase 2 wraps a *known-good* loop in a graph node
instead of debugging both at once.

## Architecture / how it works

- Every tool call flows through one **authorization pipeline** (`execute_tool`): schema-validate
  → path-jail → command allow/deny → approval hook → execute → truncate → trace. Failures are
  returned as data (`ToolResult(ok=False, …)`), never exceptions — the agent reads and reacts.
- The **coder** is a ReAct loop: it calls tools, edits files, runs commands in the sandbox, and
  self-corrects until it calls `finish_task` or a budget/loop guard trips.
- **Verify** is deliberately **no-LLM** ([ADR-0005](../adr/0005-deterministic-verify.md)): it
  auto-detects and runs the project's real checks (`compileall`, `pytest`, …) in the sandbox and
  reports pass/fail; a timeout counts as a failure. Truth comes from running the tests, not the
  model's self-assessment.

## Implementation

- `app/tools/` — `Tool` protocol, `ToolResult`/`ToolContext`, registry; `authorization.py`
  (the pipeline); `fs.py`, `search.py`, `shell.py`, `git.py`, `control.py` (the tools).
- `app/tools/sandbox.py`, `infra/sandbox/Dockerfile` — `DockerSandbox` (the security boundary)
  + `SubprocessSandbox` fallback.
- `app/workspace/` — git-backed project dirs, `base_commit`, `agent/run-<id>` branches,
  commit-per-task, cumulative diff.
- `app/agents/coder.py` — the ReAct loop; `app/agents/budget.py` — budget/no-progress guards.
- `app/verify/runner.py` — the deterministic verify runner.

## Configuration

```bash
SANDBOX__BACKEND=docker            # security boundary; 'subprocess' is the documented fallback
SANDBOX__MEM_LIMIT=1g              # sandbox resource caps (also CPUS / PIDS_LIMIT)
CODER__MAX_STEPS_PER_TASK=20       # per-task loop guards ...
CODER__MAX_WALL_CLOCK_S=900        # ... wall-clock ...
CODER__NO_PROGRESS_LIMIT=3         # ... and no-progress detection
```

## Testing and validation

- **Hermetic:** the tool authorization pipeline (path-jail, allow/deny, truncation), the fs
  tools incl. the 1 MiB size guard, the coder loop's budget/no-progress guards, and the verify
  runner — all unit-tested (`tests/test_tools_*.py`, `test_fs_tools.py`, `test_coder*.py`).

## Live validation

**End to end (live model):** the coder builds a running, test-passing Python project entirely
inside the Docker sandbox (live `qwen2.5-coder:7b`, converged in 4 steps), confirmed by the
independent verify runner (`tests/test_coder_e2e.py`, opt-in integration).

## What worked

The closed loop is real: the coder writes code, runs the actual tests, and self-corrects — and a
separate deterministic runner confirms the result, so "it works" isn't the model's opinion.

## Known limitations / honest findings

- The `edit_file` tool can be *misused* by a weak model into a runaway rewrite; a 1 MiB
  `write_file`/`edit_file` size guard (added in Phase 3 after live validation) bounds the damage.
  The tool is correct; the misuse is model behavior.
- The Docker sandbox is the security boundary; the `subprocess` fallback trades network isolation
  for convenience and is for dev machines without Docker only.

## Key engineering decisions

- [ADR-0001 — Capability-bounded agents, not role-based multi-agent](../adr/0001-capability-bounded-agents.md)
- [ADR-0005 — Deterministic (non-LLM) verify node](../adr/0005-deterministic-verify.md)
- [ADR-0007 — Sandboxed execution](../adr/0007-sandboxed-execution.md)

## Current status

✅ **Complete and verified.**

---

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 0 — Foundations](phase-0-foundations.md) · [Next: Phase 2 — Orchestration →](phase-2-orchestration.md)
