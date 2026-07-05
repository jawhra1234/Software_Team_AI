# Phase 1 — Core Coder loop (the highest-risk core)

> **Goal:** Prove the single most important and riskiest capability: an agentic loop that grounds in real files, edits a git-backed workspace, **runs commands in a sandbox**, and closes the loop with a deterministic verify — building and *running* a tiny project from a spec.
>
> **Why this phase comes second (before the graph):** If the tool + sandbox + verify loop doesn't work, no amount of orchestration matters. We validate the core mechanic in isolation, then structure it with LangGraph in Phase 2. Building the graph first would mean debugging orchestration and the core loop simultaneously — the debt this roadmap is designed to avoid.

## Objectives
1. The tool abstraction layer + authorization pipeline (ARCHITECTURE §6).
2. A sandbox for `run_command` (Docker `--network=none`, cwd-jail, timeouts, allowlist).
3. A git-backed workspace lifecycle (create/attach, branch, commit-per-task).
4. A single ReAct-style `coder` loop (not yet a graph node) using the Phase-0 provider.
5. A deterministic `verify` step that runs auto-detected checks and returns structured results.
6. End-to-end: from a small spec, the coder creates and runs a working tiny project.

## Scope
**In:** `Tool` protocol + registry, `ToolContext`, tools (`read_file`, `edit_file`, `write_file`, `list_dir`, `search_code` via ripgrep, `run_command`, `git_*`, `finish_task`), sandbox, workspace lifecycle, git workflow, coder ReAct loop, verify command detection + runner, budgets/step caps.
**Out (later phases):** LangGraph graph & checkpointer (Phase 2), HITL interrupts (Phase 2), RAG/`retrieve` (Phase 3 — `search_code` ripgrep is enough here), reviewer (Phase 4), UI (Phase 6). `retrieve` tool may be a stub.

## Prerequisites
- Phase 0 complete (provider, `structured_call`, logging/tracing, infra).
- `ripgrep` available in the sandbox image.

## Work breakdown & deliverables
| # | Task | Deliverable |
|---|---|---|
| 1.1 | `tools/base.py` — `Tool` protocol, `ToolResult`, `ToolContext`, `ToolRegistry`, `to_langchain_tool` adapter | Provider-agnostic tool interface |
| 1.2 | `tools/sandbox.py` — Docker-backed executor (`--network=none`, mem/CPU/time limits, workspace mount) | `run_in_sandbox(cmd, timeout)` |
| 1.3 | Authorization pipeline — schema validate → path-jail → allow/deny-list → execute → truncate → trace | Central `execute_tool()` all tools flow through |
| 1.4 | `tools/fs.py` — `read_file`, `edit_file` (patch/search-replace), `write_file`, `list_dir` | File tools jailed to workspace |
| 1.5 | `tools/search.py` — `search_code` via ripgrep (symbol index deferred to Phase 3) | Text/keyword search tool |
| 1.6 | `tools/git.py` — `status/diff/add/commit/checkout/branch` | Git ops over the workspace |
| 1.7 | `workspace/lifecycle.py` — create/attach, `git init`/clone, `base_commit`, `work_branch` | Workspace manager |
| 1.8 | `agents/coder.py` — ReAct loop: prompt → tool-calls → observe → repeat until `finish_task` or budget | Standalone coder callable (no graph yet) |
| 1.9 | `verify/runner.py` — detect project type (`package.json`/`pyproject.toml`/…), run checks, build `VerifyResult` | Deterministic verify function |
| 1.10 | Budgets — `max_steps_per_task`, token/wall-clock caps; no-progress detection | Loop guards |

## Testing strategy
- **Tool unit tests:** each tool's happy path + error path; error surfaces as `ToolResult(ok=False)` fed back to the loop.
- **Sandbox security tests:** network egress blocked; path traversal (`../`) rejected; timeout kills a hanging command; denied command refused.
- **Git workflow test:** create workspace → coder edits → commit-per-task → `git diff base..HEAD` reflects changes.
- **Verify test:** a project with a passing test suite → `VerifyResult.passed=True`; introduce a failing test → `passed=False` with captured tails.
- **End-to-end task:** spec = "create a Python function `add` with a passing pytest." Coder grounds, writes, runs pytest, `verify` passes. This is the phase's acceptance demo.
- **Budget test:** a task that can't converge trips `max_steps_per_task` and stops cleanly (marks task failed) rather than looping forever.

## Definition of Done
- The end-to-end task above passes: coder produces a **running, test-passing** tiny project in a git-backed workspace, entirely inside the sandbox.
- All tool calls pass through the single authorization pipeline (verified by test).
- Sandbox security tests pass (no network, path-jail, timeout, allowlist).
- `verify` returns a correct structured result for both passing and failing suites.
- Budgets prevent infinite loops.
- Lint, type-check, and Phase-1 tests green; traces visible in Langfuse.

## Risks & mitigations
- **7B model tool-calling reliability** → keep the tool set small and descriptions crisp; rely on `structured_call` + `capabilities` fallback from Phase 0; cap steps.
- **Docker sandbox friction on Windows** → document Docker Desktop/WSL2 setup; provide a restricted-subprocess fallback executor behind a config flag (still cwd-jailed + timeout + allowlist) for machines without Docker, with the security tradeoff documented.
- **Coder thrashing / non-convergence** → `max_steps_per_task`, no-progress detection (state/diff hash), and marking the task failed rather than looping.
- **Edit correctness** → prefer patch/search-replace `edit_file` over full-file rewrite to reduce tokens and clobbering.

## What Phase 2 builds on this
Phase 2 wraps this proven coder + verify loop in the LangGraph state machine (`plan → human_gate → coder → verify → review → finalize`), adds the checkpointer and HITL interrupts, and introduces the `plan` node. It does **not** revisit tools/sandbox/workspace — those are settled here.
