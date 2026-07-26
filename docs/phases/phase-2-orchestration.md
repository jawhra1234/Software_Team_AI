# Phase 2 — LangGraph orchestration + HITL

[← Back to README](../../README.md) · [Docs index](../README.md) · [Build plan](../build-plans/PHASE-2.md)

> The proven Phase-1 loop, wrapped in a real LangGraph state machine: six nodes, conditional routing, human-in-the-loop gates, and crash-safe checkpoints.

## What was built

- **State schema** (`app/graph/state.py`) — the full `AgentState`: `Plan`, `Task`,
  `VerifyResult`, `Review`, `HITLRequest` / `HITLResponse`, `Budget`, `FileRef`, with custom
  reducers (`merge_by_path`, `merge_counts`) so file changes and retry counts accumulate
  correctly across steps. State holds only references and truncated output — **never raw file
  contents** (enforced by tests, per [ADR-0002](../adr/0002-three-sources-of-truth.md)).
- **The compiled graph** (`app/graph/build_graph.py`) — a real LangGraph `StateGraph` with
  six nodes (`plan → human_gate → coder → verify → review → finalize`) wired with conditional
  routing (`app/graph/routing.py`), and a recursion limit set above the run's step budget so
  **escalation, not a raw `GraphRecursionError`, is always the terminal path**.
- **Planner** (`app/agents/planner.py`, `app/graph/nodes/plan.py`) — bounded read-only
  grounding (`list_dir` / `read_file` / `search_code`) then a structured `Plan` emission;
  pauses via `interrupt()` for genuinely blocking clarification questions. Tolerant `StrList`
  coercion keeps a small model's list-of-objects quirk from failing validation.
- **Coder node** (`app/graph/nodes/coder.py`) — wraps the Phase-1 coder loop into the graph:
  selects the next task, or (in "fix mode") synthesizes an ad-hoc task from failing
  verify/review feedback; commits at task boundaries; derives `changed_files` from git.
- **Verify & review nodes** — `verify` (`app/graph/nodes/verify.py`) is the Phase-1
  deterministic runner, now with retry-then-escalate logic. `review_stub` was a rule-based
  placeholder (approved whenever files changed) that exercised both routing branches — later
  **replaced in [Phase 4](phase-4-review.md)** by a real reviewer, with the node contract
  (reads diff+plan+verify, writes `review`) preserved exactly so no rewiring was needed.
- **Human-in-the-loop** (`app/graph/nodes/human_gate.py`) — one multiplexed node handling
  `plan_approval`, `escalation`, and `final_accept`, plus a direct in-tool interrupt for
  `command_approval` before `run_command` — gated by three autonomy levels (`manual` / `semi`
  / `auto`), per [ADR-0009](../adr/0009-hitl-autonomy-levels.md).
- **Durable checkpointing** (`app/graph/checkpointer.py`) — SQLite (clone-and-run default) or
  Postgres ([ADR-0010](../adr/0010-postgres-pgvector-checkpointer.md)), by config alone. A
  killed process resumes at the exact interrupt it paused at.
- **Budgets & instrumentation** (`app/graph/instrument.py`) — a run-wide budget circuit
  breaker (steps / wall-clock / tokens) wraps every node, distinct from the Phase-1 per-task
  budget; event hooks (`app/graph/events.py`) for streaming.
- **LangGraph Studio** (`langgraph.json`, `app/graph/studio.py`) — the compiled graph is
  loadable via `langgraph dev` for visual inspection.

> See the full [runtime flow diagram](../runtime-flow.md) for how these nodes connect at run time.

## How it was verified

**Tests:** happy-path run through all six nodes with zero interrupts (auto); plan-approval
interrupt/resume and plan-revise loop (semi); the full autonomy matrix including the
`final_accept` gate (manual); a verify-fail → fix → pass loop; run-wide budget exhaustion
escalating cleanly instead of crashing; and checkpoint recovery across a simulated process
restart on **both** SQLite and live Postgres.

**Live model:** `scripts/smoke_graph.py` drives the whole compiled graph with the real
`qwen2.5-coder:7b` — plan → coder → verify → review → finalize — producing correct,
test-passing code (`calc.py` + a passing `test_calc.py`), reaching `status: succeeded`,
including the coder recovering from a failed command mid-task.

## Key decisions

- [ADR-0002 — Three separated sources of truth](../adr/0002-three-sources-of-truth.md)
- [ADR-0009 — HITL via `interrupt` with three autonomy levels](../adr/0009-hitl-autonomy-levels.md)
- [ADR-0010 — Postgres + pgvector + LangGraph checkpointer](../adr/0010-postgres-pgvector-checkpointer.md)

---

[← Phase 1 — Core coder loop](phase-1-coder-loop.md) · Next: [Phase 3 — RAG + memory →](phase-3-rag-and-memory.md)
