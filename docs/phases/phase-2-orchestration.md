# Phase 2 — LangGraph orchestration + HITL

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 1 — Core coder loop](phase-1-coder-loop.md) · [Next: Phase 3 — RAG + memory →](phase-3-rag-and-memory.md)

> *This is an **as-built** writeup. For the original forward-looking specification, see the [Phase 2 build plan](../build-plans/PHASE-2.md).*

The proven Phase-1 loop, wrapped in a real LangGraph state machine: six nodes, conditional
routing, human-in-the-loop gates, and crash-safe checkpoints.

## What this phase adds

The compiled six-node graph (`plan → human_gate → coder → verify → review → finalize`), the
`AgentState` schema, conditional routing, three human-in-the-loop autonomy levels, durable
checkpointing, and a run-wide budget circuit breaker.

## Why it was needed

A single loop can't do plan → build → verify → review with human gates and crash recovery. A
state machine makes control flow explicit, checkpointable, and resumable — and gives every later
phase (RAG, review, evals) a stable topology to plug into.

## Architecture / how it works

- **Six nodes**, wired with conditional edges (`app/graph/routing.py`). Routing is a pure
  function of state: every node that can escalate sets `hitl_request` explicitly, so routing only
  ever checks its presence.
- **Three sources of truth** ([ADR-0002](../adr/0002-three-sources-of-truth.md)): control flow +
  small artifacts live in checkpointed LangGraph state; the actual code lives in the git
  workspace; state holds only references and truncated output, **never raw file contents**
  (enforced by tests).
- **Human-in-the-loop** ([ADR-0009](../adr/0009-hitl-autonomy-levels.md)): one multiplexed
  `human_gate` handles `plan_approval` / `escalation` / `final_accept`, plus an in-tool interrupt
  for `command_approval` — gated by `manual` / `semi` / `auto`.
- **Durability** ([ADR-0010](../adr/0010-postgres-pgvector-checkpointer.md)): SQLite (default) or
  Postgres checkpointer by config; a killed process resumes at the exact interrupt it paused at.
- The recursion limit sits above the sum of bounded per-node retries, so **escalation — not a raw
  `GraphRecursionError` — is always the terminal path** for a run that keeps failing.

> Full annotated node-by-node diagram: **[docs/runtime-flow.md](../runtime-flow.md)**.

## Implementation

- `app/graph/state.py` — `AgentState` + value objects + reducers (`merge_by_path`, `merge_counts`).
- `app/graph/build_graph.py` — assembles/compiles the `StateGraph`; `routing.py` — conditional edges.
- `app/graph/nodes/` — `plan`, `coder`, `verify`, `review_stub` (later replaced in Phase 4),
  `finalize`, `human_gate`.
- `app/agents/planner.py` — bounded read-only grounding + structured `Plan` emission.
- `app/graph/checkpointer.py` — SQLite/Postgres selection; `app/graph/instrument.py` — run-wide
  budget breaker + event hooks; `app/graph/studio.py`, `langgraph.json` — LangGraph Studio entry.

## Configuration

```bash
CHECKPOINTER__BACKEND=sqlite       # or 'postgres' (ADR-0010)
GRAPH__RECURSION_LIMIT=200         # above the run step budget so escalation is the terminal path
GRAPH__MAX_VERIFY_RETRIES=3        # verify-fail → fix cycles before escalating
GRAPH__MAX_RUN_STEPS=60            # run-wide budget circuit breaker (also WALL_CLOCK / TOKENS)
PLANNER__GROUNDING_STEPS=6         # bounded read-only grounding rounds before a Plan
```

## Testing and validation

- **Hermetic (graph tests):** happy-path through all six nodes with zero interrupts (auto);
  plan-approval interrupt/resume and plan-revise loop (semi); the full autonomy matrix incl. the
  `final_accept` gate (manual); a verify-fail → fix → pass loop; run-wide budget exhaustion
  escalating cleanly instead of crashing; and checkpoint recovery across a simulated process
  restart on **both** SQLite and live Postgres.

## Live validation

`scripts/smoke_graph.py` drives the whole compiled graph with the real `qwen2.5-coder:7b` —
plan → coder → verify → review → finalize — producing correct, test-passing code (`calc.py` + a
passing `test_calc.py`), reaching `status: succeeded`, incl. the coder recovering from a failed
command mid-task.

## What worked

The graph is genuinely crash-safe and human-gated: interrupts checkpoint and resume exactly where
they paused (proven on live Postgres), and a runaway run always ends in a clean escalation rather
than a crash.

## Known limitations / honest findings

- The Phase-2 `review` node was a deliberate **rule-based stub** (approve when files changed) —
  enough to exercise both routing branches, replaced by the real reviewer in
  [Phase 4](phase-4-review.md) with the node contract preserved exactly (no rewiring).
- A small model can emit a malformed `Plan`; the `StrList` / `Task.kind` tolerant coercions
  (the latter added in Phase 3) keep common quirks from failing the whole plan.

## Key engineering decisions

- [ADR-0002 — Three separated sources of truth](../adr/0002-three-sources-of-truth.md)
- [ADR-0009 — HITL via `interrupt` with three autonomy levels](../adr/0009-hitl-autonomy-levels.md)
- [ADR-0010 — Postgres + pgvector + LangGraph checkpointer](../adr/0010-postgres-pgvector-checkpointer.md)

## Current status

✅ **Complete and verified.**

---

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 1 — Core coder loop](phase-1-coder-loop.md) · [Next: Phase 3 — RAG + memory →](phase-3-rag-and-memory.md)
