# Phase 2 — LangGraph orchestration + HITL

> **Goal:** Wrap the proven Phase-1 coder + verify loop in the LangGraph state machine defined in `ARCHITECTURE.md §4` — introducing the `plan` node, the full `AgentState`, durable checkpointing, and human-in-the-loop interrupts — so the system runs an end-to-end `plan → approve → code → verify → review(stub) → finalize` flow with resume-after-crash.
>
> **Why this phase comes third (after the core loop, before RAG):** Phase 1 proved the risky mechanic (tools + sandbox + verify) in isolation. Phase 2 now debugs *orchestration only* — routing, state, checkpointing, interrupts — rather than orchestration and the core loop simultaneously. Grounding (RAG) is deferred to Phase 3 because the graph must exist before retrieval has anywhere to plug in; `search_code` (ripgrep) from Phase 1 is sufficient grounding to make the graph runnable.

## Objectives
1. Implement the complete `AgentState` schema (`ARCHITECTURE.md §5`) with correct reducers.
2. Build the 6-node graph and all conditional edges exactly per `ARCHITECTURE.md §4` topology.
3. Implement the `plan` node (spec + architecture + task list via `structured_call`).
4. Port the Phase-1 coder loop into the `coder` node; add per-task advancement and fix-mode.
5. Wire `verify` (Phase 1 runner) and a **minimal `review` node** (stub verdict logic; full adversarial reviewer is Phase 4).
6. Add the `finalize` node.
7. Integrate the LangGraph checkpointer (SQLite dev default, Postgres path) for durability/resume.
8. Implement HITL via `interrupt()`: the multiplexed `human_gate` node (`plan_approval`, `escalation`, `final_accept`) plus in-tool `command_approval`.
9. Enforce budgets, retry caps, and no-progress/loop detection at the graph level.

## Scope
**In:** `graph/state.py` (schema + reducers), `graph/build_graph.py` (nodes, edges, routing, recursion limit), `agents/planner.py`, `graph/nodes/*` wrappers for coder/verify/review-stub/finalize, `human_gate` node + resume handling, checkpointer integration, autonomy-level gate logic, budget/retry/loop enforcement, run/thread identity, event streaming hooks for the graph (node transitions).
**Out (later phases):** `retrieve`/RAG and symbol index (Phase 3 — `plan`/`coder` use `search_code` ripgrep only for now), the full adversarial reviewer with severity reasoning (Phase 4 — Phase 2 ships a thin reviewer that always returns `approved` or a simple heuristic so the edge is exercised), long-term/episodic memory writes (Phase 3+; `finalize` memory hook is a stub per ADR-0002/§16), custom UI (Phase 6 — use LangGraph Studio + raw event stream), cloud checkpointer/queue (Phase 7).

## Prerequisites
- Phase 1 complete: tool registry + authorization pipeline, sandbox, git-backed workspace lifecycle, standalone coder ReAct loop, deterministic `verify` runner, budgets/step caps.
- Phase 0 complete: provider abstraction, `structured_call`, logging/tracing, docker-compose (Postgres available for the Postgres checkpointer path).

## Work breakdown & deliverables
| # | Task | Deliverable |
|---|---|---|
| 2.1 | `graph/state.py` — full `AgentState` + value objects; reducers: `merge_by_path` (changed_files), `add_messages` (coder_scratch), `add` (errors/node_history/clarification_answers), `merge_counts` (retries) | Typed state; unit-tested reducers |
| 2.2 | `agents/planner.py` — `plan` node: grounds via `search_code`/`read_file`, emits validated `Plan` (spec fields embedded), sets `needs_clarification`, records `assumptions` | `plan` node; clarification `interrupt` path |
| 2.3 | `graph/nodes/coder.py` — wrap Phase-1 loop: read `current_task_id`, run task, advance task pointer, write `changed_files`/`diff_summary`/task status; fix-mode consumes `verify_result`/`review` | `coder` node with per-task iteration |
| 2.4 | `graph/nodes/verify.py` — wrap Phase-1 runner; write `VerifyResult`; increment `retries.verify` | `verify` node |
| 2.5 | `graph/nodes/review_stub.py` — minimal `Review` (verdict via simple rule; structured shape correct) | `review` node placeholder (Phase-4-ready) |
| 2.6 | `graph/nodes/finalize.py` — set terminal `status`, produce `diff_summary`, git branch/diff artifact, **stub** episodic memory hook | `finalize` node |
| 2.7 | `graph/nodes/human_gate.py` — multiplexed `interrupt(HITLRequest)`; apply `hitl_response` (incl. `plan_approval` edits patching `plan`) | `human_gate` node |
| 2.8 | `command_approval` — in-tool interrupt before `run_command` when `autonomy=semi` (wire into Phase-1 authorization pipeline) | Gated tool execution |
| 2.9 | `graph/build_graph.py` — assemble nodes + conditional edges per §4 topology; set recursion limit | Compiled graph |
| 2.10 | Routing functions — `route_after_plan`, `route_after_coder` (more tasks? / budget / loop), `route_after_verify` (pass/fail/exhausted), `route_after_review` (approved/changes/exhausted), `route_after_gate` | Deterministic edge logic |
| 2.11 | Checkpointer integration — SQLite default, Postgres via config; thread_id = run identity | Durable, resumable runs |
| 2.12 | Budget/retry/loop enforcement at graph level — token/step/wall-clock caps, `retries` caps, no-progress detection (state/diff hash over `node_history`) → route to `human_gate[escalation]` | Graph-level guards |
| 2.13 | Autonomy-level gate matrix — `manual`/`semi`/`auto` decide which gates are live (ADR-0009) | Config-driven gating |
| 2.14 | Graph event hooks — emit node-transition/tool/verify events for streaming (consumed by API/UI later) | Streamable event source |

## Testing strategy
- **Reducer unit tests:** `changed_files` merges by path (latest status wins); counters sum; lists append; overwrite fields overwrite. Assert no field can hold file contents (invariant test).
- **Happy-path e2e:** small spec → `plan` produces tasks → `plan_approval` approved → `coder` completes tasks → `verify` passes → `review`(stub) approves → `finalize` succeeds. Assert final git diff and `status=succeeded`.
- **Interrupt/resume test:** run to `human_gate[plan_approval]`, assert graph pauses and a checkpoint exists; resume with `Command(resume=approve)`; assert continuation.
- **Checkpoint-recovery test:** kill the process mid-run (paused at an interrupt); reconstruct the graph and resume from the persisted checkpoint at the identical interrupt.
- **Plan-revise loop:** `plan_approval` → revise with edits → re-enters `plan` → `plan.version` increments; edited fields applied.
- **Verify-fail fix loop:** inject a failing test → `route_after_verify` sends `coder` in fix mode → passes within retry cap; exceeding cap routes to `escalation`.
- **Autonomy matrix tests:** `manual` hits all gates incl. `final_accept`; `semi` gates plan + `command_approval` + escalation; `auto` only escalation.
- **Budget/loop tests:** step/token/wall-clock cap trips → `escalation`; no-progress detection catches a stuck `coder`/`verify` cycle.
- **Routing unit tests:** each `route_*` function against crafted states covering every branch.

## Definition of Done
- The happy-path e2e passes end to end through all 6 nodes with the `review` stub.
- Interrupt → checkpoint → resume works for `plan_approval`, `escalation`, and `command_approval`; a killed process resumes at the identical interrupt.
- All routing branches and the autonomy matrix are covered by tests.
- Budgets, retry caps, and no-progress detection prevent infinite loops (tested).
- Runs are durable under both SQLite (default) and Postgres checkpointers.
- State invariant holds: no file contents / full command output in state (tested).
- The flow is observable in LangGraph Studio and traced in Langfuse; lint/type-check/tests green.

## Risks & mitigations
- **State bloat via `coder_scratch`** → aggressive pruning/summarization of tool-observation history each super-step; enforce the "tails only" invariant; assert checkpoint size stays bounded in tests.
- **7B planner producing invalid/oversized plans** → `structured_call` repair-retry (Phase 0); constrain `Plan` schema; cap task count; empty task list = failure → escalation.
- **Reviewer/coder or verify/coder ping-pong** → per-node retry caps (`retries`) + review→fix cycle cap (default 2) + no-progress detection; all converge to `escalation`.
- **LangGraph recursion limit hit before escalation fires** → set recursion limit above the sum of bounded retries so escalation (not a raw recursion error) is the terminal path; test this ordering.
- **Interrupt/resume state divergence** → treat `plan`/`verify` nodes as idempotent (re-run overwrites); never mutate on-disk workspace inside `human_gate`.
- **SQLite ↔ Postgres checkpointer parity** → run the durability test suite against both in CI-selectable modes.

## What Phase 3 builds on this
Phase 3 plugs **grounding** into the now-working graph: it implements the `retrieve` tool and the tree-sitter symbol index behind the `search_code` interface, feeds `retrieved_context` into the `plan` and `coder` nodes, and turns the `finalize` memory hook into real long-term/semantic memory writes. It does **not** change the graph topology, state schema shape, or node contracts settled here.
