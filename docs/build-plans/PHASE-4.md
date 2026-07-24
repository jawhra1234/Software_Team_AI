# Phase 4 — Autonomous Review & Self-Correction

> **Goal:** Replace the Phase-2 rule-based `review_stub` with a real **fresh-context LLM reviewer** (`ADR-0006`) that critiques the change through isolated eyes and drives a **bounded, targeted self-correction loop** — `review → coder(fix) → verify → review` — until it approves or escalates. No graph-topology or node-contract changes.
>
> **Why this phase comes fifth (after grounding):** Phase 3 gave the reviewer a real `git diff` to critique and read-only grounding tools to understand it. A reviewer without a real diff or grounding can only nitpick style. This is the last piece of the closed quality loop before Phase 5 can *measure* it — you cannot score review quality until real review exists.

## Objectives
1. A real reviewer agent that sees **only** the approved plan + implementation diff + verify/test output, grounds via **read-only** tools, never edits code, and never sees coder scratch (`ADR-0006` isolation).
2. Structured findings with severities (`blocker`/`major`/`minor`/`nit`); only **blocker/major** trigger another coding iteration (no nitpick-blocking).
3. Targeted fix hand-off: the coder receives the **specific findings** to fix, not a fresh planning request.
4. After a fix, the pipeline automatically re-runs **verify**, then **review** again.
5. Bound the review/fix loop with a configurable maximum iteration count, then **escalate** if it cannot converge.
6. Full tracing of every review/fix cycle.

## Scope
**In:** `agents/reviewer.py` (bounded read-only grounding loop → structured `Review` via `structured_call`), `graph/nodes/review.py` (real node replacing `review_stub`, **same contract** — reads diff+plan+verify, writes `review`), review-context assembly (`git diff base..HEAD` truncated + approved plan + `verify_result`, **coder_scratch excluded by construction**), targeted fix hand-off into the coder's fix-mode task, loop bounding + verdict routing, `build_graph` wiring (reviewer provider + read-only registry + retriever), reviewer config knobs, per-cycle tracing, and a `scripts/review_e2e.py` live validation harness.

**Out (explicitly deferred):** automatic long-term-memory writing of review lessons (deferred), hosted providers (Phase 7), eval scoring / dashboards / regression metrics (Phase 5), web search, multi-reviewer panels or voting, multi-coder orchestration, the reviewer auto-applying its own fixes (the coder fixes, not the reviewer).

## Prerequisites
- Phase 2 complete: the graph, the `Review`/`ReviewIssue` schema, and `review_stub` — which already owns the **final-accept gate**, the **review-cycle cap** (`max_review_cycles`, default 2), and **escalation to `human_gate`**.
- Phase 3 complete: read-only `retrieve` / `search_code` / `read_file` / `list_dir`, the read-only authorization policy, and `RetrievalCapture`.
- Existing loop topology `review → coder(fix mode) → verify → review` (Phase 2) — Phase 4 fills it with a real brain, it does not re-plumb it.

## Work breakdown & deliverables
| # | Task | Deliverable |
|---|---|---|
| 4.1 | `agents/reviewer.py` — planner-shaped: bounded read-only grounding (`retrieve`/`search_code`/`read_file`/`list_dir`) → `structured_call` emits `Review`. System prompt carries the severity rubric + isolation rules. | `review_change(...) -> Review` |
| 4.2 | Review-context assembly — build the reviewer input from `git diff base..HEAD` (tail-truncated), the approved plan, and `verify_result`; **coder_scratch never included**. | Isolated review input |
| 4.3 | `graph/nodes/review.py` — real node, **same contract** as the stub; keeps the final-accept gate, cycle cap, and escalation; wires `RetrievalCapture`. Retires `review_stub` from the graph. | Graph node |
| 4.4 | Targeted fix hand-off — pass the reviewer's concrete issues (file/line/severity/suggestion) into the coder's fix-mode task; **no re-plan**. | Precise fix task |
| 4.5 | Loop bounding + routing — `changes_requested` → coder; `approved` → final_accept/finalize; `rejected` or cap-exceeded → escalate. Cap = `max_review_cycles` (configurable). | Convergent, capped loop |
| 4.6 | `build_graph` wiring — inject the reviewer provider + read-only registry + retriever; point the graph at the real node. | Wired graph |
| 4.7 | `ReviewerSettings` — reviewer `grounding_steps`, max issues surfaced (reuse `MODELS__REVIEWER`, `GRAPH__MAX_REVIEW_CYCLES`). | Config knobs |
| 4.8 | Tracing — record each review cycle (cycle index in state/logs/Langfuse) so every review/fix round is inspectable. | Traced cycles |
| 4.9 | Hermetic tests — scripted verdicts → routing branches; severity gating; isolation invariant; cycle-cap → escalation; malformed review → escalate. | Fast test suite |
| 4.10 | Integration tests — live reviewer emits a schema-valid `Review` on a real diff; grounds via `retrieve`. | Live-model review test |
| 4.11 | `scripts/review_e2e.py` — seed a diff with a real defect → reviewer flags blocker/major → coder targeted fix → verify passes → reviewer approves → finalize `succeeded`. | End-to-end proof |

## Testing strategy
- **Isolation (the signature test):** run a review with a non-empty `coder_scratch` in state; assert that string never appears in any message sent to the reviewer. The isolation *is* the point (`ADR-0006`).
- **Severity gating:** a review carrying only `minor`/`nit` → `approved` (no loop); a review with one `major` → `changes_requested`.
- **Loop + cap:** a reviewer that keeps requesting changes → after `max_review_cycles` the node routes to `human_gate` (escalation), never loops forever.
- **Re-verify after fix:** a review-driven fix routes through `verify` *before* the next `review`, so a fix that breaks the tests is caught, not approved.
- **Robustness / fail-safe:** malformed reviewer output (repair-retry exhausted) → escalation, never a false approval.
- **Targeted fix:** the coder's fix task references the reviewer's specific issues (file/severity), and does not trigger a re-plan (no new `Plan` version).
- **Integration (live model):** the reviewer returns a schema-valid `Review`; a planted bug yields a `blocker`/`major`.
- **Live e2e:** the full catch → targeted fix → verify pass → approve cycle on a real defect, ending `succeeded`.

## Definition of Done
- The real reviewer replaces `review_stub` with the **same node contract** (reads diff+plan+verify, writes `review`).
- The reviewer sees **only** plan + diff + verify output (isolation test green), grounds via read-only tools, and never edits.
- Findings carry severities; only `blocker`/`major` cause another iteration; `minor`/`nit` do not block.
- The coder receives **targeted findings** (not a re-plan); after a fix the pipeline auto re-runs `verify` then `review`.
- The loop is bounded by a configurable cap; on non-convergence, a `rejected` verdict, or malformed output, it **escalates to `human_gate`** rather than hanging or false-approving.
- A live e2e shows a real defect caught, fixed, re-verified, and finally approved (`status: succeeded`).
- Every review/fix cycle is traced in Langfuse. Lint / type-check / hermetic + integration tests green.

## Risks & mitigations
- **Over-blocking (nitpicks treated as blockers)** → an explicit severity rubric in the reviewer prompt, gating only on `blocker`/`major`, and a severity-gating test.
- **Reviewer/coder ping-pong** → a hard cycle cap + escalation; targeted (not re-plan) fixes converge in fewer rounds.
- **Weak 7B reviewer / malformed structured output** → `structured_call` repair-retry; on a still-invalid review, **escalate** (fail safe) — never silently approve.
- **Context-isolation leak (reviewer sees coder reasoning)** → excluded by construction (the node builds the prompt from diff+plan+verify only) + the signature isolation test.
- **Oversized diff blows the context budget** → tail-truncate the diff (same cap approach as `finalize`); the reviewer retrieves specifics on demand rather than reading the whole tree.
- **A fix re-breaks verify** → the loop always re-enters `verify` before `review`; bounded jointly by `max_verify_retries` + `max_review_cycles` + the run-wide budget → escalation, never a hang.

## What the next phase builds on this
Phase 5 (eval harness) measures this loop — review precision/recall on planted defects, fix-loop success rate, and cycles-to-converge — as regression metrics, and reuses `scripts/review_e2e.py` as a fixture. It needs no topology or contract changes: Phase 4 closes the quality loop, Phase 5 quantifies it.
