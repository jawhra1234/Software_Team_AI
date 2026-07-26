# Phase 4 — Autonomous review & self-correction

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 3 — RAG + memory](phase-3-rag-and-memory.md) · [Next: Phase 5 — Evals →](phase-5-evals.md)

> *This is an **as-built** writeup. For the original forward-looking specification, see the [Phase 4 build plan](../build-plans/PHASE-4.md).*

A real, independent reviewer with fresh eyes replaces the Phase-2 stub — catching problems and
sending **targeted** fixes back through verify, a genuine bounded self-correction loop. No
graph-topology or node-contract change.

## What this phase adds

A fresh-context LLM reviewer (`app/agents/reviewer.py`) and a real `review` node that drives the
existing `review → coder(fix) → verify → review` loop with severity-gated, targeted fixes.

## Why it was needed

An author reviewing their own work misses errors anchored to their own reasoning. Phase 3 gave the
reviewer a real diff to critique and read-only grounding tools; a genuine independent reviewer is
the last piece of a closed quality loop — and the prerequisite for Phase 5 being able to *measure*
review quality.

## Architecture / how it works

The reviewer is the third real LLM role in the "3 roles + deterministic verify" design; it plugs
into the exact seam the stub occupied ([ADR-0006](../adr/0006-fresh-context-reviewer.md)).

- **Fresh-context, read-only:** shaped like the planner — a bounded grounding loop
  (`retrieve`/`search_code`/`read_file`/`list_dir`, never write/exec) then a `structured_call`
  emitting a `Review`. Priorities, in order: **correctness, security, test gaps, architecture
  (incl. duplicated logic), maintainability**; style/naming are *never* blockers.
- **Isolation is structural, not a convention:** the reviewer's input is built from exactly three
  sources — the approved `Plan`, the diff (`state["diff_summary"]`), and the `VerifyResult`. The
  code path **never reads `state["coder_scratch"]`**, so it has no access to the coder's reasoning
  to leak.
- **The verdict is not trusted blindly:** the node **deterministically overrides** the effective
  verdict from the issues' own severities — any `blocker`/`major` forces `changes_requested`,
  otherwise `approved`. A `rejected` verdict or a malformed/unparseable review both **escalate to
  a human** rather than looping or silently approving.
- **Targeted fix hand-off:** on changes requested, the coder's fix task is built **only from the
  blocker/major issues** (never a re-plan; minor/nit stay advisory). A fix always re-enters
  `verify` before the next `review`.

## Implementation

- `app/agents/reviewer.py` — the reviewer agent (grounding loop + `Review` emission).
- `app/graph/nodes/review.py` — the real node (replaces the deleted `review_stub.py`); severity
  override; final-accept gate; cycle cap; escalation.
- `app/graph/nodes/coder.py` `build_fix_task` — blocker/major-only targeted fix.
- `app/graph/build_graph.py`, `app/core/config.py` — reviewer provider DI + `ReviewerSettings`.

## Configuration

```bash
MODELS__REVIEWER__MODEL=qwen2.5-coder:7b-instruct  # the reviewer's model (swap independently)
MODELS__REVIEWER__PROVIDER=ollama                   # e.g. point at a hosted provider
GRAPH__MAX_REVIEW_CYCLES=2                           # review/fix cycles before escalating
REVIEWER__GROUNDING_STEPS=4                          # bounded read-only grounding rounds
REVIEWER__MAX_ISSUES=10                              # soft cap on surfaced issues
```

## Testing and validation

- **Hermetic (16 tests, `test_reviewer.py` + `test_graph_nodes_review.py`):** bounded grounding
  and structured emission; the severity override in **both** directions (a false "approved" with a
  blocker → forced `changes_requested`; a false "changes_requested" over only nits → forced
  `approved`); cycle-cap escalation; `rejected` → immediate escalation; malformed output →
  escalation (never a fake approval); the final-accept gate preserved; and — the **signature
  test** — a planted secret in `coder_scratch` never appears in any message sent to the reviewer,
  proving isolation structurally.

## Live validation

- **`test_reviewer_integration.py`:** the real `qwen2.5-coder:7b` reviewer reliably returns a
  **schema-valid** `Review` through `structured_call`'s repair-retry — the structured-output
  contract holds under a live model.
- **`scripts/review_e2e.py`:** seeds a real architecture/duplication defect (a scripted coder
  reimplements `apply_levy` instead of reusing it) behind a deliberately weak test that only the
  reviewer can catch — see the honest finding below.

## What worked

The loop **mechanics are proven correct**: severity override, structural isolation, targeted fix
hand-off, bounded cycle + escalation, and the malformed-output fail-safe all hold under the 16
hermetic tests. When a reviewer of *any* capability surfaces a blocker/major, the system routes,
fixes, re-verifies, and re-reviews it correctly.

## Known limitations / honest findings

**The live 7B reviewer does not reliably catch a subtle defect on its own — and this is a
model-capability limit, not a broken pipeline.** In `review_e2e.py`, the live reviewer approved the
planted duplication on its first pass with **zero grounding calls**, both before *and after* a
prompt fix mandating grounding:

| Run | Reviewer prompt | Result |
|---|---|---|
| 1st | original ("ground before judging") | `approved`, 0 issues, **no grounding calls** |
| 2nd | strengthened ("grounding is **mandatory**") | identical: `approved`, 0 issues, **no grounding calls** |

- Every wiring point fired correctly both runs; the read-only tools were available — the reviewer
  simply never called them. A 7B driven through a forced-schema tool call can be "technically
  compliant" while doing essentially no analysis. Prompt-wording is not the lever here.
- No further prompt-chasing was done — that would be tuning the benchmark, not hardening the system.
- **The real fix is config-only:** point the reviewer at a stronger model via
  `MODELS__REVIEWER__MODEL` / `MODELS__REVIEWER__PROVIDER` — zero code change. [Phase 5](phase-5-evals.md)
  now quantifies this gap (`defect_detection_rate`) so such a swap can be *proven* to help.

## Key engineering decisions

- [ADR-0006 — Fresh-context adversarial reviewer with isolated context](../adr/0006-fresh-context-reviewer.md)

## Current status

✅ **Complete and verified** (mechanics proven hermetically; live schema-validity confirmed; the
7B's subtle-defect-detection limit documented, not hidden).

---

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 3 — RAG + memory](phase-3-rag-and-memory.md) · [Next: Phase 5 — Evals →](phase-5-evals.md)
