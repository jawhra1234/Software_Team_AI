# Phase 4 — Autonomous review & self-correction

[← Back to README](../../README.md) · [Docs index](../README.md) · [Build plan](../build-plans/PHASE-4.md)

> A real, independent reviewer with fresh eyes replaces the Phase-2 stub — catching problems and sending **targeted** fixes back through verify, a genuine bounded self-correction loop. No graph-topology or node-contract change.

## What was built

The reviewer is the third and last real LLM role in the "3 roles + deterministic verify"
design; it plugs into the same seam the stub occupied ([ADR-0006](../adr/0006-fresh-context-reviewer.md)).

- **Reviewer agent** (`app/agents/reviewer.py`) — shaped exactly like the planner: a bounded,
  **read-only** grounding loop (`retrieve` / `search_code` / `read_file` / `list_dir` — never
  write/exec) then a `structured_call` emitting a `Review`. Its prompt states five judging
  priorities in order — **correctness, security, test gaps, architecture (incl. duplicated
  logic), maintainability** — and is explicit that style/formatting/naming are *never* blockers.
- **Isolation is structural, not a convention** (`app/graph/nodes/review.py`) — the reviewer's
  input is built from exactly three sources: the approved `Plan`, the diff
  (`state["diff_summary"]`), and the `VerifyResult`. The code path **never reads
  `state["coder_scratch"]`** — it simply has no access to the coder's reasoning to leak. That
  isolation *is* the point of a fresh-context reviewer.
- **The verdict is not trusted blindly** — a small model can say "approved" while listing a
  blocker, or "changes_requested" over a nit. The node **deterministically overrides** the
  effective verdict from the issues' own severities: any `blocker`/`major` forces
  `changes_requested`; otherwise `approved`. So "only blocker/major trigger another cycle" is a
  property of the code, not model compliance. A `rejected` verdict or a malformed/unparseable
  review both **escalate to a human** rather than looping or silently approving.
- **Targeted fix hand-off** (`build_fix_task` in `app/graph/nodes/coder.py`) — when changes are
  requested, the coder's fix task is built **only from the blocker/major issues** (file, what's
  wrong, the suggestion) — never a re-plan, never diluted by minor/nit noise. A fix always
  re-enters `verify` before the next `review`, so a fix that breaks the tests is caught.
- **Bounded and observable** — the same cycle cap (`GRAPH__MAX_REVIEW_CYCLES`, default 2)
  governs the loop; every cycle logs `review_produced` and, on an override, `review_verdict_overridden`.

## How it was verified

**Hermetic (16 tests, `test_reviewer.py` + `test_graph_nodes_review.py`):** bounded grounding
and structured emission; the severity override in **both** directions (a false "approved" with
a blocker → forced `changes_requested`; a false "changes_requested" over only nits → forced
`approved`); the cycle cap escalates instead of looping; a `rejected` verdict escalates
immediately; a malformed response escalates rather than fabricating a fake approval; the
final-accept gate is preserved; and — the signature test — a planted secret in `coder_scratch`
**never** appears in any message sent to the reviewer, proving isolation structurally.

**Live model (`test_reviewer_integration.py`):** the real `qwen2.5-coder:7b` reviewer reliably
returns a **schema-valid** `Review` through `structured_call`'s repair-retry, with every
severity in the valid enum — the structured-output contract holds under a live model.

### Does the live reviewer catch a real defect? (an honest limitation)

The hermetic tests prove the **loop mechanics** using scripted findings. They can't prove a
live 7B will *notice* a subtle defect on its own — so `scripts/review_e2e.py` tests exactly
that: a scripted planner+coder seed a real **architecture/duplication defect** (the coder
reimplements `pricing_rules.apply_levy` instead of reusing it) behind a deliberately **weak**
test that passes on both buggy and correct code, so only the reviewer — running **live** — can
catch it.

| Run | Reviewer prompt | Result |
|---|---|---|
| 1st | original ("ground before judging") | `approved`, **0 issues**, **no grounding calls** |
| 2nd | strengthened ("grounding is **mandatory**; check for duplication") | identical: `approved`, **0 issues**, **no grounding calls** |

**What this shows, precisely:**

1. **The pipeline is not at fault** — every wiring point fired correctly both runs; the
   read-only tools were all available; the reviewer simply never called them.
2. **A genuine, reproducible model-capability limit, not a prompt-wording problem** —
   strengthening the prompt to mandate grounding (the same class of fix that worked for the
   coder/planner in Phase 3) made **no observed difference**. A 7B driven through a
   forced-schema tool call can be "technically compliant" while doing essentially no analysis.
3. **What remains proven** — the loop's correctness doesn't depend on the model noticing
   everything: severity override, isolation, targeted fix hand-off, bounded cycle + escalation,
   and the malformed-output fail-safe are all proven by the 16 hermetic tests with scripted
   findings. So when a reviewer of *any* capability surfaces a blocker/major, the system routes,
   fixes, re-verifies, and re-reviews it correctly. What's unproven is that *this* local model
   reliably surfaces a subtle issue unprompted — a capability gap, not a design defect.
4. **Why no further prompt-chasing** — re-wording to hunt a lucky pass would drift into tuning
   the benchmark. The one legitimate fix was applied, honestly re-tested, and reported as-is.

**The smallest real fix, not yet applied:** model choice is config-only, so pointing the
reviewer at a stronger model via `MODELS__REVIEWER__MODEL` / `MODELS__REVIEWER__PROVIDER` is the
natural next lever — a real capability upgrade, zero code change. [Phase 5](phase-5-evals.md)
now quantifies exactly this gap (defect-detection rate) so such a swap can be *proven* to help.

## Key decisions

- [ADR-0006 — Fresh-context adversarial reviewer with isolated context](../adr/0006-fresh-context-reviewer.md)

---

[← Phase 3 — RAG + memory](phase-3-rag-and-memory.md) · Next: [Phase 5 — Evals →](phase-5-evals.md)
