# Phase 5 — Eval harness

> **Goal:** Turn the ad hoc live-validation scripts built in Phases 3–4
> (`rag_validate.py`, `memory_e2e.py`, `review_e2e.py`) into a small, standing,
> **scored regression suite** — so a change (a prompt edit, a model swap, a
> config tweak) can be judged "better" or "worse" against saved numbers,
> instead of re-run-and-eyeball-the-log every time.
>
> **Why this phase comes sixth (after the full loop exists):** you cannot
> measure quality until `plan → coder → verify → review → finalize` is real
> end to end. Phase 4 closed that loop; Phase 5 quantifies it. Evals then gate
> every later change — this is one of the two phases (with Phase 1's verify
> loop) that most distinguishes a production agent system from a demo, and the
> one most portfolio projects skip.

## Objectives
1. A small, fixed **task suite** spanning the failure modes already found
   live: a happy-path build, a RAG-required task (hidden helper), a
   defect-injection task (the reviewer should catch it), and a cross-run memory
   task — reusing existing fixtures rather than inventing a large new bank.
2. A harness that drives the **real compiled graph** (live models) per task
   and reduces the run to a structured report: final status, verify pass/fail
   + retry count, review verdict/cycle history, retrieval hits, tokens/wall-clock.
3. **Aggregate metrics** across the suite, split by how deterministic they are:
   - *Deterministic* (gate-worthy): **retrieval precision@k** (reusing Phase 3's
     harness — fixed index + embeddings make it reproducible) and anything the
     hermetic harness scores.
   - *Stochastic* (report-as-trend, not a hard gate): task **success rate**,
     **defect-detection rate** (did the reviewer flag the one planted defect) and
     **false-flag rate** (did it request changes on clean tasks), fix-loop
     **cycles-to-converge**, token usage + wall-clock latency. With ~5–6 tasks
     these are honest *indicative counts*, **not** statistical precision/recall —
     the suite is deliberately too small to claim that, and the doc says so.
4. **Regression tracking** — persist each run's metrics to a JSON baseline and
   diff against it. The **exit-code gate fires only on the deterministic
   metrics**; stochastic live-model metrics are printed as a before/after trend
   (and, at most, flagged advisorily on a large drop) because a 7B jitters
   run-to-run and a hard gate on it would fire on noise, not regressions.

## Scope
**In:** `app/evals/tasks.py` (a ~5–6 task suite, reusing existing live-validation
fixtures), `app/evals/runner.py` (`EvalRunner` — drives `build_graph()` live,
captures the node timeline into a `RunReport`), `app/evals/metrics.py`
(aggregation into the deterministic + stochastic metric sets above; retrieval
precision@k via `app/rag/evaluation.py`; token usage + latency from
`Budget`/`Usage` — Ollama does report `prompt_eval_count`/`eval_count`),
`app/evals/regression.py` (JSON baseline diff/comparator, exit-code gate on the
deterministic set only), `scripts/run_evals.py` (CLI entry point), hermetic
tests for the scoring/aggregation/regression logic using `FakeProvider` (no live
model needed to test the harness itself), and one live baseline run. The
baseline artifact lives with the eval code (e.g. `backend/evals/baseline.json`),
not under `docs/`.

**Out (explicitly deferred):**
- **Langfuse tracing/tagging.** The tracer (`app/core/tracing.py`) exists but is
  **not currently invoked by the graph** — a run emits no traces today. Wiring
  the tracer into every node (and only then tagging eval runs) is a separable
  observability task, not part of the measurement harness; deferred so Phase 5
  stays focused. The metrics harness reads everything it needs from the graph's
  returned state + node timeline directly, with no dependency on Langfuse.
- CI/CD pipeline wiring (Phase 7 — cloud/scale infra).
- A UI/dashboard (Phase 6 — Phase 5 only produces the data/numbers).
- Any new agent capability, tool, or graph/topology change.
- Hosted-provider work (the config seam already exists; using it is optional).
- A full SWE-bench-scale suite — deliberately "lite": a handful of
  project-specific tasks, not hundreds.

## Prerequisites
- Phase 3 complete: `app/rag/evaluation.py` (precision@k harness) — reused
  directly as the retrieval-quality metric.
- Phase 4 complete: the real reviewer + bounded fix loop — the
  defect-detection/false-flag and cycles-to-converge metrics measure exactly
  this loop.
- The three existing live-validation scripts (`rag_validate.py`,
  `memory_e2e.py`, `review_e2e.py`) as the source fixtures for the task suite.

## Work breakdown & deliverables
| # | Task | Deliverable |
|---|---|---|
| 5.1 | `app/evals/tasks.py` — `EvalTask` records (id, category, fixture setup fn, expected-outcome predicate), built from existing fixtures: happy-path (`smoke_graph.py`'s calc.py), RAG-required (`rag_validate.py`'s hidden `apply_levy` helper), defect-injection (`review_e2e.py`'s duplication defect), cross-run memory (`memory_e2e.py`) | Small fixed task suite |
| 5.2 | `app/evals/runner.py` — `EvalRunner.run(task)`: builds the graph live, streams it, captures the full node timeline (mirroring the existing scripts' timeline capture) | Structured `RunReport` per task |
| 5.3 | `app/evals/metrics.py` — reduces a list of `RunReport`s into the deterministic set (retrieval precision@k) + the stochastic set (success rate, defect-detection rate, false-flag rate, cycles-to-converge, token usage, latency) | Suite-level metrics, split by determinism |
| 5.4 | `app/evals/regression.py` — loads the saved baseline JSON, diffs against the current run; **hard-gates the deterministic metrics**, reports stochastic metrics as a before/after trend | Per-metric flags + gate decision |
| 5.5 | `scripts/run_evals.py` — CLI: run the full suite (or a `--category` filter) live, print a report table, exit non-zero **only** if a deterministic metric regressed | Runnable harness |
| 5.6 | Hermetic tests — task-suite loading, `RunReport` reduction from a scripted (`FakeProvider`) graph run, metrics aggregation, regression comparator (deterministic regression gates; stochastic drop is advisory-only), report formatting | Fast test suite |
| 5.7 | Live baseline run — run the suite once against the real models; record and report the honest numbers (including any 7B-ceiling findings, e.g. the Phase-4 reviewer-grounding gap surfacing as a low defect-detection rate) | Recorded baseline (`backend/evals/baseline.json`) |

## Testing strategy
- **Hermetic:** the task suite loads and each task's fixture setup runs
  cleanly; `RunReport` reduction from a scripted graph run (via `FakeProvider`,
  same pattern as the Phase 2–4 graph tests) computes the correct pass/fail and
  per-task metrics; the metrics aggregator correctly computes success rate,
  defect-detection/false-flag rates, and cycles-to-converge on a small synthetic
  set of `RunReport`s; the regression comparator **gates on a deterministic
  metric moving worse but only advises on a stochastic one**, and correctly
  flags improved vs unchanged; report formatting is stable and readable.
- **Live (this phase's actual validation, not an afterthought):** run
  `scripts/run_evals.py` against the real suite once, record the numbers as
  the initial baseline, and report them exactly as they come out — including
  if the local 7B's known review-grounding gap (Phase 4's finding) shows up as
  a low defect-detection rate on the defect-injection task. That is the correct,
  honest outcome to record, not a result to chase into looking better.

## Definition of Done
- A runnable suite covering happy-path / RAG-required / defect-injection /
  memory-dependent tasks, built from existing fixtures.
- Aggregate metrics computed and printed, split into the deterministic set
  (retrieval precision@k) and the stochastic set (success rate,
  defect-detection/false-flag rates, cycles-to-converge, token usage, latency).
- A regression mechanism that **hard-gates the deterministic metrics** against a
  saved JSON baseline (exit code) and reports the stochastic metrics as a
  before/after trend — inspectable via a plain diff, no new infra.
- Hermetic tests (suite loading, report reduction, metrics, regression
  comparator) green.
- One live baseline run recorded and reported honestly — numbers as they
  actually came out, including any known model-capability limitations.

## Risks & mitigations
- **Live runs are slow under this box's memory constraints** → keep the suite
  small (~5–6 tasks); support running a single category via `--category`.
- **7B nondeterminism makes a hard gate on live metrics fire on noise** → the
  exit-code gate covers only the *deterministic* metrics (retrieval precision@k
  over a fixed index, hermetic-scored logic); stochastic live-model metrics are
  reported as trends and documented as a capability baseline, never a pass/fail
  bar. This is the core methodological guard, not an afterthought.
- **Tiny suite ≠ statistics** → the ~5–6 task counts are reported as honest
  indicative rates (defect-detection, false-flag), explicitly *not* claimed as
  statistical precision/recall; the doc and the report both say so.
- **Scope creep into CI/dashboards/observability** → explicitly out of scope;
  Phase 5 stops at producing the measurement + comparison and a non-zero exit
  code. It does not run itself (CI, Phase 7), visualize itself (UI, Phase 6), or
  wire Langfuse tracing into the graph (a separate observability task).

## What the next phase builds on this
Phase 6 (Mission-control UI) surfaces these eval reports and metric trends
visually instead of as a printed table. Phase 7 (Cloud + scale) can wire the
exit-code contract `scripts/run_evals.py` establishes into an actual CI gate —
Phase 5 only needs to make that contract exist, not run it automatically.
