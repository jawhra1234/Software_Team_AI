# Phase 5 — Eval harness

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 4 — Review](phase-4-review.md) · Next: —

> *This is an **as-built** writeup. For the original forward-looking specification, see the [Phase 5 build plan](../build-plans/PHASE-5.md).*

Turns the Phases 3–4 one-off validation scripts into a small, standing, **scored regression
suite** — every run reduces to hard numbers diffed against a saved baseline, so any change can be
judged *better or worse*, not just *different*. It adds no runtime node; it's a measurement layer
that drives the existing pipeline.

## What this phase adds

A fixed 5-task suite, a runner that scores one real graph run, aggregate metrics split into a
gate-worthy *deterministic* set and a *stochastic* trend set, a JSON regression gate, and a
recorded live baseline.

## Why it was needed

Every prior phase was proven by running a script and reading the log — rigorous once, but not
*comparable*: there was no way to tell if next week's prompt tweak or model swap made things
better or worse. Phase 5 makes quality **measurable**, which is what lets every future change be
judged objectively.

## Architecture / how it works

```
  fixed task suite (5 fixtures)  →  run the REAL pipeline live, capture outcome  →  aggregate metrics
                                     (retrieval task = precision@k only)             ├─ deterministic → GATE
                                                                                     └─ stochastic   → trend
                                                                             →  diff vs backend/evals/baseline.json
                                                                                (exit non-zero ONLY on a deterministic drop)
```

- **Five frozen fixtures**, reusing Phase 3–4 fixtures: `happy_path`, `rag_required` (hidden
  helper), `defect_injection` (scripted coder plants a defect so the **live reviewer** is what's
  tested), `cross_run_memory` (two runs), `retrieval` (precision@k, no graph run).
- **Metrics split by reproducibility:** a **deterministic** metric (`retrieval_precision_at_k` —
  same index + embeddings → same number, so *gate-worthy*) and **stochastic** ones (success rate,
  defect-detection, false-flag, cycles, latency). The 5-task counts are honest *indicative rates*,
  **not** claimed as statistical precision/recall.
- **The gate fires only on a deterministic regression** — a local 7B jitters run to run, so
  hard-gating a stochastic metric would fire on noise. Stochastic metrics are reported as a trend.

## Implementation

- `app/evals/report.py` — `RunReport` (per-task outcome; JSON-serializable).
- `app/evals/runner.py` — streams one real graph run → task-agnostic capture.
- `app/evals/metrics.py` — aggregation into the deterministic + stochastic sets.
- `app/evals/regression.py` — baseline diff + gate.
- `app/evals/tasks.py` — the 5 fixtures.
- `scripts/run_evals.py` — CLI; `backend/evals/baseline.json` — the recorded baseline.

## Configuration

No new settings. Model choice (the main lever this phase measures) is the existing per-role config
(`MODELS__CODER__MODEL`, `MODELS__REVIEWER__MODEL`, …).

## Testing and validation

- **Hermetic (`test_evals_metrics.py` / `_regression.py` / `_runner.py`):** the whole scoring
  path — aggregation, the deterministic-gate-vs-stochastic-trend split, the baseline round-trip,
  and `run_graph` capturing a real graph run via `FakeProvider` — runs without a live model.

## Live validation

The suite was run once against the real `qwen2.5-coder:7b` (`scripts/run_evals.py
--update-baseline`, ~54 min on a 16 GB box), recording `backend/evals/baseline.json`:

| Metric | Value | Reading |
|---|---|---|
| **`retrieval_precision_at_k`** (deterministic, gated) | **0.75** | RAG surfaces the right symbol in 3 of 4 fixed queries |
| `task_success_rate` | 0.33 | 1 of 3 multi-step tasks fully converged |
| `defect_detection_rate` | 0.00 | the reviewer missed the planted defect ([Phase 4](phase-4-review.md)'s finding, quantified) |
| `false_flag_rate` | 0.00 | the reviewer did **not** wrongly block correct code |
| `memory_influence_rate` | 1.00 | a past run's memory correctly shaped the next run's planning |

## What worked

The harness cleanly **separates what the architecture does from what the local model can't do**:
on two of the "failed" tasks the *feature under test worked* — the coder reused the hidden helper
(`reused_helper=True`) and cross-run memory carried run 1 into run 2 (`memory_influenced=True`) —
even though the 7B couldn't fully converge the multi-step coding. It also handled a broken-Ollama
run gracefully (per-task errors recorded, suite continued, no crash).

## Known limitations / honest findings

- `task_success_rate 0.33` and `defect_detection 0.00` are **model-capability numbers, not design
  defects** — pointing the coder/reviewer at a stronger model (config-only) should move them up,
  and the baseline will show it in hard numbers.
- The baseline is recorded exactly as it came out — not massaged. The deterministic
  `retrieval_precision_at_k = 0.75` (one of four paraphrase queries misses) is the reproducible
  anchor the gate protects.
- **Out of scope (by design):** CI wiring (Phase 7), a dashboard (Phase 6), and wiring Langfuse
  tracing into the graph — the tracer exists but isn't invoked by graph nodes yet.

## Key engineering decisions

Phase 5 reuses Phase 3's retrieval precision@k harness ([ADR-0008](../adr/0008-hybrid-code-rag.md))
as its deterministic metric and measures the [Phase 4](phase-4-review.md) review loop; it adds no
new architectural decision record.

## Current status

✅ **Complete and verified** (harness hermetically tested; one honest live baseline recorded).

---

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 4 — Review](phase-4-review.md) · Next: —
