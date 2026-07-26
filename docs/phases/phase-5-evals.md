# Phase 5 — Eval harness

[← Back to README](../../README.md) · [Docs index](../README.md) · [Build plan](../build-plans/PHASE-5.md)

> Turns the Phases 3–4 one-off validation scripts into a small, standing, **scored regression suite** — every run reduces to hard numbers diffed against a saved baseline, so any change can be judged *better or worse*, not just *different*. It adds no runtime node; it's a measurement layer that drives the existing pipeline.

## The eval loop

```
  ┌──────────── fixed task suite — 5 frozen fixtures (app/evals/tasks.py) ────────────┐
  │  happy_path · rag_required · defect_injection · cross_run_memory · retrieval@k     │
  └───────────────────────────────────────┬───────────────────────────────────────────┘
                                          │  each task →
                                          ▼
                          ┌──────────────────────────────┐
                          │   run the REAL pipeline live   │  plan→coder→verify→review→finalize
                          │   & capture the outcome        │  (retrieval task = precision@k only)
                          └───────────────┬───────────────┘
                                          ▼   per-task RunReport
                          ┌──────────────────────────────┐
                          │      aggregate metrics         │
                          │  • deterministic → GATE         │  retrieval precision@k
                          │  • stochastic   → trend         │  success / defect-detect / false-flag / cycles
                          └───────────────┬───────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │   diff vs backend/evals/        │  exit non-zero ONLY on a deterministic
                          │   baseline.json                 │  drop; stochastic metrics reported, never gated
                          └──────────────────────────────┘
```

## What was built

- **Task suite** (`app/evals/tasks.py`) — five frozen fixtures reusing what Phases 3–4
  validated live: a happy-path build, a RAG-required task (hidden helper), a defect-injection
  task (the reviewer *should* catch a planted duplication — its coder is **scripted** so the
  defect reliably exists), a cross-run memory task (two runs), and a retrieval precision@k
  measurement. Each carries an automatic pass/fail check; none take arbitrary input, so their
  numbers are comparable across runs.
- **Runner** (`app/evals/runner.py`) — streams one real graph run to a terminal state
  (auto-aborting escalation interrupts, since an eval has no human), reducing it to a
  task-agnostic capture: review verdicts per cycle, retrieved symbols, verify pass/retries,
  step count, wall-clock.
- **Metrics, split by reproducibility** (`app/evals/metrics.py`) — a **deterministic** set
  (retrieval precision@k — same index + embeddings → same number, so it's *gate-worthy*) and a
  **stochastic** set (success rate, defect-detection rate, false-flag rate, cycles-to-converge,
  steps, latency). The 5–6 task counts are honest *indicative rates*, **not** claimed as
  statistical precision/recall — the suite is deliberately too small for that.
- **Regression gate** (`app/evals/regression.py` + `scripts/run_evals.py`) — diffs a run
  against `backend/evals/baseline.json` and exits non-zero **only** if a *deterministic* metric
  regressed. A local 7B jitters run to run, so hard-gating a stochastic metric would fire on
  noise; those are printed as a before/after trend instead.

## How it was verified

**Hermetic (`test_evals_metrics.py` / `_regression.py` / `_runner.py`):** the whole scoring
path — aggregation, the deterministic-gate-vs-stochastic-trend split, the baseline round-trip,
and `run_graph` capturing a real graph run via `FakeProvider` — runs without a live model, from
hand-built reports.

### The recorded live baseline — and what the numbers honestly say

Running the suite once against the real `qwen2.5-coder:7b` (`scripts/run_evals.py
--update-baseline`, ~54 min on a 16 GB box) produced `backend/evals/baseline.json`:

| Metric | Value | Reading |
|---|---|---|
| **`retrieval_precision_at_k`** (deterministic, gated) | **0.75** | RAG surfaces the right symbol in 3 of 4 fixed queries |
| `task_success_rate` | 0.33 | 1 of 3 multi-step tasks fully converged |
| `defect_detection_rate` | 0.00 | the reviewer missed the planted defect ([Phase 4](phase-4-review.md)'s finding, now quantified) |
| `false_flag_rate` | 0.00 | the reviewer did **not** wrongly block correct code |
| `memory_influence_rate` | 1.00 | a past run's memory correctly shaped the next run's planning |

**The key thing the baseline shows** is a clean separation between *what the architecture does*
and *what the local model can't do*. On two of the "failed" tasks the **feature under test
actually worked** — the coder reused the hidden helper (`reused_helper=True`), and cross-run
memory carried run 1's record into run 2 (`memory_influenced=True`) — but the run still ended
`failed` because the 7B couldn't fully converge the multi-step coding. So `task_success_rate
0.33` and `defect_detection 0.00` are **model-capability numbers, not design defects**,
consistent with every Phase 3–4 finding.

Recorded exactly as it came out, not massaged. Because model choice is a **config-only swap**,
pointing the coder/reviewer at a stronger model should move the stochastic metrics up — and the
baseline will show it in hard numbers. The deterministic `retrieval_precision_at_k = 0.75` is
the reproducible anchor the regression gate protects.

**Validation harness:** `scripts/run_evals.py` (`--category <name>` to run one category,
`--update-baseline` to record).

## Key decisions

Phase 5 reuses Phase 3's retrieval precision@k harness ([ADR-0008](../adr/0008-hybrid-code-rag.md))
as its deterministic metric and measures the [Phase 4](phase-4-review.md) review loop; it adds
no new architectural decision.

---

[← Phase 4 — Review](phase-4-review.md) · Back to the [Docs index](../README.md)
