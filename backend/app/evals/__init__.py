"""Eval harness (Phase 5): a small, scored, repeatable regression suite.

Turns the Phase 3-4 live-validation scripts into fixed, auto-scored tasks so a
change (prompt edit, model swap, config tweak) can be judged better/worse
against a saved baseline instead of re-run-and-eyeball. Metrics split into a
*deterministic* set (retrieval precision@k — gate-worthy) and a *stochastic*
set (live-model success/defect-detection/false-flag rates, cycles, latency —
reported as trends, never hard-gated, because a 7B jitters run to run).
"""
