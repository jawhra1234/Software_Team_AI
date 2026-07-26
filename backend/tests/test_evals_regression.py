"""Phase 5 — regression comparison: deterministic gate vs stochastic trend (hermetic)."""

from __future__ import annotations

from pathlib import Path

from app.evals.metrics import SuiteMetrics, aggregate
from app.evals.regression import compare, load_baseline, save_baseline
from app.evals.report import RETRIEVAL, RunReport


def _metrics(**over: float | int | None) -> SuiteMetrics:
    base: dict[str, float | int | None] = {
        "retrieval_precision_at_k": 1.0,
        "task_success_rate": 1.0,
        "defect_detection_rate": 1.0,
        "false_flag_rate": 0.0,
        "memory_influence_rate": 1.0,
        "avg_review_cycles": 1.0,
        "total_steps": 10,
        "total_wall_clock_s": 5.0,
        "task_count": 5,
    }
    base.update(over)
    return SuiteMetrics(**base)  # type: ignore[arg-type]


def test_deterministic_drop_is_a_gated_regression() -> None:
    baseline = _metrics().to_dict()
    current = _metrics(retrieval_precision_at_k=0.75)
    result = compare(current, baseline)
    assert result.has_gated_regression
    assert [d.key for d in result.gated_regressions] == ["retrieval_precision_at_k"]


def test_stochastic_drop_is_reported_but_not_gated() -> None:
    baseline = _metrics().to_dict()
    # success rate and defect detection both fell — real signal, but NOT a build-failing gate
    current = _metrics(task_success_rate=0.5, defect_detection_rate=0.0)
    result = compare(current, baseline)
    assert not result.has_gated_regression
    downs = {d.key for d in result.deltas if d.direction == "down"}
    assert {"task_success_rate", "defect_detection_rate"} <= downs


def test_deterministic_improvement_is_not_a_regression() -> None:
    baseline = _metrics(retrieval_precision_at_k=0.5).to_dict()
    result = compare(_metrics(retrieval_precision_at_k=1.0), baseline)
    assert not result.has_gated_regression


def test_no_baseline_never_gates() -> None:
    result = compare(_metrics(retrieval_precision_at_k=0.0), None)
    assert not result.has_gated_regression  # first run just records; nothing to regress against


def test_baseline_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    metrics = _metrics(retrieval_precision_at_k=0.8)
    save_baseline(path, metrics, stamped_at="2026-01-01T00:00:00Z")
    loaded = load_baseline(path)
    assert loaded is not None
    assert loaded["retrieval_precision_at_k"] == 0.8
    # a re-run identical to the saved baseline gates nothing
    assert not compare(metrics, loaded).has_gated_regression


def test_load_missing_baseline_returns_none(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "nope.json") is None


def test_metrics_flow_from_aggregate_through_compare() -> None:
    # End-to-end of the pure path: reports -> metrics -> baseline -> a later regressed run.
    good = aggregate([RunReport(task_id="retr", category=RETRIEVAL, precision_at_k=1.0)])
    worse = aggregate([RunReport(task_id="retr", category=RETRIEVAL, precision_at_k=0.5)])
    assert compare(worse, good.to_dict()).has_gated_regression
