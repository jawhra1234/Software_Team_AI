"""Phase 5 — metrics aggregation over hand-built RunReports (hermetic, no live model)."""

from __future__ import annotations

from app.evals.metrics import aggregate
from app.evals.report import (
    CROSS_RUN_MEMORY,
    DEFECT_INJECTION,
    HAPPY_PATH,
    RAG_REQUIRED,
    RETRIEVAL,
    RunReport,
)


def _graph(task_id: str, category: str, **kw: object) -> RunReport:
    return RunReport(task_id=task_id, category=category, **kw)  # type: ignore[arg-type]


def test_success_rate_only_over_success_expected_categories() -> None:
    reports = [
        _graph("h", HAPPY_PATH, status="succeeded"),
        _graph("r", RAG_REQUIRED, status="failed"),
        # defect_injection is NOT success-expected — its status must not affect success rate.
        _graph("d", DEFECT_INJECTION, status="failed"),
    ]
    m = aggregate(reports)
    assert m.task_success_rate == 0.5  # 1 of {happy, rag}, defect excluded


def test_defect_detection_and_false_flag_are_distinct() -> None:
    reports = [
        # clean task the reviewer wrongly blocked -> a false flag
        _graph("h", HAPPY_PATH, status="succeeded", review_flagged_blocking=True),
        # defect task the reviewer correctly blocked -> a catch
        _graph("d", DEFECT_INJECTION, status="succeeded", review_flagged_blocking=True),
    ]
    m = aggregate(reports)
    assert m.defect_detection_rate == 1.0  # caught the 1 planted defect
    assert m.false_flag_rate == 1.0  # falsely blocked the 1 clean task


def test_defect_missed_reads_as_zero_detection() -> None:
    m = aggregate([_graph("d", DEFECT_INJECTION, status="succeeded", review_flagged_blocking=False)])
    assert m.defect_detection_rate == 0.0
    assert m.false_flag_rate is None  # no clean tasks -> not 0%, but "no data"


def test_memory_influence_rate() -> None:
    m = aggregate([
        _graph("m1", CROSS_RUN_MEMORY, status="succeeded", memory_influenced=True),
        _graph("m2", CROSS_RUN_MEMORY, status="succeeded", memory_influenced=False),
    ])
    assert m.memory_influence_rate == 0.5


def test_retrieval_precision_is_the_deterministic_metric() -> None:
    m = aggregate([RunReport(task_id="retr", category=RETRIEVAL, precision_at_k=0.75)])
    assert m.retrieval_precision_at_k == 0.75
    # a retrieval-only report contributes no graph metrics
    assert m.task_success_rate is None


def test_avg_review_cycles_only_over_reviewed_runs() -> None:
    m = aggregate([
        _graph("h", HAPPY_PATH, status="succeeded", review_verdicts=["approved"]),
        _graph("d", DEFECT_INJECTION, status="succeeded",
               review_verdicts=["changes_requested", "approved"], review_flagged_blocking=True),
        # a run that never reached review must not drag the average toward zero
        _graph("x", HAPPY_PATH, status="failed"),
    ]).avg_review_cycles
    assert m == 1.5  # (1 + 2) / 2 reviewed runs


def test_harness_error_excluded_from_graph_metrics() -> None:
    m = aggregate([
        _graph("h", HAPPY_PATH, status="succeeded"),
        _graph("boom", HAPPY_PATH, error="workspace blew up"),
    ])
    assert m.task_success_rate == 1.0  # the errored report is excluded, not counted as failure


def test_empty_suite_is_all_none_not_zero() -> None:
    m = aggregate([])
    assert m.task_success_rate is None
    assert m.retrieval_precision_at_k is None
    assert m.task_count == 0
