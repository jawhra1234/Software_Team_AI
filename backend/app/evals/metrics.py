"""Suite-level metric aggregation (Phase 5).

Pure functions over a list of :class:`RunReport` — no I/O, no live model — so
the whole scoring path is unit-testable from hand-built reports.

Metrics are split by how reproducible they are, and that split is the whole
point (see PHASE-5.md):

* **Deterministic** — ``retrieval_precision_at_k``. Same index + embeddings →
  same number every run, so the regression gate is *allowed* to fail the build
  on it.
* **Stochastic** — everything else (success/defect-detection/false-flag rates,
  avg review cycles, steps, wall-clock). A local 7B jitters run to run; these
  are reported as trends and never hard-gated.

Rates are ``value / count``; a rate over zero applicable tasks is ``None`` (not
0.0) so "no data" never masquerades as "0%".
"""

from __future__ import annotations

from dataclasses import dataclass

from app.evals.report import (
    CLEAN_CODE,
    CROSS_RUN_MEMORY,
    DEFECT_INJECTION,
    RETRIEVAL,
    SUCCESS_EXPECTED,
    RunReport,
)

#: Metric keys the regression gate is allowed to fail the build on.
DETERMINISTIC_KEYS = ("retrieval_precision_at_k",)


@dataclass
class SuiteMetrics:
    """Aggregated numbers for one suite run."""

    # deterministic (gate-worthy)
    retrieval_precision_at_k: float | None
    # stochastic (trend-only)
    task_success_rate: float | None
    defect_detection_rate: float | None
    false_flag_rate: float | None
    memory_influence_rate: float | None
    avg_review_cycles: float | None
    total_steps: int
    total_wall_clock_s: float
    task_count: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "retrieval_precision_at_k": self.retrieval_precision_at_k,
            "task_success_rate": self.task_success_rate,
            "defect_detection_rate": self.defect_detection_rate,
            "false_flag_rate": self.false_flag_rate,
            "memory_influence_rate": self.memory_influence_rate,
            "avg_review_cycles": self.avg_review_cycles,
            "total_steps": self.total_steps,
            "total_wall_clock_s": round(self.total_wall_clock_s, 2),
            "task_count": self.task_count,
        }


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or None when no tasks are applicable (so it never reads as 0%)."""
    return numerator / denominator if denominator else None


def aggregate(reports: list[RunReport]) -> SuiteMetrics:
    """Reduce per-task reports into suite-level metrics (see module docstring)."""
    graph_reports = [r for r in reports if r.category != RETRIEVAL and r.error is None]

    # Success rate: only over categories expected to end `succeeded`.
    success_pool = [r for r in graph_reports if r.category in SUCCESS_EXPECTED]
    successes = sum(1 for r in success_pool if r.status == "succeeded")

    # Defect detection: over defect-injection tasks, did the reviewer raise a blocker/major?
    defect_pool = [r for r in graph_reports if r.category == DEFECT_INJECTION]
    defects_caught = sum(1 for r in defect_pool if r.review_flagged_blocking)

    # False flags: over clean-code tasks, the reviewer should NOT have blocked.
    clean_pool = [r for r in graph_reports if r.category in CLEAN_CODE]
    false_flags = sum(1 for r in clean_pool if r.review_flagged_blocking)

    # Memory influence: over cross-run-memory tasks.
    memory_pool = [r for r in graph_reports if r.category == CROSS_RUN_MEMORY]
    memory_hits = sum(1 for r in memory_pool if r.memory_influenced)

    reviewed = [r for r in graph_reports if r.review_cycles > 0]
    avg_cycles = (
        sum(r.review_cycles for r in reviewed) / len(reviewed) if reviewed else None
    )

    # Deterministic: precision@k comes from the retrieval task(s), if present.
    retrieval_reports = [
        r for r in reports if r.category == RETRIEVAL and r.precision_at_k is not None
    ]
    precision = (
        sum(r.precision_at_k for r in retrieval_reports) / len(retrieval_reports)  # type: ignore[misc]
        if retrieval_reports
        else None
    )

    return SuiteMetrics(
        retrieval_precision_at_k=precision,
        task_success_rate=_rate(successes, len(success_pool)),
        defect_detection_rate=_rate(defects_caught, len(defect_pool)),
        false_flag_rate=_rate(false_flags, len(clean_pool)),
        memory_influence_rate=_rate(memory_hits, len(memory_pool)),
        avg_review_cycles=avg_cycles,
        total_steps=sum(r.steps for r in reports),
        total_wall_clock_s=sum(r.wall_clock_s for r in reports),
        task_count=len(reports),
    )
