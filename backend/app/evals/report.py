"""Per-task run report (Phase 5).

A ``RunReport`` is the structured, JSON-serializable record of one eval task's
outcome — the single unit both the metrics aggregator and the regression
comparator consume. It is plain data with no behavior, so the whole scoring
path is testable from hand-built reports without ever running a live model.

Not every field applies to every task: graph-run tasks populate
``status``/``verify_*``/``review_*``/retrieval/memory fields; the retrieval-only
task populates ``precision_at_k``. Unused fields keep their neutral defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Task categories. Each drives which metrics a report contributes to
# (see app/evals/metrics.py) — kept as constants so tasks and metrics agree.
HAPPY_PATH = "happy_path"
RAG_REQUIRED = "rag_required"
DEFECT_INJECTION = "defect_injection"
CROSS_RUN_MEMORY = "cross_run_memory"
RETRIEVAL = "retrieval"

#: Categories whose run is *expected to end* ``succeeded`` (contribute to success rate).
SUCCESS_EXPECTED = frozenset({HAPPY_PATH, RAG_REQUIRED, CROSS_RUN_MEMORY})
#: Categories whose code is *correct*, so the reviewer should NOT block (false-flag signal).
CLEAN_CODE = frozenset({HAPPY_PATH, RAG_REQUIRED})


@dataclass
class RunReport:
    """One eval task's outcome. Serializable to/from a plain dict for the baseline."""

    task_id: str
    category: str

    # --- graph-run outcome (None/0 for the retrieval-only task) ---
    status: str | None = None  # succeeded | failed | cancelled | error
    verify_passed: bool | None = None
    verify_retries: int = 0
    #: One entry per review-node visit, in order (the *effective* verdict the node routed on).
    review_verdicts: list[str] = field(default_factory=list)
    #: True if any review cycle raised a blocker/major (i.e. requested changes on substance).
    review_flagged_blocking: bool = False
    #: Symbols surfaced by ``retrieve`` across the run (dedup, order-preserving).
    retrieved_symbols: list[str] = field(default_factory=list)
    #: RAG task only — did the final code actually reuse the expected helper symbol?
    expected_symbol_reused: bool | None = None
    #: Cross-run-memory task only — did run 2's planner context carry run 1's record?
    memory_influenced: bool | None = None
    #: Graph node visits (from ``node_history``) — a work/effort proxy (tokens aren't tracked).
    steps: int = 0
    wall_clock_s: float = 0.0

    # --- retrieval-only task ---
    precision_at_k: float | None = None

    #: Set when the task raised instead of producing a clean outcome (harness error, not a
    #: model failure — kept distinct so it doesn't silently count as a task failure).
    error: str | None = None
    notes: str = ""

    @property
    def review_cycles(self) -> int:
        return len(self.review_verdicts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunReport:
        # Tolerate unknown keys (forward-compat with older/newer baselines).
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
