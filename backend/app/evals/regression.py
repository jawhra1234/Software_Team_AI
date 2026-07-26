"""Baseline persistence + regression comparison (Phase 5).

Saves a suite run's metrics to a JSON baseline and diffs a later run against
it. The **gate** (a build-failing regression) applies *only* to the
deterministic metrics (``metrics.DETERMINISTIC_KEYS``): a local 7B jitters run
to run, so hard-gating a stochastic metric would fire on noise, not real
regressions. Stochastic metrics are still diffed and reported — as advisory
trend, never a gate.

The baseline is a plain JSON file (git-diffable, no new infra); ``None`` means
"no baseline yet" and the first run simply records one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.evals.metrics import DETERMINISTIC_KEYS, SuiteMetrics

#: A deterministic metric may dip by at most this (float noise / k-rounding) before
#: it's called a regression. Deterministic metrics are reproducible, so this is tiny.
_GATE_TOLERANCE = 1e-9


@dataclass
class MetricDelta:
    key: str
    baseline: float | int | None
    current: float | int | None
    deterministic: bool

    @property
    def direction(self) -> str:
        if self.baseline is None or self.current is None:
            return "n/a"
        if self.current > self.baseline:
            return "up"
        if self.current < self.baseline:
            return "down"
        return "same"

    @property
    def is_gated_regression(self) -> bool:
        """A build-failing regression: a *deterministic* metric that dropped materially."""
        if not self.deterministic or self.baseline is None or self.current is None:
            return False
        return self.current < self.baseline - _GATE_TOLERANCE


@dataclass
class RegressionResult:
    deltas: list[MetricDelta]

    @property
    def gated_regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.is_gated_regression]

    @property
    def has_gated_regression(self) -> bool:
        return bool(self.gated_regressions)


def compare(
    current: SuiteMetrics, baseline: dict[str, float | int | None] | None
) -> RegressionResult:
    """Diff ``current`` metrics against a saved ``baseline`` dict (None → first run)."""
    cur = current.to_dict()
    base = baseline or {}
    deltas = [
        MetricDelta(
            key=key,
            baseline=base.get(key),
            current=value,
            deterministic=key in DETERMINISTIC_KEYS,
        )
        for key, value in cur.items()
    ]
    return RegressionResult(deltas=deltas)


def load_baseline(path: Path) -> dict[str, float | int | None] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", data)  # tolerate either a bare dict or a wrapped one
    return metrics if isinstance(metrics, dict) else None


def save_baseline(path: Path, metrics: SuiteMetrics, *, stamped_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stamped_at": stamped_at, "metrics": metrics.to_dict()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
