"""Budget and loop guards for the coder ReAct loop (Task 1.10).

Bounds a task run by step count, wall-clock, and (optionally) tokens, and
detects no-progress loops (the state signature not changing across steps). The
tracker returns a human-readable *reason* when a limit trips rather than
raising, so the coder can classify the outcome and stop cleanly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.config import CoderSettings


@dataclass
class BudgetTracker:
    """Mutable budget/loop state for a single task attempt."""

    max_steps: int
    max_wall_clock_s: float
    no_progress_limit: int
    max_tokens: int | None = None

    steps: int = 0
    tokens: int = 0
    _started_at: float | None = field(default=None, repr=False)
    _last_signature: str | None = field(default=None, repr=False)
    _no_progress: int = field(default=0, repr=False)

    @classmethod
    def from_settings(cls, settings: CoderSettings) -> BudgetTracker:
        return cls(
            max_steps=settings.max_steps_per_task,
            max_wall_clock_s=settings.max_wall_clock_s,
            no_progress_limit=settings.no_progress_limit,
            max_tokens=settings.max_tokens,
        )

    def start(self) -> None:
        self._started_at = time.monotonic()

    def elapsed_s(self) -> float:
        return 0.0 if self._started_at is None else time.monotonic() - self._started_at

    def tick_step(self) -> None:
        self.steps += 1

    def add_tokens(self, count: int) -> None:
        self.tokens += count

    def exceeded_reason(self) -> str | None:
        """Return why the budget is exhausted, or None if within limits."""
        if self.steps >= self.max_steps:
            return f"step budget exhausted ({self.steps}/{self.max_steps})"
        elapsed = self.elapsed_s()
        if elapsed >= self.max_wall_clock_s:
            return f"wall-clock budget exhausted ({elapsed:.0f}s/{self.max_wall_clock_s:.0f}s)"
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            return f"token budget exhausted ({self.tokens}/{self.max_tokens})"
        return None

    def record_progress(self, signature: str) -> None:
        """Update the no-progress counter from a state signature."""
        if signature == self._last_signature:
            self._no_progress += 1
        else:
            self._no_progress = 0
            self._last_signature = signature

    def no_progress_reason(self) -> str | None:
        if self._no_progress >= self.no_progress_limit:
            return f"no progress for {self._no_progress} consecutive steps"
        return None
