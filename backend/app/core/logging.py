"""Structured logging (Task 0.7).

JSON logging via ``structlog`` with a run/trace id threaded through every log
line using context vars, so a run can be followed end to end. ``configure_logging``
is idempotent and safe to call at process start.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure ``structlog`` for JSON (prod) or console (dev) output."""
    level_name = settings.log_level.upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)


def bind_run_context(*, run_id: str, trace_id: str | None = None) -> None:
    """Bind run/trace identifiers into the logging context for this task."""
    context: dict[str, str] = {"run_id": run_id}
    if trace_id is not None:
        context["trace_id"] = trace_id
    structlog.contextvars.bind_contextvars(**context)


def clear_run_context() -> None:
    """Clear any bound run/trace context (call at the end of a run)."""
    structlog.contextvars.clear_contextvars()
