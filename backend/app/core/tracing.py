"""Observability / tracing (Task 0.7).

Thin, defensive wrapper over self-hosted Langfuse (``ARCHITECTURE.md §14``).
Tracing is disabled by default and degrades to a no-op when Langfuse is not
configured or the SDK is unavailable, so nothing in the system depends on a
running Langfuse instance. The ``trace_run`` context manager binds the run/trace
id into the logging context for correlation.

Note: the Langfuse SDK surface varies across major versions; generation
recording is best-effort and failures are logged, never raised.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from app.core.config import Settings, get_settings
from app.core.logging import bind_run_context, clear_run_context, get_logger

log = get_logger(__name__)


@runtime_checkable
class Tracer(Protocol):
    """Minimal tracing surface used by the application."""

    def record_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def flush(self) -> None: ...


class NoopTracer:
    """Tracer used when Langfuse is disabled or unavailable."""

    def record_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return None

    def flush(self) -> None:
        return None


class LangfuseTracer:
    """Records generations to a Langfuse client (best-effort)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def record_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._client.generation(
                name=name,
                model=model,
                input=input,
                output=output,
                metadata=metadata or {},
            )
        except Exception as exc:
            log.warning("langfuse_generation_failed", error=str(exc))

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            log.warning("langfuse_flush_failed", error=str(exc))


def get_tracer(settings: Settings | None = None) -> Tracer:
    """Return a Langfuse-backed tracer when configured, else a no-op tracer."""
    settings = settings or get_settings()
    cfg = settings.langfuse
    if not (cfg.enabled and cfg.public_key and cfg.secret_key):
        return NoopTracer()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=cfg.public_key,
            secret_key=cfg.secret_key,
            host=cfg.host,
        )
        return LangfuseTracer(client)
    except Exception as exc:
        log.warning("langfuse_init_failed", error=str(exc))
        return NoopTracer()


@contextmanager
def trace_run(*, run_id: str, trace_id: str | None = None) -> Any:
    """Bind run/trace context for the duration of a run and clear it afterwards."""
    bind_run_context(run_id=run_id, trace_id=trace_id or run_id)
    try:
        yield
    finally:
        clear_run_context()
