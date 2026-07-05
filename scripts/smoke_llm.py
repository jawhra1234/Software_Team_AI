"""Phase-0 smoke: a validated structured LLM call, traced (Task 0.9 / DoD).

Performs one end-to-end structured call against the configured provider and,
when Langfuse is enabled, records the generation so a trace appears in the UI.

    # from repo root, with the backend venv active:
    python scripts/smoke_llm.py

Requires a running Ollama with the configured chat model pulled. Exits non-zero
on failure so it can gate a bootstrap check.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the backend package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pydantic import BaseModel, Field  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.core.tracing import get_tracer, trace_run  # noqa: E402
from app.providers.base import ChatMessage  # noqa: E402
from app.providers.factory import get_provider  # noqa: E402


class HealthReport(BaseModel):
    """Tiny schema the model must populate — proves structured output works."""

    status: str = Field(description="one word: 'ok' if you can read this")
    language: str = Field(description="the programming language best for quick scripts")
    confidence: float = Field(description="0..1 confidence in the answer")


def main() -> int:
    settings = get_settings()
    configure_logging(settings)
    log = get_logger("smoke")
    tracer = get_tracer(settings)

    provider = get_provider("coder", settings)
    messages = [
        ChatMessage(role="system", content="You are a precise assistant."),
        ChatMessage(
            role="user",
            content="Report your health as a HealthReport object.",
        ),
    ]

    with trace_run(run_id="smoke-0"):
        log.info("smoke_start", model=provider.model, provider=settings.provider)
        report = provider.structured(messages, HealthReport)
        log.info("smoke_result", report=report.model_dump())
        tracer.record_generation(
            name="smoke_llm",
            model=provider.model,
            input=[m.model_dump() for m in messages],
            output=report.model_dump(),
            metadata={"phase": "0"},
        )
        tracer.flush()

    print(f"OK: {report.model_dump()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
