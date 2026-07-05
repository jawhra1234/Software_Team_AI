"""Task 0.7 — logging (JSON + run/trace context) and tracing (no-op default)."""

from __future__ import annotations

import json

import pytest
from app.core.config import Settings
from app.core.logging import (
    bind_run_context,
    clear_run_context,
    configure_logging,
    get_logger,
)
from app.core.tracing import LangfuseTracer, NoopTracer, Tracer, get_tracer, trace_run


def test_json_logs_include_bound_run_context(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(Settings(_env_file=None, log_json=True))
    bind_run_context(run_id="run-123", trace_id="trace-abc")
    try:
        get_logger("test").info("hello", answer=42)
    finally:
        clear_run_context()

    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "hello"
    assert record["run_id"] == "run-123"
    assert record["trace_id"] == "trace-abc"
    assert record["answer"] == 42
    assert record["level"] == "info"


def test_clear_run_context_removes_ids(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(Settings(_env_file=None, log_json=True))
    bind_run_context(run_id="run-1")
    clear_run_context()
    get_logger("test").info("after_clear")
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "run_id" not in record


def test_trace_run_binds_and_clears(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(Settings(_env_file=None, log_json=True))
    with trace_run(run_id="run-xyz"):
        get_logger("test").info("inside")
    get_logger("test").info("outside")

    lines = capsys.readouterr().out.strip().splitlines()
    inside = json.loads(lines[-2])
    outside = json.loads(lines[-1])
    assert inside["run_id"] == "run-xyz"
    assert inside["trace_id"] == "run-xyz"  # defaults to run_id
    assert "run_id" not in outside


def test_tracer_is_noop_when_disabled() -> None:
    tracer = get_tracer(Settings(_env_file=None))
    assert isinstance(tracer, NoopTracer)
    # No-op methods must not raise.
    tracer.record_generation(name="x", model="m", input=[], output={})
    tracer.flush()


def test_tracers_satisfy_protocol() -> None:
    assert isinstance(NoopTracer(), Tracer)
    assert isinstance(LangfuseTracer(client=object()), Tracer)
