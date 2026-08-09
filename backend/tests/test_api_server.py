"""Phase 6 — Mission-Control API end to end via TestClient (hermetic, scripted providers).

Drives a full run through the HTTP surface with FakeProviders — no Ollama — proving
start → live events → HITL respond → succeeded works over the API, and that the run
manager wraps the real graph without touching it.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from app.api.run_manager import RunManager
from app.api.server import create_app
from app.core.config import PlannerSettings, ReviewerSettings, Settings
from app.graph.state import Plan, Review
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.tools.sandbox import SubprocessSandbox

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)
_PASSING = "def add(a, b):\n    return a + b\n"
_TEST = "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def _multi(*calls: tuple[str, dict[str, Any]]) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[ToolCall(id=str(i), name=n, arguments=a) for i, (n, a) in enumerate(calls)],
    )


def _emit(schema: type, payload: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="e", name=emit_tool_name(schema), arguments=payload)]
    )


def _fake(responses: list[ChatResponse]) -> Any:
    from tests.fakes import FakeProvider

    return FakeProvider(capabilities=_CAPS, responses=responses)


def _client(autonomy_providers: dict[str, Any]) -> TestClient:
    settings = Settings(
        _env_file=None,
        planner=PlannerSettings(grounding_steps=0),
        reviewer=ReviewerSettings(grounding_steps=0),
    )
    sandbox = SubprocessSandbox(settings.sandbox.model_copy(update={"backend": "subprocess"}))
    manager = RunManager(settings, sandbox=sandbox, provider_overrides=autonomy_providers)
    return TestClient(create_app(manager))


def _drain_events(client: TestClient, run_id: str, timeout_s: float = 60.0) -> list[dict[str, Any]]:
    """Read the SSE stream to its end marker, returning parsed event dicts."""
    events: list[dict[str, Any]] = []
    deadline = time.time() + timeout_s
    with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if time.time() > deadline:
                raise AssertionError("SSE stream did not end in time")
            if line.startswith("data: "):
                payload = line[len("data: "):]
                if payload.strip() and payload.strip() != "{}":
                    events.append(json.loads(payload))
            if any(e.get("type") in ("done", "error") for e in events):
                break
    return events


def test_health() -> None:
    client = _client({})
    assert client.get("/api/health").json() == {"status": "ok"}


def test_auto_run_streams_to_succeeded() -> None:
    """auto autonomy: no gates — start, stream node events, reach done/succeeded."""
    providers = {
        "planner": _fake([_emit(Plan, {
            "summary": "add calc",
            "tasks": [{"id": "task-1", "title": "calc", "description": "d", "kind": "create"}],
        })]),
        "coder": _fake([
            _multi(("write_file", {"path": "calc.py", "content": _PASSING}),
                   ("write_file", {"path": "test_calc.py", "content": _TEST})),
            ChatResponse(content="", tool_calls=[
                ToolCall(id="f", name="finish_task", arguments={"summary": "done"})]),
        ]),
        "reviewer": _fake([_emit(Review, {"verdict": "approved", "issues": [], "summary": "ok"})]),
    }
    client = _client(providers)
    run_id = client.post("/api/runs", json={"request": "build calc", "autonomy": "auto"}).json()["run_id"]

    events = _drain_events(client, run_id)
    kinds = [e.get("type") for e in events]
    assert "node_start" in kinds  # the sink's node lifecycle events came through
    # Phase-6 live activity: individual tool calls stream as "tool" events so the UI
    # isn't blank while a long-running node works (the coder writes files here).
    tool_events = [e for e in events if e.get("type") == "tool"]
    assert tool_events, "expected streamed tool events"
    assert any((e.get("data") or {}).get("tool") == "write_file" for e in tool_events)
    done = next(e for e in events if e.get("type") == "done")
    assert done["status"] == "succeeded"

    snap = client.get(f"/api/runs/{run_id}").json()
    assert snap["status"] == "done"
    assert snap["final_state"]["review"]["verdict"] == "approved"


def test_semi_run_pauses_for_plan_approval_then_resumes() -> None:
    """semi autonomy: the run pauses at the plan-approval gate; POST /respond resumes it."""
    providers = {
        "planner": _fake([_emit(Plan, {
            "summary": "add calc",
            "tasks": [{"id": "task-1", "title": "calc", "description": "d", "kind": "create"}],
        })]),
        "coder": _fake([
            _multi(("write_file", {"path": "calc.py", "content": _PASSING}),
                   ("write_file", {"path": "test_calc.py", "content": _TEST})),
            ChatResponse(content="", tool_calls=[
                ToolCall(id="f", name="finish_task", arguments={"summary": "done"})]),
        ]),
        "reviewer": _fake([_emit(Review, {"verdict": "approved", "issues": [], "summary": "ok"})]),
    }
    client = _client(providers)
    run_id = client.post("/api/runs", json={"request": "build calc", "autonomy": "semi"}).json()["run_id"]

    # Wait for the interrupt to surface.
    deadline = time.time() + 30
    while time.time() < deadline:
        snap = client.get(f"/api/runs/{run_id}").json()
        if snap["status"] == "waiting_human":
            break
        time.sleep(0.1)
    assert snap["status"] == "waiting_human"
    assert snap["pending_interrupt"]["kind"] == "plan_approval"

    # Answer the gate → run completes.
    assert client.post(f"/api/runs/{run_id}/respond", json={"decision": "approve"}).json()["ok"]
    events = _drain_events(client, run_id)
    done = next(e for e in events if e.get("type") == "done")
    assert done["status"] == "succeeded"


def test_respond_when_not_waiting_is_409() -> None:
    client = _client({})
    # unknown run -> 404
    assert client.post("/api/runs/nope/respond", json={"decision": "approve"}).status_code == 404
