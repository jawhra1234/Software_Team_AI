"""Task 2.2 — planner: bounded grounding + structured Plan emission, hermetic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agents.planner import Planner
from app.agents.toolcalls import extract_tool_calls
from app.core.config import PlannerSettings, Settings
from app.graph.state import Plan
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.tools.base import ToolContext
from app.tools.registry import build_planner_registry

from tests.fakes import FakeProvider

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)

_PLAN_PAYLOAD = {
    "summary": "Add a calculator module.",
    "tasks": [
        {
            "id": "task-1",
            "title": "Add calc.py",
            "description": "Create add(a, b).",
            "kind": "create",
            "target_paths": ["calc.py"],
            "acceptance_criteria": ["calc.py defines add(a, b)"],
        }
    ],
}


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path, run_id="t")


def _tool_call(name: str, **arguments: object) -> ChatResponse:
    return ChatResponse(content="", tool_calls=[ToolCall(id="1", name=name, arguments=dict(arguments))])


def _emit_plan(payload: dict[str, Any]) -> ChatResponse:
    tool_name = emit_tool_name(Plan)
    return ChatResponse(content="", tool_calls=[ToolCall(id="9", name=tool_name, arguments=payload)])


def test_planner_grounds_then_emits_plan(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, planner=PlannerSettings(grounding_steps=3))
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            _tool_call("list_dir", path="."),
            ChatResponse(content="I have enough context."),  # no tool call -> breaks grounding
            _emit_plan(_PLAN_PAYLOAD),
        ],
    )
    planner = Planner(provider, build_planner_registry(), settings)
    plan = planner.create_plan(user_request="build a calculator", ctx=_ctx(tmp_path))

    assert plan.summary == "Add a calculator module."
    assert plan.version == 1
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "task-1"
    # 3 provider calls: grounding round 1 (tool call), grounding round 2 (breaks), final emit.
    assert len(provider.calls) == 3


def test_planner_stops_grounding_at_step_cap(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, planner=PlannerSettings(grounding_steps=2))
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            _tool_call("list_dir", path="."),
            _tool_call("list_dir", path="."),  # would loop forever without the cap
            _emit_plan(_PLAN_PAYLOAD),
        ],
    )
    planner = Planner(provider, build_planner_registry(), settings)
    plan = planner.create_plan(user_request="build a calculator", ctx=_ctx(tmp_path))
    assert plan.tasks[0].id == "task-1"
    assert len(provider.calls) == 3  # 2 grounding rounds (cap) + 1 emit


def test_planner_increments_version_on_revision(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, planner=PlannerSettings(grounding_steps=1))
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            ChatResponse(content="no grounding needed"),
            _emit_plan(_PLAN_PAYLOAD),
        ],
    )
    planner = Planner(provider, build_planner_registry(), settings)
    prior = Plan(version=1, summary="old plan")
    plan = planner.create_plan(
        user_request="build a calculator", ctx=_ctx(tmp_path), prior_plan=prior
    )
    assert plan.version == 2


def test_planner_renders_clarifications_and_prior_plan(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, planner=PlannerSettings(grounding_steps=1))
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            ChatResponse(content="no grounding needed"),
            _emit_plan(_PLAN_PAYLOAD),
        ],
    )
    planner = Planner(provider, build_planner_registry(), settings)
    prior = Plan(version=1, summary="old plan", assumptions=["use python"])
    planner.create_plan(
        user_request="build a calculator",
        ctx=_ctx(tmp_path),
        prior_plan=prior,
        clarification_answers=["use integers only"],
    )
    first_user_msg = provider.calls[0].messages[1].content
    assert "old plan" in first_user_msg
    assert "use python" in first_user_msg
    assert "use integers only" in first_user_msg


def test_planner_registry_is_read_only() -> None:
    registry = build_planner_registry()
    for forbidden in ("write_file", "edit_file", "run_command", "finish_task", "git_commit"):
        assert forbidden not in registry


def test_extract_tool_calls_shared_with_coder() -> None:
    # Sanity: the shared helper module works standalone (coder tests cover it fully).
    registry = build_planner_registry()
    calls = extract_tool_calls("", [ToolCall(id="1", name="list_dir", arguments={})], registry)
    assert calls[0].name == "list_dir"
