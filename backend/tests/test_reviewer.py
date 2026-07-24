"""Task 4.1 — reviewer: bounded read-only grounding + structured Review emission, hermetic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agents.reviewer import Reviewer
from app.core.config import ReviewerSettings, Settings
from app.graph.state import Plan, Review, Task, VerifyResult
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.tools.base import ToolContext
from app.tools.registry import build_planner_registry

from tests.fakes import FakeProvider

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)

_APPROVED_PAYLOAD = {"verdict": "approved", "issues": [], "summary": "looks correct"}


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path, run_id="t")


def _tool_call(name: str, **arguments: object) -> ChatResponse:
    return ChatResponse(content="", tool_calls=[ToolCall(id="1", name=name, arguments=dict(arguments))])


def _emit_review(payload: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="9", name=emit_tool_name(Review), arguments=payload)]
    )


def test_reviewer_grounds_then_emits_review(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, reviewer=ReviewerSettings(grounding_steps=3))
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            _tool_call("read_file", path="calc.py"),
            ChatResponse(content="I have enough context."),  # no tool call -> breaks grounding
            _emit_review(_APPROVED_PAYLOAD),
        ],
    )
    reviewer = Reviewer(provider, build_planner_registry(), settings)
    review = reviewer.review_change(
        plan=None, diff="+def add(a, b): return a + b", verify_result=None, ctx=_ctx(tmp_path)
    )

    assert review.verdict == "approved"
    assert review.issues == []
    # 3 provider calls: grounding round 1 (tool call), grounding round 2 (breaks), final emit.
    assert len(provider.calls) == 3


def test_reviewer_stops_grounding_at_step_cap(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, reviewer=ReviewerSettings(grounding_steps=2))
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            _tool_call("read_file", path="a.py"),
            _tool_call("read_file", path="a.py"),  # would loop forever without the cap
            _emit_review(_APPROVED_PAYLOAD),
        ],
    )
    reviewer = Reviewer(provider, build_planner_registry(), settings)
    review = reviewer.review_change(plan=None, diff="+x", verify_result=None, ctx=_ctx(tmp_path))
    assert review.verdict == "approved"
    assert len(provider.calls) == 3  # 2 grounding rounds (cap) + 1 emit


def test_reviewer_input_carries_plan_diff_and_verify_only(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, reviewer=ReviewerSettings(grounding_steps=0))
    provider = FakeProvider(capabilities=_CAPS, responses=[_emit_review(_APPROVED_PAYLOAD)])
    reviewer = Reviewer(provider, build_planner_registry(), settings)

    plan = Plan(
        summary="Add a calculator.",
        tasks=[Task(id="task-1", title="Add calc.py", description="d", kind="create",
                    acceptance_criteria=["calc.py defines add(a, b)"])],
    )
    verify_result = VerifyResult(passed=True, summary="all checks passed")
    reviewer.review_change(
        plan=plan, diff="+def add(a, b):\n+    return a + b", verify_result=verify_result,
        ctx=_ctx(tmp_path),
    )

    rendered = provider.calls[0].messages[1].content
    assert "Add a calculator." in rendered
    assert "calc.py defines add(a, b)" in rendered
    assert "def add(a, b)" in rendered
    assert "PASSED" in rendered


def test_reviewer_registry_is_read_only() -> None:
    # Isolation (ADR-0006): the reviewer must never be able to edit code.
    registry = build_planner_registry()
    for forbidden in ("write_file", "edit_file", "run_command", "finish_task", "git_commit"):
        assert forbidden not in registry
