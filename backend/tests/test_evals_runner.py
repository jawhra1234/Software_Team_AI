"""Phase 5 — run_graph capture against a scripted (FakeProvider) graph (hermetic).

Proves the runner reduces a real compiled-graph run into a correct
GraphRunResult (status, verify, review verdicts, steps) without a live model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.clock import now_iso
from app.core.config import PlannerSettings, ReviewerSettings, Settings
from app.evals.runner import run_graph
from app.graph.build_graph import build_graph
from app.graph.state import Plan, Review, new_run_state
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.tools.git import Git
from app.tools.sandbox import SubprocessSandbox
from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import FakeProvider

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


def _sandbox() -> SubprocessSandbox:
    return SubprocessSandbox(
        Settings(_env_file=None).sandbox.model_copy(update={"backend": "subprocess"})
    )


def test_run_graph_captures_happy_path(tmp_path: Path) -> None:
    git = Git(tmp_path)
    git.init()
    base = git.commit("init")

    settings = Settings(
        _env_file=None,
        planner=PlannerSettings(grounding_steps=0),
        reviewer=ReviewerSettings(grounding_steps=0),
    )
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        planner_provider=FakeProvider(capabilities=_CAPS, responses=[
            _emit(Plan, {"summary": "add calc",
                         "tasks": [{"id": "task-1", "title": "calc", "description": "d",
                                    "kind": "create"}]}),
        ]),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[
            _multi(("write_file", {"path": "calc.py", "content": _PASSING}),
                   ("write_file", {"path": "test_calc.py", "content": _TEST})),
            ChatResponse(content="", tool_calls=[
                ToolCall(id="f", name="finish_task", arguments={"summary": "done"})]),
        ]),
        reviewer_provider=FakeProvider(capabilities=_CAPS, responses=[
            _emit(Review, {"verdict": "approved", "issues": [], "summary": "ok"}),
        ]),
    )

    state = dict(new_run_state(
        run_id="r1", project_id="p1", user_request="build calc",
        workspace_path=str(tmp_path), autonomy_level="auto",
        max_tokens=None, max_steps=100, max_wall_clock_s=3600, started_at=now_iso(),
    ))
    state["base_commit"] = base
    state["work_branch"] = git.current_branch()

    res = run_graph(graph, state, {"configurable": {"thread_id": "r1"}})

    assert res.status == "succeeded"
    assert res.verify_passed is True
    assert res.verify_retries == 0
    assert res.review_verdicts == ["approved"]
    assert res.review_flagged_blocking is False
    assert res.steps > 0  # node_history captured
    assert res.wall_clock_s >= 0.0
