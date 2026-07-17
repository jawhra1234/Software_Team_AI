"""Task 2.2 graph wiring — plan node: approval gating, clarification interrupt, failure escalation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agents.planner import Planner
from app.core.clock import now_iso
from app.core.config import PlannerSettings, Settings
from app.graph.nodes.plan import make_plan_node
from app.graph.state import AgentState, Plan, new_run_state
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.tools.registry import build_planner_registry
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from tests.fakes import FakeProvider

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)


def _emit(payload: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="9", name=emit_tool_name(Plan), arguments=payload)]
    )


_GOOD_PLAN = {
    "summary": "Add a calculator.",
    "tasks": [
        {"id": "task-1", "title": "add calc.py", "description": "d", "kind": "create"}
    ],
}
_BLOCKING_PLAN = {
    "summary": "Need clarity.",
    "open_questions": ["Should this support floats or integers only?"],
    "tasks": [],
}


def _state(tmp_path: Path, autonomy: str = "auto") -> AgentState:
    return new_run_state(
        run_id="r1", project_id="p1", user_request="build a calculator",
        workspace_path=str(tmp_path), autonomy_level=autonomy,  # type: ignore[arg-type]
        max_tokens=None, max_steps=50, max_wall_clock_s=3600,
        started_at=now_iso(),
    )


def _planner(responses: list[ChatResponse], grounding_steps: int = 1) -> Planner:
    settings = Settings(_env_file=None, planner=PlannerSettings(grounding_steps=grounding_steps))
    provider = FakeProvider(capabilities=_CAPS, responses=responses)
    return Planner(provider, build_planner_registry(), settings)


def test_plan_node_auto_autonomy_skips_approval_gate(tmp_path: Path) -> None:
    node = make_plan_node(_planner([ChatResponse(content="no grounding"), _emit(_GOOD_PLAN)]))
    patch = node(_state(tmp_path, autonomy="auto"))
    assert patch["plan"].summary == "Add a calculator."
    assert patch["hitl_request"] is None


def test_plan_node_semi_autonomy_requests_approval(tmp_path: Path) -> None:
    node = make_plan_node(_planner([ChatResponse(content="no grounding"), _emit(_GOOD_PLAN)]))
    patch = node(_state(tmp_path, autonomy="semi"))
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "plan_approval"


def test_plan_node_empty_task_list_escalates(tmp_path: Path) -> None:
    empty_plan = {"summary": "nothing to do", "tasks": []}
    node = make_plan_node(_planner([ChatResponse(content="no grounding"), _emit(empty_plan)]))
    patch = node(_state(tmp_path, autonomy="auto"))
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "escalation"
    assert patch["errors"][0].kind == "planning_failed"


def test_plan_node_structured_failure_escalates(tmp_path: Path) -> None:
    # Every response is unparseable prose -> structured_call exhausts repairs and raises.
    node = make_plan_node(
        _planner([ChatResponse(content="not json") for _ in range(5)], grounding_steps=0)
    )
    patch = node(_state(tmp_path, autonomy="auto"))
    assert patch["hitl_request"].kind == "escalation"
    assert "Planning failed" in patch["hitl_request"].context


def test_plan_node_clarification_interrupt_then_resume(tmp_path: Path) -> None:
    # LangGraph replays a node from the top on resume: the *draft* create_plan
    # call (the one producing open_questions) runs once before the original
    # pause and once again during replay, before interrupt() returns its
    # cached answer — so the FakeProvider must serve the same blocking draft
    # twice. Only the call *after* the interrupt line is genuinely new. This
    # mirrors the replay characteristic documented in graph/nodes/coder.py for
    # command_approval.
    planner = _planner(
        [
            ChatResponse(content="no grounding"),
            _emit(_BLOCKING_PLAN),  # draft, original execution -> hits interrupt()
            ChatResponse(content="no grounding"),
            _emit(_BLOCKING_PLAN),  # draft, replayed on resume -> interrupt() returns cached answer
            ChatResponse(content="no grounding"),
            _emit(_GOOD_PLAN),  # final plan, incorporating the human's answer
        ],
        grounding_steps=1,
    )
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("plan", make_plan_node(planner))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", END)
    compiled = graph.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t1"}}
    state = _state(tmp_path, autonomy="semi")
    paused = compiled.invoke(state, config=config)  # type: ignore[call-overload]
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["kind"] == "clarification"

    resumed = compiled.invoke(  # type: ignore[call-overload]
        Command(resume={"answers": ["integers only"]}), config=config
    )
    assert resumed["plan"].summary == "Add a calculator."
    assert resumed["clarification_answers"] == ["integers only"]
    assert resumed["plan"].open_questions == []
