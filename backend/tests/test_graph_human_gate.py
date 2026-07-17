"""Task 2.7 — human_gate node: interrupt/resume, plan edits, status normalization.

Uses a tiny single-node graph (the human_gate node itself) with an in-memory
checkpointer to exercise the real ``interrupt()``/``Command(resume=...)``
mechanics without needing the full 6-node graph.
"""

from __future__ import annotations

from typing import Any

from app.core.clock import now_iso
from app.graph.nodes.human_gate import _apply_plan_edits, make_human_gate_node
from app.graph.state import AgentState, HITLRequest, Plan, Task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


def _compiled() -> Any:
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("human_gate", make_human_gate_node())
    graph.add_edge(START, "human_gate")
    graph.add_edge("human_gate", END)
    return graph.compile(checkpointer=InMemorySaver())


def _initial_state(hitl_request: HITLRequest) -> dict[str, Any]:
    return {
        "run_id": "r1", "project_id": "p1", "thread_id": "r1",
        "user_request": "x", "clarification_answers": [], "workspace_path": ".",
        "changed_files": [], "coder_scratch": [], "autonomy_level": "manual",
        "hitl_request": hitl_request, "retries": {}, "errors": [], "node_history": [],
        "status": "planning",
        "budget": {
            "max_tokens": None, "max_steps": 50, "max_wall_clock_s": 3600.0,
            "tokens_used": 0, "steps_used": 0, "started_at": now_iso(),
        },
    }


def test_interrupt_pauses_and_resume_approves() -> None:
    compiled = _compiled()
    config = {"configurable": {"thread_id": "t1"}}
    request = HITLRequest(kind="plan_approval", context="my plan", options=["approve", "revise", "abort"])

    result = compiled.invoke(_initial_state(request), config=config)
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["kind"] == "plan_approval"

    resumed = compiled.invoke(Command(resume={"decision": "approve"}), config=config)
    assert resumed["hitl_response"].decision == "approve"


def test_resume_with_bare_string_decision() -> None:
    compiled = _compiled()
    config = {"configurable": {"thread_id": "t2"}}
    request = HITLRequest(kind="escalation", context="oops")
    compiled.invoke(_initial_state(request), config=config)
    resumed = compiled.invoke(Command(resume="abort"), config=config)
    assert resumed["hitl_response"].decision == "abort"
    assert resumed["status"] == "failed"


def test_plan_approval_abort_sets_cancelled_status() -> None:
    compiled = _compiled()
    config = {"configurable": {"thread_id": "t3"}}
    request = HITLRequest(kind="plan_approval", context="p")
    compiled.invoke(_initial_state(request), config=config)
    resumed = compiled.invoke(Command(resume={"decision": "abort"}), config=config)
    assert resumed["status"] == "cancelled"


def test_plan_approval_revise_with_edits_patches_plan() -> None:
    compiled = _compiled()
    config = {"configurable": {"thread_id": "t4"}}
    request = HITLRequest(kind="plan_approval", context="p")
    state = _initial_state(request)
    state["plan"] = Plan(summary="old", tasks=[Task(id="t1", title="t", description="d", kind="create")])
    compiled.invoke(state, config=config)
    resumed = compiled.invoke(
        Command(resume={"decision": "revise", "edits": {"summary": "new summary"}}), config=config
    )
    assert resumed["plan"].summary == "new summary"
    assert resumed["plan"].version == 2


def test_final_accept_accept_decision() -> None:
    compiled = _compiled()
    config = {"configurable": {"thread_id": "t5"}}
    request = HITLRequest(kind="final_accept", context="review ok")
    compiled.invoke(_initial_state(request), config=config)
    resumed = compiled.invoke(Command(resume={"decision": "accept"}), config=config)
    assert resumed["hitl_response"].decision == "accept"
    assert resumed.get("status") != "failed"  # unaffected by final_accept


# ---------------------------------------------------------------------------
# _apply_plan_edits unit tests (no graph needed)
# ---------------------------------------------------------------------------
def test_apply_plan_edits_whitelisted_fields_only() -> None:
    plan = Plan(summary="old", architecture_notes="n1")
    edited = _apply_plan_edits(plan, {"summary": "new", "not_a_real_field": "ignored"})
    assert edited is not None
    assert edited.summary == "new"
    assert edited.version == 2
    assert not hasattr(edited, "not_a_real_field")


def test_apply_plan_edits_replaces_tasks() -> None:
    plan = Plan(summary="s", tasks=[Task(id="t1", title="a", description="d", kind="create")])
    edited = _apply_plan_edits(
        plan, {"tasks": [{"id": "t2", "title": "b", "description": "d2", "kind": "modify"}]}
    )
    assert edited is not None
    assert edited.tasks[0].id == "t2"


def test_apply_plan_edits_noop_without_edits() -> None:
    plan = Plan(summary="s")
    assert _apply_plan_edits(plan, {}) is plan
    assert _apply_plan_edits(None, {"summary": "x"}) is None
