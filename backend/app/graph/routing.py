"""Conditional edge functions (Task 2.10).

Every routing function is a pure function of state — none of them write state
(LangGraph edges can't). The convention that makes this possible: every node
that can escalate always sets ``hitl_request`` explicitly, to either a real
``HITLRequest`` or ``None`` — never leaving a stale value from an earlier
cycle. Routing therefore only ever needs to check ``hitl_request`` presence to
know whether to detour through ``human_gate``.
"""

from __future__ import annotations

from app.graph.state import AgentState

_ESCALATION_ORIGINS = ("coder", "plan")


def route_after_plan(state: AgentState) -> str:
    if state.get("hitl_request") is not None:
        return "human_gate"  # plan_approval gate (semi/manual) or a planning failure
    return "coder"


def route_after_gate(state: AgentState) -> str:
    request = state.get("hitl_request")
    response = state.get("hitl_response")
    if request is None or response is None:
        return "finalize"  # defensive: shouldn't happen given routing guarantees

    decision = response.decision
    if request.kind == "plan_approval":
        if decision == "approve":
            return "coder"
        if decision == "revise":
            return "plan"
        return "finalize"  # abort

    if request.kind == "escalation":
        if decision == "retry":
            origin = request.payload.get("origin_node", "coder")
            return origin if origin in _ESCALATION_ORIGINS else "coder"
        return "finalize"  # accept-as-is or abort

    if request.kind == "final_accept":
        return "finalize" if decision == "accept" else "coder"  # request_changes

    return "finalize"


def route_after_coder(state: AgentState) -> str:
    if state.get("hitl_request") is not None:
        return "human_gate"
    if state.get("current_task_id") is not None:
        return "coder"
    return "verify"


def route_after_verify(state: AgentState) -> str:
    if state.get("hitl_request") is not None:
        return "human_gate"
    result = state.get("verify_result")
    if result is not None and result.passed:
        return "review"
    return "coder"


def route_after_review(state: AgentState) -> str:
    if state.get("hitl_request") is not None:
        return "human_gate"
    review = state.get("review")
    if review is not None and review.verdict == "approved":
        return "finalize"
    return "coder"
