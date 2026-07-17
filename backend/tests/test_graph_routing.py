"""Task 2.10 — routing functions: every branch, driven purely by state."""

from __future__ import annotations

from app.graph.routing import (
    route_after_coder,
    route_after_gate,
    route_after_plan,
    route_after_review,
    route_after_verify,
)
from app.graph.state import HITLRequest, HITLResponse, Review, VerifyResult


def _state(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"hitl_request": None}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# route_after_plan
# ---------------------------------------------------------------------------
def test_route_after_plan_no_request_goes_to_coder() -> None:
    assert route_after_plan(_state()) == "coder"  # type: ignore[arg-type]


def test_route_after_plan_with_request_goes_to_human_gate() -> None:
    state = _state(hitl_request=HITLRequest(kind="plan_approval", context="x"))
    assert route_after_plan(state) == "human_gate"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# route_after_gate
# ---------------------------------------------------------------------------
def test_route_after_gate_plan_approval_approve() -> None:
    state = _state(
        hitl_request=HITLRequest(kind="plan_approval", context="x"),
        hitl_response=HITLResponse(decision="approve"),
    )
    assert route_after_gate(state) == "coder"  # type: ignore[arg-type]


def test_route_after_gate_plan_approval_revise() -> None:
    state = _state(
        hitl_request=HITLRequest(kind="plan_approval", context="x"),
        hitl_response=HITLResponse(decision="revise"),
    )
    assert route_after_gate(state) == "plan"  # type: ignore[arg-type]


def test_route_after_gate_plan_approval_abort() -> None:
    state = _state(
        hitl_request=HITLRequest(kind="plan_approval", context="x"),
        hitl_response=HITLResponse(decision="abort"),
    )
    assert route_after_gate(state) == "finalize"  # type: ignore[arg-type]


def test_route_after_gate_escalation_retry_uses_origin() -> None:
    state = _state(
        hitl_request=HITLRequest(
            kind="escalation", context="x", payload={"origin_node": "plan"}
        ),
        hitl_response=HITLResponse(decision="retry"),
    )
    assert route_after_gate(state) == "plan"  # type: ignore[arg-type]


def test_route_after_gate_escalation_retry_defaults_to_coder() -> None:
    state = _state(
        hitl_request=HITLRequest(kind="escalation", context="x", payload={}),
        hitl_response=HITLResponse(decision="retry"),
    )
    assert route_after_gate(state) == "coder"  # type: ignore[arg-type]


def test_route_after_gate_escalation_accept_or_abort_goes_to_finalize() -> None:
    for decision in ("accept", "abort"):
        state = _state(
            hitl_request=HITLRequest(kind="escalation", context="x"),
            hitl_response=HITLResponse(decision=decision),
        )
        assert route_after_gate(state) == "finalize"  # type: ignore[arg-type]


def test_route_after_gate_final_accept() -> None:
    accept = _state(
        hitl_request=HITLRequest(kind="final_accept", context="x"),
        hitl_response=HITLResponse(decision="accept"),
    )
    assert route_after_gate(accept) == "finalize"  # type: ignore[arg-type]

    request_changes = _state(
        hitl_request=HITLRequest(kind="final_accept", context="x"),
        hitl_response=HITLResponse(decision="request_changes"),
    )
    assert route_after_gate(request_changes) == "coder"  # type: ignore[arg-type]


def test_route_after_gate_defensive_fallback_without_response() -> None:
    state = _state(hitl_request=HITLRequest(kind="plan_approval", context="x"), hitl_response=None)
    assert route_after_gate(state) == "finalize"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# route_after_coder
# ---------------------------------------------------------------------------
def test_route_after_coder_escalates_when_hitl_set() -> None:
    state = _state(hitl_request=HITLRequest(kind="escalation", context="x"))
    assert route_after_coder(state) == "human_gate"  # type: ignore[arg-type]


def test_route_after_coder_more_tasks_loops_back() -> None:
    assert route_after_coder(_state(current_task_id="task-2")) == "coder"  # type: ignore[arg-type]


def test_route_after_coder_all_done_goes_to_verify() -> None:
    assert route_after_coder(_state(current_task_id=None)) == "verify"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# route_after_verify
# ---------------------------------------------------------------------------
def test_route_after_verify_pass_goes_to_review() -> None:
    state = _state(verify_result=VerifyResult(passed=True))
    assert route_after_verify(state) == "review"  # type: ignore[arg-type]


def test_route_after_verify_fail_goes_to_coder() -> None:
    state = _state(verify_result=VerifyResult(passed=False))
    assert route_after_verify(state) == "coder"  # type: ignore[arg-type]


def test_route_after_verify_escalates() -> None:
    state = _state(
        verify_result=VerifyResult(passed=False),
        hitl_request=HITLRequest(kind="escalation", context="x"),
    )
    assert route_after_verify(state) == "human_gate"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# route_after_review
# ---------------------------------------------------------------------------
def test_route_after_review_approved_goes_to_finalize() -> None:
    state = _state(review=Review(verdict="approved"))
    assert route_after_review(state) == "finalize"  # type: ignore[arg-type]


def test_route_after_review_changes_requested_goes_to_coder() -> None:
    state = _state(review=Review(verdict="changes_requested"))
    assert route_after_review(state) == "coder"  # type: ignore[arg-type]


def test_route_after_review_escalates() -> None:
    state = _state(
        review=Review(verdict="changes_requested"),
        hitl_request=HITLRequest(kind="escalation", context="x"),
    )
    assert route_after_review(state) == "human_gate"  # type: ignore[arg-type]
