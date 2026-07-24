"""Task 4.2/4.3 graph wiring — review node: severity gating, isolation, cycle cap,
rejected/malformed escalation (replaces Task 2.5's deterministic review_stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agents.reviewer import Reviewer
from app.core.clock import now_iso
from app.core.config import GraphSettings, ReviewerSettings, Settings
from app.graph.nodes.review import make_review_node
from app.graph.state import AgentState, Plan, Review, VerifyResult, new_run_state
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.tools.registry import build_planner_registry

from tests.fakes import FakeProvider

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)


def _emit(payload: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="9", name=emit_tool_name(Review), arguments=payload)]
    )


def _state(tmp_path: Path, **overrides: object) -> AgentState:
    state = new_run_state(
        run_id="r1", project_id="p1", user_request="build a calculator",
        workspace_path=str(tmp_path), autonomy_level="auto",
        max_tokens=None, max_steps=50, max_wall_clock_s=3600, started_at=now_iso(),
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _reviewer(responses: list[ChatResponse], grounding_steps: int = 0) -> tuple[Reviewer, FakeProvider]:
    settings = Settings(_env_file=None, reviewer=ReviewerSettings(grounding_steps=grounding_steps))
    provider = FakeProvider(capabilities=_CAPS, responses=responses)
    return Reviewer(provider, build_planner_registry(), settings), provider


# ---------------------------------------------------------------------------
# Severity gating (Task 4.4/4.5): the effective verdict is derived from issue
# severities, not trusted blindly from the model's own `verdict` claim.
# ---------------------------------------------------------------------------
def test_review_node_approves_with_no_blocking_issues(tmp_path: Path) -> None:
    reviewer, _ = _reviewer([_emit({"verdict": "approved", "issues": [], "summary": "clean"})])
    node = make_review_node(reviewer, GraphSettings())
    patch = node(_state(tmp_path, diff_summary="+x = 1"))
    assert patch["review"].verdict == "approved"
    assert patch["hitl_request"] is None


def test_review_node_minor_and_nit_never_block(tmp_path: Path) -> None:
    payload = {
        "verdict": "approved",
        "issues": [
            {"severity": "minor", "description": "could be simplified"},
            {"severity": "nit", "description": "prefer snake_case"},
        ],
        "summary": "fine overall",
    }
    reviewer, _ = _reviewer([_emit(payload)])
    node = make_review_node(reviewer, GraphSettings())
    patch = node(_state(tmp_path, diff_summary="+x = 1"))
    assert patch["review"].verdict == "approved"
    assert patch["hitl_request"] is None


def test_review_node_overrides_false_approval_when_blocker_present(tmp_path: Path) -> None:
    # The model claims "approved" but lists a blocker — the node must not trust it.
    payload = {
        "verdict": "approved",
        "issues": [{"severity": "blocker", "description": "SQL injection in query()"}],
        "summary": "looks fine",
    }
    reviewer, _ = _reviewer([_emit(payload)])
    node = make_review_node(reviewer, GraphSettings(max_review_cycles=3))
    patch = node(_state(tmp_path, diff_summary="+x = 1", retries={"review": 0}))
    assert patch["review"].verdict == "changes_requested"
    assert patch["retries"] == {"review": 1}


def test_review_node_overrides_false_changes_requested_when_only_nits(tmp_path: Path) -> None:
    # The model claims "changes_requested" but only lists a nit — must not block.
    payload = {
        "verdict": "changes_requested",
        "issues": [{"severity": "nit", "description": "rename variable"}],
        "summary": "minor style thing",
    }
    reviewer, _ = _reviewer([_emit(payload)])
    node = make_review_node(reviewer, GraphSettings())
    patch = node(_state(tmp_path, diff_summary="+x = 1"))
    assert patch["review"].verdict == "approved"
    assert patch["hitl_request"] is None
    assert "retries" not in patch  # no cycle spent on a non-blocking review


def test_review_node_major_triggers_changes_requested(tmp_path: Path) -> None:
    payload = {
        "verdict": "changes_requested",
        "issues": [{"severity": "major", "description": "missing error handling", "file": "a.py"}],
        "summary": "one real issue",
    }
    reviewer, _ = _reviewer([_emit(payload)])
    node = make_review_node(reviewer, GraphSettings(max_review_cycles=3))
    patch = node(_state(tmp_path, diff_summary="+x = 1", retries={"review": 0}))
    assert patch["review"].verdict == "changes_requested"
    assert patch["hitl_request"] is None  # 1 < 3, no escalation yet


# ---------------------------------------------------------------------------
# Cycle cap + rejected + malformed -> escalation (Task 4.5)
# ---------------------------------------------------------------------------
def test_review_node_escalates_when_cycles_exhausted(tmp_path: Path) -> None:
    payload = {
        "verdict": "changes_requested",
        "issues": [{"severity": "major", "description": "still broken"}],
        "summary": "not there yet",
    }
    reviewer, _ = _reviewer([_emit(payload)])
    node = make_review_node(reviewer, GraphSettings(max_review_cycles=1))
    patch = node(_state(tmp_path, diff_summary="+x = 1", retries={"review": 0}))
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "escalation"
    assert patch["hitl_request"].payload["origin_node"] == "coder"


def test_review_node_rejected_escalates_immediately_regardless_of_cycle_count(tmp_path: Path) -> None:
    payload = {"verdict": "rejected", "issues": [], "summary": "approach is fundamentally wrong"}
    reviewer, _ = _reviewer([_emit(payload)])
    node = make_review_node(reviewer, GraphSettings(max_review_cycles=10))  # cap far from exhausted
    patch = node(_state(tmp_path, diff_summary="+x = 1", retries={"review": 0}))
    assert patch["review"].verdict == "rejected"
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "escalation"
    assert "review" not in patch or "retries" not in patch  # not counted as a normal fix cycle


def test_review_node_malformed_output_escalates_not_false_approval(tmp_path: Path) -> None:
    # Every response is unparseable prose -> structured_call exhausts repairs and raises.
    reviewer, _ = _reviewer([ChatResponse(content="not json") for _ in range(5)])
    node = make_review_node(reviewer, GraphSettings())
    patch = node(_state(tmp_path, diff_summary="+x = 1"))
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "escalation"
    assert "review" not in patch  # no fabricated Review on failure


# ---------------------------------------------------------------------------
# Final-accept gate (carried over from review_stub's contract)
# ---------------------------------------------------------------------------
def test_review_node_final_accept_gate_in_manual_autonomy(tmp_path: Path) -> None:
    reviewer, _ = _reviewer([_emit({"verdict": "approved", "issues": [], "summary": "good"})])
    node = make_review_node(reviewer, GraphSettings())
    patch = node(_state(tmp_path, diff_summary="+x = 1", autonomy_level="manual"))
    assert patch["review"].verdict == "approved"
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "final_accept"


def test_review_node_no_final_accept_gate_outside_manual(tmp_path: Path) -> None:
    for autonomy in ("auto", "semi"):
        reviewer, _ = _reviewer([_emit({"verdict": "approved", "issues": [], "summary": "good"})])
        node = make_review_node(reviewer, GraphSettings())
        patch = node(_state(tmp_path, diff_summary="+x = 1", autonomy_level=autonomy))
        assert patch["hitl_request"] is None


# ---------------------------------------------------------------------------
# Isolation (ADR-0006, the signature invariant): coder_scratch is never seen.
# ---------------------------------------------------------------------------
def test_review_node_never_sees_coder_scratch(tmp_path: Path) -> None:
    from langchain_core.messages import HumanMessage

    secret = "SECRET_CODER_REASONING_MUST_NOT_LEAK_TO_REVIEWER"
    reviewer, provider = _reviewer([_emit({"verdict": "approved", "issues": [], "summary": "ok"})])
    node = make_review_node(reviewer, GraphSettings())
    node(
        _state(
            tmp_path,
            diff_summary="+x = 1",
            coder_scratch=[HumanMessage(content=secret)],
            plan=Plan(summary="a plan", tasks=[]),
            verify_result=VerifyResult(passed=True, summary="ok"),
        )
    )
    all_content = "\n".join(m.content or "" for call in provider.calls for m in call.messages)
    assert secret not in all_content


def test_review_node_input_is_built_from_plan_diff_verify_only(tmp_path: Path) -> None:
    """Confirms the node reads state["diff_summary"] (already truncated by coder/
    finalize) rather than recomputing anything — the isolated 3-source contract."""
    reviewer, provider = _reviewer([_emit({"verdict": "approved", "issues": [], "summary": "ok"})])
    node = make_review_node(reviewer, GraphSettings())
    node(
        _state(
            tmp_path,
            diff_summary="+added a distinctive line xyz123",
            plan=Plan(summary="distinctive plan summary", tasks=[]),
            verify_result=VerifyResult(passed=True, summary="distinctive verify summary"),
        )
    )
    rendered = provider.calls[0].messages[1].content
    assert "distinctive plan summary" in rendered
    assert "distinctive line xyz123" in rendered
    assert "distinctive verify summary" in rendered
