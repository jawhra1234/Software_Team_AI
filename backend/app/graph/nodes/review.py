"""Review node (Task 4.2/4.3, ADR-0006).

Wraps :class:`app.agents.reviewer.Reviewer`, replacing the Phase-2
``review_stub`` with a real fresh-context adversarial reviewer — **same node
contract** (reads the diff + plan + verify result, writes ``review``), same
final-accept gate, cycle cap, and escalation the stub already implemented.

Isolation (ADR-0006): the reviewer's input is built from exactly three
sources — ``state["plan"]``, ``state["diff_summary"]``, and
``state["verify_result"]`` — and nothing else. In particular
``state["coder_scratch"]`` is never read here; that is the whole point of a
*fresh-context* reviewer.

Severity gating (Task 4.4/4.5): the model's own ``verdict`` is not trusted
blindly — a small local model can mislabel it. The **effective** verdict used
for routing is derived deterministically from the issues' severities: any
``blocker``/``major`` forces ``changes_requested`` regardless of what the model
claimed; otherwise (no blocker/major) it's ``approved``. ``rejected`` is
respected as-is (a judgment call no deterministic rule can make) and always
escalates rather than looping. A malformed/unparseable review — the repair-retry
in ``structured_call`` already exhausted — also escalates rather than risking a
false approval.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.agents.reviewer import Reviewer
from app.core.clock import now_iso
from app.core.config import GraphSettings
from app.core.errors import StructuredOutputError
from app.core.logging import get_logger
from app.graph.retrieval import RetrievalCapture
from app.graph.state import AgentState, ErrorRecord, HITLRequest, Review
from app.tools.base import ToolContext

if TYPE_CHECKING:
    from app.rag.retriever import Retriever

log = get_logger("graph.nodes.review")

NodeFn = Any  # Callable[[AgentState], dict[str, Any]] — see graph/instrument.py for the alias

_BLOCKING_SEVERITIES = frozenset({"blocker", "major"})


def _effective_verdict(review: Review) -> str:
    """Derive the verdict routing actually uses (Task 4.5) — see module docstring."""
    if review.verdict == "rejected":
        return "rejected"
    if any(issue.severity in _BLOCKING_SEVERITIES for issue in review.issues):
        return "changes_requested"
    return "approved"


def make_review_node(
    reviewer: Reviewer, graph_settings: GraphSettings, retriever: Retriever | None = None
) -> NodeFn:
    def _node(state: AgentState) -> dict[str, Any]:
        ctx = ToolContext(
            workspace_path=Path(state["workspace_path"]),
            run_id=state["run_id"],
            retriever=retriever,
            project_id=state["project_id"],
        )
        capture = RetrievalCapture()

        try:
            review = reviewer.review_change(
                plan=state.get("plan"),
                diff=state.get("diff_summary", ""),
                verify_result=state.get("verify_result"),
                ctx=ctx,
                on_tool_result=capture.observe,
            )
        except StructuredOutputError as exc:
            return _review_failed(str(exc))

        effective = _effective_verdict(review)
        if effective != review.verdict:
            log.warning(
                "review_verdict_overridden", model_verdict=review.verdict, effective=effective
            )
            review = review.model_copy(update={"verdict": effective})

        patch: dict[str, Any] = {
            "review": review,
            "retrieved_context": capture.chunks,
            "hitl_request": None,
        }

        if effective == "approved":
            if state["autonomy_level"] == "manual":
                patch["hitl_request"] = HITLRequest(
                    kind="final_accept",
                    context=review.summary,
                    options=["accept", "request_changes"],
                )
            return patch

        if effective == "rejected":
            patch["hitl_request"] = HITLRequest(
                kind="escalation",
                context=f"Reviewer rejected the change: {review.summary}",
                options=["retry", "accept", "abort"],
                payload={"origin_node": "coder"},
            )
            return patch

        # changes_requested: bounded fix-loop cycle count, same contract as review_stub.
        current = state.get("retries", {}).get("review", 0)
        patch["retries"] = {"review": 1}
        if current + 1 >= graph_settings.max_review_cycles:
            patch["hitl_request"] = HITLRequest(
                kind="escalation",
                context=f"Review requested changes after {current + 1} cycle(s): {review.summary}",
                options=["retry", "accept", "abort"],
                payload={"origin_node": "coder"},
            )
        return patch

    return _node


def _review_failed(message: str) -> dict[str, Any]:
    # Mirrors plan.py's `_plan_failed`: no synthetic Review is fabricated — a
    # malformed reviewer output must escalate, never be dressed up as real
    # findings. `hitl_request` alone is enough for routing to detour correctly.
    log.warning("review_failed", error=message)
    return {
        "hitl_request": HITLRequest(
            kind="escalation",
            context=f"Review failed: {message}",
            options=["retry", "abort"],
            payload={"origin_node": "coder"},
        ),
        "errors": [ErrorRecord(node="review", kind="review_failed", message=message, ts=now_iso())],
    }
