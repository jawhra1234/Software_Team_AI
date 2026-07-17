"""Review-stub node (Task 2.5).

Deterministic placeholder — no LLM. Approves whenever the run produced changes
(the normal case, since this node only runs after verify has passed); flags
"nothing changed" as a minor issue otherwise, so both routing branches
(approved / changes_requested) are exercisable without a live model. The real
fresh-context adversarial reviewer (ADR-0006) replaces this in Phase 4 without
changing the node's contract (reads the diff + plan, writes ``review``).

Also owns the ``final_accept`` gate: when the verdict is approved and autonomy
is ``manual``, it requests final human sign-off before ``finalize``.
"""

from __future__ import annotations

from typing import Any

from app.core.config import GraphSettings
from app.graph.state import AgentState, HITLRequest, Review, ReviewIssue


def make_review_node(graph_settings: GraphSettings) -> Any:
    def _node(state: AgentState) -> dict[str, Any]:
        changed = state.get("changed_files", [])
        if changed:
            review = Review(
                verdict="approved",
                summary="stub reviewer: changes present (Phase 4 adds real adversarial review)",
            )
        else:
            review = Review(
                verdict="changes_requested",
                issues=[
                    ReviewIssue(
                        severity="minor",
                        description="no files changed since the base commit",
                    )
                ],
                summary="stub reviewer: nothing to review",
            )

        patch: dict[str, Any] = {"review": review, "hitl_request": None}

        if review.verdict == "approved":
            if state["autonomy_level"] == "manual":
                patch["hitl_request"] = HITLRequest(
                    kind="final_accept",
                    context=review.summary,
                    options=["accept", "request_changes"],
                )
            return patch

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
