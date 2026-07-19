"""Plan node (Task 2.2 graph wiring, ARCHITECTURE.md §4.1).

Wraps :class:`app.agents.planner.Planner`. If the drafted plan has blocking
open questions and autonomy requires it, pauses via a direct ``interrupt()``
for clarification, then re-plans incorporating the human's answers. Malformed
structured output (repair-retry already exhausted inside ``structured_call``)
or an empty task list is treated as a planning failure and escalated rather
than left to crash the graph.

Known LangGraph characteristic (same class as ``graph/nodes/coder.py``'s
command-approval note): resuming an in-node ``interrupt()`` replays the node
from the top. The *draft* ``create_plan`` call here (the one that produces
``open_questions``) therefore runs once before the original pause and once
again during replay, before ``interrupt()`` returns its cached answer — a
grounding-only, read-only call, so replay is safe (no side effects to double
up); only the LLM call itself repeats.

Retrieval (Task 3.12): a fresh :class:`RetrievalCapture` per node invocation
collects any ``retrieve`` tool hits the planner made, surfaced as the
(ephemeral, overwritten-per-step) ``retrieved_context`` in the patch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.types import interrupt

from app.agents.planner import Planner
from app.core.clock import now_iso
from app.core.config import PlannerSettings
from app.core.errors import StructuredOutputError
from app.core.logging import get_logger
from app.graph.planning_context import build_planner_context
from app.graph.retrieval import RetrievalCapture
from app.graph.state import AgentState, ErrorRecord, HITLRequest, Plan
from app.tools.base import ToolContext

if TYPE_CHECKING:
    from app.memory.episodic import EpisodicMemory
    from app.memory.long_term import LongTermMemory
    from app.rag.retriever import Retriever

log = get_logger("graph.nodes.plan")

NodeFn = Any  # Callable[[AgentState], dict[str, Any]] — see graph/instrument.py for the alias


def make_plan_node(
    planner: Planner,
    retriever: Retriever | None = None,
    long_term: LongTermMemory | None = None,
    episodic: EpisodicMemory | None = None,
    *,
    settings: PlannerSettings | None = None,
) -> NodeFn:
    planner_settings = settings or PlannerSettings()

    def _node(state: AgentState) -> dict[str, Any]:
        ctx = ToolContext(
            workspace_path=Path(state["workspace_path"]),
            run_id=state["run_id"],
            retriever=retriever,
            project_id=state["project_id"],
        )
        prior_plan = state.get("plan")
        clarifications = list(state.get("clarification_answers", []))
        capture = RetrievalCapture()

        # Build the memory block once (best-effort inside), then reuse it for both
        # the draft and any post-clarification re-plan — avoids double DB hits and
        # double tokens, and is read-only so the interrupt-replay note below holds.
        grounding_context = build_planner_context(
            state["user_request"],
            state["project_id"],
            long_term=long_term,
            episodic=episodic,
            settings=planner_settings,
        )

        try:
            plan = planner.create_plan(
                user_request=state["user_request"],
                ctx=ctx,
                prior_plan=prior_plan,
                clarification_answers=clarifications or None,
                on_tool_result=capture.observe,
                grounding_context=grounding_context,
            )
        except StructuredOutputError as exc:
            return _plan_failed(str(exc))

        if plan.open_questions and state["autonomy_level"] != "auto":
            hitl = HITLRequest(
                kind="clarification",
                context=plan.summary,
                options=[],
                payload={"open_questions": plan.open_questions, "assumptions": plan.assumptions},
            )
            raw_answer = interrupt(hitl.model_dump())
            new_answers = _coerce_answers(raw_answer)
            try:
                plan = planner.create_plan(
                    user_request=state["user_request"],
                    ctx=ctx,
                    prior_plan=prior_plan,
                    clarification_answers=[*clarifications, *new_answers],
                    on_tool_result=capture.observe,
                    grounding_context=grounding_context,
                )
            except StructuredOutputError as exc:
                return _plan_failed(str(exc))
            return _plan_ok(plan, state, capture, extra_answers=new_answers)

        if not plan.tasks:
            return _plan_failed("planner produced an empty task list")

        return _plan_ok(plan, state, capture)

    return _node


def _plan_ok(
    plan: Plan,
    state: AgentState,
    capture: RetrievalCapture,
    *,
    extra_answers: list[str] | None = None,
) -> dict[str, Any]:
    """Build the success patch, requesting plan_approval unless autonomy is 'auto'.

    Routing (``route_after_plan``) makes its decision purely from
    ``hitl_request`` presence — same discipline as every other node — so this
    node, not the router, decides whether the approval gate is live.
    """
    patch: dict[str, Any] = {
        "plan": plan,
        "needs_clarification": False,
        "status": "planning",
        "retrieved_context": capture.chunks,
    }
    if extra_answers:
        patch["clarification_answers"] = extra_answers

    if state["autonomy_level"] != "auto":
        patch["hitl_request"] = HITLRequest(
            kind="plan_approval",
            context=plan.summary,
            options=["approve", "revise", "abort"],
        )
    else:
        patch["hitl_request"] = None
    return patch


def _plan_failed(message: str) -> dict[str, Any]:
    log.warning("plan_failed", error=message)
    return {
        "needs_clarification": False,
        "hitl_request": HITLRequest(
            kind="escalation",
            context=f"Planning failed: {message}",
            options=["retry", "abort"],
            payload={"origin_node": "plan"},
        ),
        "errors": [ErrorRecord(node="plan", kind="planning_failed", message=message, ts=now_iso())],
    }


def _coerce_answers(raw: Any) -> list[str]:
    if isinstance(raw, dict) and isinstance(raw.get("answers"), list):
        return [str(a) for a in raw["answers"]]
    if isinstance(raw, list):
        return [str(a) for a in raw]
    return [str(raw)]
