"""Node instrumentation: events, node_history, run-wide budget circuit breaker (Task 2.9/2.12/2.14).

``instrument_node`` wraps every node function so these cross-cutting concerns
live in one place instead of being duplicated in each node body:

* emits ``node_start``/``node_end`` events to the configured :class:`EventSink`
* appends the node's name to ``node_history`` (loop detection + tracing)
* enforces the run-wide :class:`~app.graph.state.Budget` as a circuit breaker:
  if the run-wide step/wall-clock/token budget is already exhausted, the
  wrapped node is skipped entirely and an escalation ``HITLRequest`` is
  returned instead — this is distinct from the coder's *per-task* budget
  (``CoderSettings``/``BudgetTracker``), which still governs a single task's
  internal tool-calling loop.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.graph.events import EventSink, GraphEvent, NullEventSink
from app.graph.state import AgentState, Budget, HITLRequest

NodeFn = Callable[[AgentState], dict[str, Any]]


def _elapsed_s(started_at: str) -> float:
    started = datetime.fromisoformat(started_at)
    return (datetime.now(UTC) - started).total_seconds()


def budget_exceeded_reason(budget: Budget) -> str | None:
    if budget.steps_used >= budget.max_steps:
        return f"run step budget exhausted ({budget.steps_used}/{budget.max_steps})"
    elapsed = _elapsed_s(budget.started_at)
    if elapsed >= budget.max_wall_clock_s:
        return f"run wall-clock budget exhausted ({elapsed:.0f}s/{budget.max_wall_clock_s:.0f}s)"
    if budget.max_tokens is not None and budget.tokens_used >= budget.max_tokens:
        return f"run token budget exhausted ({budget.tokens_used}/{budget.max_tokens})"
    return None


def instrument_node(
    name: str, fn: NodeFn, sink: EventSink | None = None, *, enforce_budget: bool = True
) -> NodeFn:
    """Wrap ``fn`` with budget enforcement, event emission, and history tracking.

    ``enforce_budget=False`` exempts safety-valve nodes (``human_gate``,
    ``finalize``) from the circuit breaker — those must always be reachable
    even when the run-wide budget is exhausted, since they are how the run
    surfaces that fact to a human and closes out cleanly.
    """
    sink = sink or NullEventSink()

    def _wrapped(state: AgentState) -> dict[str, Any]:
        sink.emit(GraphEvent(kind="node_start", node=name))

        if enforce_budget:
            budget = state["budget"]
            reason = budget_exceeded_reason(budget)
            if reason is not None:
                sink.emit(
                    GraphEvent(kind="node_end", node=name, data={"skipped": True, "reason": reason})
                )
                return {
                    "node_history": [name],
                    "hitl_request": HITLRequest(
                        kind="escalation",
                        context=reason,
                        options=["accept", "abort"],
                        payload={"origin_node": "coder"},
                    ),
                }

        patch = fn(state)
        if enforce_budget:
            budget = state["budget"]
            patch.setdefault(
                "budget", budget.model_copy(update={"steps_used": budget.steps_used + 1})
            )
        patch.setdefault("node_history", [])
        patch["node_history"] = [*patch["node_history"], name]

        event_data = {k: _safe(v) for k, v in patch.items()}
        sink.emit(GraphEvent(kind="node_end", node=name, data=event_data))
        return patch

    return _wrapped


def _safe(value: Any) -> Any:
    """Best-effort JSON-friendly projection of a patch value, for event payloads."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return value
