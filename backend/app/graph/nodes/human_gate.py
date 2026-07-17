"""Human gate node (Task 2.7, ADR-0009).

The single multiplexed HITL pause point for ``plan_approval``, ``escalation``,
and ``final_accept`` (``clarification`` and ``command_approval`` are direct
interrupts inside the ``plan``/``coder`` nodes — see those modules). Reads the
``hitl_request`` an upstream node populated, calls ``interrupt()``, and applies
the human's decision: plan edits are merged into ``plan``; an abort decision
sets a provisional terminal ``status`` that ``finalize`` normalizes.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from app.graph.state import AgentState, HITLResponse, Plan, Task

_EDITABLE_PLAN_FIELDS = ("summary", "architecture_notes")


def make_human_gate_node() -> Any:
    def _node(state: AgentState) -> dict[str, Any]:
        request = state.get("hitl_request")
        if request is None:
            # Defensive: routing only sends us here when a request was set.
            return {"hitl_response": HITLResponse(decision="abort", note="missing hitl_request")}

        raw = interrupt(request.model_dump())
        response = _coerce_response(raw)
        patch: dict[str, Any] = {"hitl_response": response}

        if request.kind == "plan_approval":
            if response.decision == "revise" and response.edits:
                patch["plan"] = _apply_plan_edits(state.get("plan"), response.edits)
            if response.decision == "abort":
                patch["status"] = "cancelled"
        elif request.kind == "escalation" and response.decision == "abort":
            patch["status"] = "failed"

        return patch

    return _node


def _coerce_response(raw: Any) -> HITLResponse:
    if isinstance(raw, dict):
        edits = raw.get("edits")
        return HITLResponse(
            decision=str(raw.get("decision", "")),
            edits=edits if isinstance(edits, dict) else {},
            note=raw.get("note"),
        )
    return HITLResponse(decision=str(raw))


def _apply_plan_edits(plan: Plan | None, edits: dict[str, Any]) -> Plan | None:
    """Merge a conservative, whitelisted subset of human edits into ``plan``.

    Only known-safe fields are applied (tasks wholesale replacement, summary,
    architecture_notes) — arbitrary keys are ignored rather than bypassing
    ``Plan``'s schema validation.
    """
    if plan is None or not edits:
        return plan
    updates: dict[str, Any] = {}
    raw_tasks = edits.get("tasks")
    if isinstance(raw_tasks, list):
        updates["tasks"] = [Task(**t) if isinstance(t, dict) else t for t in raw_tasks]
    for field in _EDITABLE_PLAN_FIELDS:
        value = edits.get(field)
        if isinstance(value, str):
            updates[field] = value
    if not updates:
        return plan
    updates["version"] = plan.version + 1
    return plan.model_copy(update=updates)
