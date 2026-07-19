"""Finalize node (Task 2.6, 3.11).

Terminal node. Produces the final diff summary from git, normalizes the run's
terminal status (a prior ``human_gate`` abort may have already set
"cancelled"/"failed" — this node preserves that; otherwise reaching finalize
at all implies success), and — as of Phase 3 — writes a real episodic-memory
record (best-effort) when an :class:`EpisodicMemory` is wired in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger
from app.graph.state import AgentState, RunStatus
from app.memory.episodic import RunRecord
from app.tools.authorization import truncate_output
from app.tools.git import Git

if TYPE_CHECKING:
    from app.memory.episodic import EpisodicMemory

log = get_logger("graph.nodes.finalize")

_DIFF_TAIL_CHARS = 8000


def make_finalize_node(episodic: EpisodicMemory | None = None) -> Any:
    def _node(state: AgentState) -> dict[str, Any]:
        workspace_path = Path(state["workspace_path"])
        git = Git(workspace_path)
        base = state.get("base_commit")
        needs_diff = bool(base) and git.current_commit() != base
        diff_text = git.diff(f"{base}..HEAD") if needs_diff else "(no changes)"

        status = _final_status(state)
        _record_episode(episodic, state, status)

        return {
            "status": status,
            "diff_summary": truncate_output(diff_text, _DIFF_TAIL_CHARS),
            "hitl_request": None,
        }

    return _node


def _final_status(state: AgentState) -> RunStatus:
    current = state.get("status")
    if current in ("cancelled", "failed"):
        return current
    return "succeeded"


def _record_episode(episodic: EpisodicMemory | None, state: AgentState, status: RunStatus) -> None:
    plan = state.get("plan")
    tasks = plan.tasks if plan is not None else []
    record = RunRecord(
        run_id=state["run_id"],
        project_id=state["project_id"],
        status=status,
        summary=plan.summary if plan is not None else "",
        tasks_total=len(tasks),
        tasks_done=sum(1 for t in tasks if t.status == "done"),
    )
    if episodic is not None:
        episodic.record(record)  # best-effort; never raises
    log.info("episodic_recorded", run_id=record.run_id, status=status, wired=episodic is not None)
