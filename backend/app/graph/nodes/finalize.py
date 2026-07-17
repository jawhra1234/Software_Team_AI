"""Finalize node (Task 2.6).

Terminal node. Produces the final diff summary from git, normalizes the run's
terminal status (a prior ``human_gate`` abort may have already set
"cancelled"/"failed" — this node preserves that; otherwise reaching finalize
at all implies success), and writes an episodic-memory stub hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.graph.state import AgentState, RunStatus
from app.tools.authorization import truncate_output
from app.tools.git import Git

log = get_logger("graph.nodes.finalize")

_DIFF_TAIL_CHARS = 8000


def make_finalize_node() -> Any:
    def _node(state: AgentState) -> dict[str, Any]:
        workspace_path = Path(state["workspace_path"])
        git = Git(workspace_path)
        base = state.get("base_commit")
        needs_diff = bool(base) and git.current_commit() != base
        diff_text = git.diff(f"{base}..HEAD") if needs_diff else "(no changes)"

        status = _final_status(state)
        _write_episodic_memory_stub(state, status)

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


def _write_episodic_memory_stub(state: AgentState, status: RunStatus) -> None:
    """Stub hook (ADR-0002 / ARCHITECTURE.md §16): Phase 3 makes this a real write."""
    log.info("episodic_memory_stub", run_id=state["run_id"], status=status)
