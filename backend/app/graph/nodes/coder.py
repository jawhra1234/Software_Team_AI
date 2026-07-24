"""Coder node (Task 2.3, 2.8, ARCHITECTURE.md §4.3).

Wraps the Phase-1 :class:`app.agents.coder.Coder` ReAct loop. Two modes:

* **Task mode** — a pending :class:`~app.graph.state.Task` from the plan is
  selected, run, and its status/attempts updated in place.
* **Fix mode** — reached when all tasks are already "done" but verify failed
  or review requested changes (i.e. there is no *pending* task to select). An
  ad hoc :class:`CoderTask` is synthesized directly from the verify/review
  feedback, since ``ARCHITECTURE.md §4.3`` calls for fixing "the specific
  verify_result/review issues, not the whole task."

Either way: commits outstanding changes at the boundary, derives
``changed_files``/``diff_summary`` from git, and escalates on budget/no-progress
exhaustion.

Command approval (Task 2.8): when autonomy requires it, the tool-authorization
pipeline's approval hook is wired to a direct ``interrupt()`` call inside the
coder's tool loop. Known LangGraph characteristic: because this interrupt lives
*inside* a multi-step node, resuming re-executes the node from the top — prior
steps in this same task attempt (including their LLM calls) replay before
reaching the same interrupt point, which then returns its cached answer
immediately. This is inherent to in-node interrupts (not a bug); read tools are
idempotent so this is safe, and the coder's low temperature limits divergence
on replay. See the Phase 2 completion report for the full discussion.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langgraph.types import interrupt

from app.agents.coder import Coder, CoderTask, workspace_signature
from app.core.clock import now_iso
from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.retrieval import RetrievalCapture
from app.graph.state import AgentState, ErrorRecord, FileRef, HITLRequest, Task
from app.providers.base import LLMProvider
from app.tools.authorization import truncate_output
from app.tools.base import ToolContext, ToolRegistry
from app.tools.git import Git
from app.tools.sandbox import Sandbox
from app.workspace.lifecycle import Workspace

if TYPE_CHECKING:
    from app.rag.retriever import Retriever

log = get_logger("graph.nodes.coder")

_STATUS_MAP: dict[str, Literal["added", "modified", "deleted", "unchanged"]] = {
    "A": "added",
    "D": "deleted",
}


def select_next_task(tasks: list[Task]) -> Task | None:
    """Pick the next pending task whose dependencies are all done."""
    done_ids = {t.id for t in tasks if t.status == "done"}
    for task in tasks:
        if task.status == "pending" and set(task.depends_on) <= done_ids:
            return task
    return None


def build_fix_task(state: AgentState) -> CoderTask | None:
    """Synthesize an ad hoc fix task from verify/review feedback (fix mode)."""
    verify_result = state.get("verify_result")
    if verify_result is not None and not verify_result.passed:
        failing = [c for c in verify_result.checks if not c.passed]
        details = "\n".join(
            f"- {c.name} ({c.cmd}) exit={c.exit_code}: {c.stderr_tail or c.stdout_tail}"
            for c in failing
        )
        return CoderTask(
            description=f"Fix the failing verification checks.\n{verify_result.summary}\n{details}",
            acceptance_criteria=["all verify checks pass"],
        )
    review = state.get("review")
    if review is not None and review.verdict == "changes_requested":
        # Targeted hand-off (Task 4.4, ADR-0006): only blocker/major issues are
        # actionable findings — minor/nit are advisory only (still visible in
        # `review.issues`/`summary` in state for tracing) and must not be handed
        # to the coder as work items, so a fix cycle can't be triggered by nits.
        blocking = [i for i in review.issues if i.severity in ("blocker", "major")]
        details = "\n".join(
            f"- [{i.severity}] {i.description}" + (f" ({i.file})" if i.file else "")
            for i in blocking
        )
        criteria = [i.description for i in blocking] or ["address review feedback"]
        return CoderTask(
            description=f"Address the reviewer's requested changes.\n{review.summary}\n{details}",
            acceptance_criteria=criteria,
        )
    return None


def _to_coder_task(task: Task) -> CoderTask:
    return CoderTask(
        description=f"{task.title}\n\n{task.description}",
        acceptance_criteria=task.acceptance_criteria,
        target_paths=task.target_paths,
    )


def _file_refs_since(git: Git, base_commit: str) -> list[FileRef]:
    if git.current_commit() == base_commit:
        return []
    return [
        FileRef(path=path, status=_STATUS_MAP.get(letter, "modified"))
        for letter, path in git.diff_name_status(base_commit)
    ]


def make_coder_node(
    provider: LLMProvider,
    registry: ToolRegistry,
    settings: Settings,
    sandbox: Sandbox,
    retriever: Retriever | None = None,
) -> Any:
    def _node(state: AgentState) -> dict[str, Any]:
        plan = state.get("plan")
        if plan is None or not plan.tasks:
            return _escalate("coder invoked with no plan/tasks", origin="plan")

        task_id = state.get("current_task_id")
        task = next((t for t in plan.tasks if t.id == task_id), None) if task_id else None
        if task is None:
            task = select_next_task(plan.tasks)

        fix_task: CoderTask | None = None
        if task is None:
            fix_task = build_fix_task(state)
            if fix_task is None:
                # Nothing pending and no failure to fix — routing should have gone elsewhere.
                return {"current_task_id": None, "hitl_request": None}

        workspace_path = Path(state["workspace_path"])
        workspace = Workspace(
            project_id=state["project_id"],
            path=workspace_path,
            git=Git(workspace_path),
            base_commit=state.get("base_commit", ""),
            work_branch=state.get("work_branch", ""),
        )
        ctx = ToolContext(
            workspace_path=workspace_path,
            run_id=state["run_id"],
            sandbox=sandbox,
            workspace=workspace,
            retriever=retriever,
            project_id=state["project_id"],
        )
        autonomy = state["autonomy_level"]
        capture = RetrievalCapture()

        def approve(tool_name: str, args: dict[str, Any]) -> bool:
            request = HITLRequest(
                kind="command_approval",
                context=f"Approve running: {args.get('command', tool_name)}",
                options=["approve", "deny"],
            )
            answer = interrupt(request.model_dump())
            return str(answer).strip().lower() in ("approve", "yes", "true")

        before_sig = workspace_signature(workspace_path)
        coder = Coder(
            provider, registry, settings, autonomy=autonomy,
            approve=approve if autonomy != "auto" else None,
            on_tool_result=capture.observe,
        )

        coder_task = _to_coder_task(task) if task is not None else fix_task
        assert coder_task is not None  # guaranteed by the task-or-fix_task check above
        if task is not None:
            task.status = "in_progress"
        outcome = coder.run_task(coder_task, ctx)
        if task is not None:
            task.attempts += 1

        if outcome.status == "completed":
            after_sig = workspace_signature(workspace_path)
            if fix_task is not None and after_sig == before_sig:
                # Fix mode made zero changes — don't burn a verify/review retry on a no-op.
                return _escalate("fix attempt made no changes to the workspace", origin="coder")

            if task is not None:
                task.status = "done"
            patch = _commit_and_diff(workspace, state, label=task.id if task else "fix")
            patch["plan"] = plan
            patch["retrieved_context"] = capture.chunks
            remaining = select_next_task(plan.tasks)
            patch["current_task_id"] = remaining.id if remaining else None
            patch["hitl_request"] = None
            return patch

        if task is not None:
            task.status = "failed"
        patch = _commit_and_diff(workspace, state, label=task.id if task else "fix")
        patch["plan"] = plan
        patch["retrieved_context"] = capture.chunks
        patch["errors"] = [
            ErrorRecord(node="coder", kind=outcome.status, message=outcome.summary, ts=now_iso())
        ]
        label = task.id if task is not None else "fix-attempt"
        patch.update(_escalate(f"task '{label}': {outcome.summary}", origin="coder"))
        return patch

    return _node


def _commit_and_diff(workspace: Workspace, state: AgentState, *, label: str) -> dict[str, Any]:
    if workspace.git.has_changes():
        workspace.commit_task(f"agent: progress on {label}")
    base = state.get("base_commit", "")
    changed = _file_refs_since(workspace.git, base) if base else []
    diff_text = workspace.diff_since_base() if base else workspace.status()
    return {
        "changed_files": changed,
        "diff_summary": truncate_output(diff_text, 4000),
    }


def _escalate(reason: str, *, origin: str) -> dict[str, Any]:
    log.warning("coder_escalate", reason=reason, origin=origin)
    return {
        "hitl_request": HITLRequest(
            kind="escalation",
            context=reason,
            options=["retry", "accept", "abort"],
            payload={"origin_node": origin},
        ),
    }
