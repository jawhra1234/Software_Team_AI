"""LangGraph state schema (Task 2.1, ARCHITECTURE.md §5).

Defines the value objects and the ``AgentState`` TypedDict that drives the
6-node graph. State holds control flow and small structured artifacts only —
**never file contents or un-truncated command output** (``ARCHITECTURE.md``
tenet #5 / ADR-0002): file changes are tracked by :class:`FileRef` (path +
status + blob sha), and verify output is truncated to head/tail tails.

Reducer rules: ``changed_files`` merges by path (latest write wins);
``coder_scratch``/``errors``/``node_history``/``clarification_answers`` append;
``retries`` sums per key; every other field is last-write-wins (the LangGraph
default for un-annotated keys).
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from app.graph.reducers import merge_by_path, merge_counts

AutonomyLevel = Literal["manual", "semi", "auto"]
RunStatus = Literal["planning", "running", "paused", "succeeded", "failed", "cancelled"]
HITLKind = Literal[
    "plan_approval", "clarification", "escalation", "final_accept", "command_approval"
]


def _coerce_str_list(value: Any) -> Any:
    """Coerce each item of a list to a string.

    Local models frequently emit list-of-objects (e.g. an acceptance criterion
    as ``{"type": "file_exists", "path": "calc.py"}``) where the schema asks for
    list-of-strings. These fields are free-text guidance the coder reads, not
    machine-parsed, so coercing non-strings to a compact JSON/string is safe and
    makes structured planning robust to that common quirk. Non-list values are
    passed through untouched so normal validation still applies.
    """
    if not isinstance(value, list):
        return value
    coerced: list[str] = []
    for item in value:
        if isinstance(item, str):
            coerced.append(item)
        elif isinstance(item, (dict, list)):
            coerced.append(json.dumps(item, ensure_ascii=False))
        else:
            coerced.append(str(item))
    return coerced


#: A list-of-strings field tolerant of a model emitting list-of-objects.
StrList = Annotated[list[str], BeforeValidator(_coerce_str_list)]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
class Budget(BaseModel):
    """Run-wide budget (distinct from the per-task budget in ``CoderSettings``)."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None
    max_steps: int
    max_wall_clock_s: float
    tokens_used: int = 0
    steps_used: int = 0
    started_at: str


class FileRef(BaseModel):
    """A pointer to a changed file — never the contents (ADR-0002)."""

    model_config = ConfigDict(extra="forbid")

    path: str
    status: Literal["added", "modified", "deleted", "unchanged"]
    blob_sha: str | None = None


class Task(BaseModel):
    """A single unit of planned work."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Short stable id, e.g. 'task-1', 'task-2'.")
    title: str = Field(description="Short imperative title, e.g. 'Add add() to calc.py'.")
    description: str = Field(description="What to do and why, specific enough to act on.")
    kind: Literal["create", "modify", "test", "docs", "fix"]
    target_paths: StrList = Field(
        default_factory=list, description="Workspace-relative files this task will touch."
    )
    acceptance_criteria: StrList = Field(
        default_factory=list, description="Concrete, checkable conditions for 'done'."
    )
    depends_on: StrList = Field(
        default_factory=list, description="Ids of tasks that must complete first."
    )
    status: Literal["pending", "in_progress", "done", "failed", "skipped"] = "pending"
    attempts: int = 0


class Plan(BaseModel):
    """The planner's spec + architecture notes + ordered task list."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    summary: str = Field(description="One or two sentences describing what will be built.")
    functional_requirements: StrList = Field(
        default_factory=list, description="What the system must do, derived from the request."
    )
    non_functional: StrList = Field(
        default_factory=list, description="Quality constraints, e.g. performance, security."
    )
    constraints: StrList = Field(
        default_factory=list, description="Hard limits: stack, environment, must-not-do."
    )
    assumptions: StrList = Field(
        default_factory=list,
        description="Reasonable defaults you chose instead of asking — record them here.",
    )
    open_questions: StrList = Field(
        default_factory=list,
        description=(
            "ONLY truly blocking questions with no reasonable default "
            "(e.g. contradictory requirements). Leave empty whenever possible."
        ),
    )
    architecture_notes: str = Field(
        default="", description="Key design/architecture decisions, grounded in the existing repo."
    )
    tech_stack: dict[str, str] = Field(
        default_factory=dict, description="Technology choices, e.g. {'language': 'python'}."
    )
    tasks: list[Task] = Field(
        default_factory=list, description="Ordered, small, independently-verifiable tasks."
    )


class CheckResult(BaseModel):
    """Outcome of one verify check (canonical shape; ``app.verify.runner`` imports this)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cmd: str
    passed: bool
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""


class VerifyResult(BaseModel):
    """Aggregate verify outcome (canonical shape; ADR-0005)."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)
    summary: str = ""


class ReviewIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["blocker", "major", "minor", "nit"]
    file: str | None = None
    line: int | None = None
    description: str
    suggestion: str | None = None


class Review(BaseModel):
    """Reviewer verdict (Phase 2 ships a rule-based stub; ADR-0006 arrives Phase 4)."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["approved", "changes_requested", "rejected"]
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str = ""


class RetrievedChunk(BaseModel):
    """Ephemeral RAG hit (Phase 3); never persisted long-term."""

    model_config = ConfigDict(extra="forbid")

    path: str
    symbol: str | None = None
    score: float
    content: str


class HITLRequest(BaseModel):
    """Payload for a human-in-the-loop pause (ADR-0009).

    ``kind`` extends ``ARCHITECTURE.md``'s four-value enum with
    ``"clarification"`` — the plan node's own direct interrupt for blocking
    open questions, distinct from ``plan_approval`` (approving an already-drafted
    plan). See the Phase 2 completion report for the rationale.
    """

    model_config = ConfigDict(extra="forbid")

    kind: HITLKind
    context: str
    options: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class HITLResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str
    edits: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


class ErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str
    kind: str
    message: str
    ts: str


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # identity
    run_id: str
    project_id: str
    thread_id: str

    # request / planning
    user_request: str
    intent: NotRequired[str]
    needs_clarification: NotRequired[bool]
    clarification_answers: Annotated[list[str], operator.add]
    plan: NotRequired[Plan]

    # execution
    current_task_id: NotRequired[str | None]
    workspace_path: str
    base_commit: NotRequired[str]
    work_branch: NotRequired[str]
    changed_files: Annotated[list[FileRef], merge_by_path]
    diff_summary: NotRequired[str]
    coder_scratch: Annotated[list[AnyMessage], add_messages]

    # quality gates
    verify_result: NotRequired[VerifyResult]
    review: NotRequired[Review]
    retrieved_context: NotRequired[list[RetrievedChunk]]

    # control / HITL / observability
    autonomy_level: AutonomyLevel
    hitl_request: NotRequired[HITLRequest | None]
    hitl_response: NotRequired[HITLResponse | None]
    budget: Budget
    retries: Annotated[dict[str, int], merge_counts]
    errors: Annotated[list[ErrorRecord], operator.add]
    node_history: Annotated[list[str], operator.add]
    status: RunStatus


def new_run_state(
    *,
    run_id: str,
    project_id: str,
    user_request: str,
    workspace_path: str,
    autonomy_level: AutonomyLevel,
    max_tokens: int | None,
    max_steps: int,
    max_wall_clock_s: float,
    started_at: str,
) -> AgentState:
    """Construct the initial state for a fresh run (all reducer fields empty)."""
    return AgentState(
        run_id=run_id,
        project_id=project_id,
        thread_id=run_id,
        user_request=user_request,
        clarification_answers=[],
        workspace_path=workspace_path,
        changed_files=[],
        coder_scratch=[],
        autonomy_level=autonomy_level,
        budget=Budget(
            max_tokens=max_tokens,
            max_steps=max_steps,
            max_wall_clock_s=max_wall_clock_s,
            started_at=started_at,
        ),
        retries={},
        errors=[],
        node_history=[],
        status="planning",
    )
