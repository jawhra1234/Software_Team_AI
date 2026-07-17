"""Coder ReAct loop (Task 1.8).

A single agentic loop (not yet a graph node) that grounds in the workspace via
tools, edits files, runs commands in the sandbox, and self-corrects until the
task's acceptance criteria are met (``finish_task``) or a budget/loop guard
trips. Every tool call flows through the authorization pipeline (Task 1.3).

Tool-call extraction is tolerant: it uses native ``tool_calls`` when present and
falls back to parsing a JSON tool-call envelope from message content (some local
models emit calls as text) — the same robustness class as ``structured_call``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.agents.budget import BudgetTracker
from app.agents.toolcalls import extract_tool_calls
from app.core.config import Settings
from app.core.logging import get_logger
from app.providers.base import ChatMessage, LLMProvider
from app.tools.authorization import ApprovalHook, AuthorizationPolicy, execute_tool
from app.tools.base import ToolContext, ToolRegistry
from app.tools.control import FINISH_TASK

log = get_logger("agents.coder")

_IGNORE_DIRS = frozenset({".git", "__pycache__", ".venv", "venv", "node_modules"})

CoderStatus = Literal["completed", "failed", "budget_exceeded", "no_progress"]


@dataclass
class CoderTask:
    """A single unit of work for the coder (Phase 2 maps graph Tasks onto this)."""

    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    target_paths: list[str] = field(default_factory=list)


@dataclass
class CoderOutcome:
    status: CoderStatus
    summary: str
    steps: int


_SYSTEM_PROMPT = """\
You are an autonomous software engineer working inside a sandboxed git workspace.
The workspace directory is your current working directory.

Rules:
- Ground before acting: read existing files with read_file / list_dir / search_code
  before assuming anything. Never invent file contents.
- Make the smallest change that satisfies the acceptance criteria.
- Use write_file / edit_file to change files, and run_command to run tests/builds.
- Verify your work by running the tests with run_command (e.g. `python -m pytest -q`).
- When the acceptance criteria are met and tests pass, call finish_task with a summary.
- Always act by calling a tool. Do not answer in prose.
"""


class Coder:
    """Runs a :class:`CoderTask` to completion via a tool-calling loop."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        settings: Settings,
        *,
        autonomy: Literal["manual", "semi", "auto"] = "auto",
        approve: ApprovalHook | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._settings = settings
        self._policy = AuthorizationPolicy.from_settings(settings, autonomy=autonomy)
        #: Command-approval hook (Task 2.8); wired to a LangGraph interrupt by
        #: the graph's coder node. None locally reproduces Phase-1 behavior.
        self._approve = approve

    def run_task(self, task: CoderTask, ctx: ToolContext) -> CoderOutcome:
        budget = BudgetTracker.from_settings(self._settings.coder)
        budget.start()
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=_render_task(task)),
        ]
        tool_specs = self._registry.specs()

        while True:
            reason = budget.exceeded_reason()
            if reason is not None:
                log.warning("coder_budget_exceeded", reason=reason, steps=budget.steps)
                return CoderOutcome("budget_exceeded", reason, budget.steps)
            budget.tick_step()

            response = self._provider.chat(messages, tools=tool_specs)
            if response.usage is not None:
                budget.add_tokens(response.usage.output_tokens or 0)

            calls = extract_tool_calls(response.content, response.tool_calls, self._registry)
            messages.append(
                ChatMessage(role="assistant", content=response.content, tool_calls=calls or None)
            )

            if not calls:
                messages.append(
                    ChatMessage(
                        role="user",
                        content="You did not call a tool. Call a tool now, or finish_task if done.",
                    )
                )
                budget.record_progress("__no_tool__")
                if budget.no_progress_reason() is not None:
                    return CoderOutcome("no_progress", "model stopped calling tools", budget.steps)
                continue

            for call in calls:
                if call.name == FINISH_TASK:
                    summary = str(call.arguments.get("summary", "task finished"))
                    log.info("coder_finished", steps=budget.steps)
                    return CoderOutcome("completed", summary, budget.steps)
                result = execute_tool(
                    self._registry, call.name, call.arguments, ctx, self._policy,
                    approve=self._approve,
                )
                observation = (
                    result.output if result.ok else f"ERROR: {result.error}\n{result.output}"
                )
                messages.append(
                    ChatMessage(
                        role="tool", name=call.name, tool_call_id=call.id, content=observation
                    )
                )

            budget.record_progress(workspace_signature(ctx.workspace_path))
            if budget.no_progress_reason() is not None:
                return CoderOutcome("no_progress", "repeated steps without changes", budget.steps)


def _render_task(task: CoderTask) -> str:
    parts = [f"Task: {task.description}"]
    if task.acceptance_criteria:
        parts.append("Acceptance criteria:")
        parts.extend(f"- {c}" for c in task.acceptance_criteria)
    if task.target_paths:
        parts.append("Relevant paths: " + ", ".join(task.target_paths))
    return "\n".join(parts)


def workspace_signature(root: Path) -> str:
    """Content hash of workspace files (for no-progress detection)."""
    digest = hashlib.sha1()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in _IGNORE_DIRS for part in rel.parts) or not path.is_file():
            continue
        digest.update(rel.as_posix().encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()
