"""Central tool authorization pipeline (Task 1.3).

Every tool call flows through :func:`execute_tool`:

    schema-validate → path-jail → command allow/deny → approval hook
    → execute → truncate output → trace

Failures at any gate are returned as ``ToolResult(ok=False, ...)`` — data for
the agent, never an exception. The command-approval hook is the seam for the
Phase-2 HITL interrupt; in Phase 1 it is an optional callback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from app.tools.security import PathJailError, resolve_within

log = get_logger("tools.authorization")

#: Arg field names treated as filesystem paths and subjected to the path-jail.
PATH_FIELDS = frozenset({"path", "paths", "file", "dir", "directory", "src", "dst"})

#: Optional hook: return True to allow a command-approval-gated call.
ApprovalHook = Callable[[str, dict[str, Any]], bool]


@dataclass
class AuthorizationPolicy:
    """Runtime policy for the pipeline, derived from settings."""

    allow_commands: list[str]
    deny_substrings: list[str]
    output_tail_chars: int
    autonomy: Literal["manual", "semi", "auto"] = "auto"

    @classmethod
    def from_settings(
        cls, settings: Settings, *, autonomy: Literal["manual", "semi", "auto"] = "auto"
    ) -> AuthorizationPolicy:
        return cls(
            allow_commands=list(settings.sandbox.allow_commands),
            deny_substrings=list(settings.sandbox.deny_substrings),
            output_tail_chars=settings.coder.output_tail_chars,
            autonomy=autonomy,
        )


def truncate_output(text: str, max_chars: int) -> str:
    """Keep head+tail of ``text`` within ``max_chars``, marking any elision."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n...[truncated {dropped} chars]...\n{text[-half:]}"


def _check_paths(args: BaseModel, workspace_path: Path) -> str | None:
    for field, value in args.model_dump().items():
        if field not in PATH_FIELDS:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str):
                try:
                    resolve_within(workspace_path, item)
                except PathJailError as exc:
                    return str(exc)
    return None


def _check_command(command: str, policy: AuthorizationPolicy) -> str | None:
    tokens = command.strip().split()
    if not tokens:
        return "empty command"
    first = Path(tokens[0])
    if first.name not in policy.allow_commands and first.stem not in policy.allow_commands:
        return f"command '{tokens[0]}' is not in the allowlist"
    lowered = command.lower()
    for pattern in policy.deny_substrings:
        if pattern.lower() in lowered:
            return f"command contains a denied pattern: '{pattern}'"
    return None


def execute_tool(
    registry: ToolRegistry,
    name: str,
    raw_args: dict[str, Any],
    ctx: ToolContext,
    policy: AuthorizationPolicy,
    *,
    approve: ApprovalHook | None = None,
) -> ToolResult:
    """Validate, authorize, execute, truncate, and trace a single tool call."""
    try:
        tool: Tool[Any] = registry.get(name)
    except KeyError as exc:
        log.warning("tool_unknown", tool=name)
        return ToolResult.failure(str(exc))

    try:
        args = tool.args_schema.model_validate(raw_args)
    except ValidationError as exc:
        log.warning("tool_bad_args", tool=name)
        return ToolResult.failure(f"invalid arguments for '{name}': {exc}")

    path_error = _check_paths(args, ctx.workspace_path)
    if path_error is not None:
        log.warning("tool_path_rejected", tool=name, error=path_error)
        return ToolResult.failure(path_error)

    dumped = args.model_dump()
    command = dumped.get("command")
    if isinstance(command, str):
        command_error = _check_command(command, policy)
        if command_error is not None:
            log.warning("tool_command_rejected", tool=name, error=command_error)
            return ToolResult.failure(command_error)

    # ADR-0009 autonomy matrix: `manual` gates commands too (it is the strictest
    # level, not a lesser one) — the gate is live whenever autonomy != "auto".
    if (
        tool.requires_approval
        and policy.autonomy != "auto"
        and approve is not None
        and not approve(name, dumped)
    ):
        log.info("tool_approval_denied", tool=name)
        return ToolResult.failure("command approval denied by human")

    try:
        result = tool.run(args, ctx)
    except Exception as exc:
        log.warning("tool_raised", tool=name, error=str(exc))
        result = ToolResult.failure(f"tool '{name}' raised: {exc}")

    result.output = truncate_output(result.output, policy.output_tail_chars)
    log.info("tool_executed", tool=name, ok=result.ok)
    return result
