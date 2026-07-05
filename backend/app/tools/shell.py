"""run_command tool (Task 1.x) — executes a command in the sandbox.

Wraps the configured :class:`Sandbox`. ``requires_approval=True`` so it flows
through the command-approval gate under `semi` autonomy (Phase 2 HITL). A
non-zero exit or timeout is returned as a failed ``ToolResult`` — data the agent
reads and reacts to.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolContext, ToolResult


class RunCommandArgs(BaseModel):
    command: str = Field(description="Shell command to run in the workspace sandbox.")
    timeout_s: float | None = Field(default=None, description="Optional per-command timeout.")


class RunCommand(Tool[RunCommandArgs]):
    name = "run_command"
    description = "Run a shell command in the sandbox (workspace as CWD, no network)."
    args_schema = RunCommandArgs
    requires_approval = True

    def run(self, args: RunCommandArgs, ctx: ToolContext) -> ToolResult:
        if ctx.sandbox is None:
            return ToolResult.failure("no sandbox configured in tool context")
        outcome = ctx.sandbox.run(
            args.command, workspace_path=ctx.workspace_path, timeout_s=args.timeout_s
        )
        combined = outcome.stdout
        if outcome.stderr:
            combined = f"{combined}\n[stderr]\n{outcome.stderr}" if combined else outcome.stderr
        meta = {"exit_code": outcome.exit_code, "timed_out": outcome.timed_out}
        if outcome.ok:
            return ToolResult.success(output=combined, **meta)
        error = "command timed out" if outcome.timed_out else f"exit code {outcome.exit_code}"
        return ToolResult.failure(error=error, output=combined, **meta)
