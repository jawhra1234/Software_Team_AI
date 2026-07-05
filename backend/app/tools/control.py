"""Control tools (Task 1.x) — signals the coder emits to end a task."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolContext, ToolResult

FINISH_TASK = "finish_task"


class FinishTaskArgs(BaseModel):
    summary: str = Field(description="Short summary of what was accomplished.")


class FinishTask(Tool[FinishTaskArgs]):
    name = FINISH_TASK
    description = "Call this when the task's acceptance criteria are met, with a summary."
    args_schema = FinishTaskArgs

    def run(self, args: FinishTaskArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.success(output=args.summary, finished=True)
