"""Tool abstraction layer — protocol, result, context, registry (Task 1.1).

Implements the tool contract from ``ARCHITECTURE.md §6``. Domain tools depend
only on the types here and never import LangChain; :func:`to_langchain_tool`
is the single adapter that binds a tool to the agent framework (Phase 2).

Every tool call is expected to flow through the central authorization pipeline
(:func:`app.tools.authorization.execute_tool`, Task 1.3), which validates the
raw arguments against ``args_schema`` before invoking :meth:`Tool.run`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, Field

from app.providers.base import ToolSpec

if TYPE_CHECKING:
    from app.agents.budget import BudgetTracker
    from app.core.tracing import Tracer
    from app.rag.retriever import Retriever
    from app.tools.sandbox import Sandbox
    from app.workspace.lifecycle import Workspace

TArgs = TypeVar("TArgs", bound=BaseModel)


class ToolResult(BaseModel):
    """Uniform result of a tool invocation.

    A non-zero command exit or a rejected edit is represented as ``ok=False``
    with a populated ``error`` — it is data fed back to the agent, not an
    exception.
    """

    ok: bool
    output: str = ""
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(cls, output: str = "", **meta: Any) -> ToolResult:
        return cls(ok=True, output=output, meta=meta)

    @classmethod
    def failure(cls, error: str, output: str = "", **meta: Any) -> ToolResult:
        return cls(ok=False, error=error, output=output, meta=meta)


@dataclass
class ToolContext:
    """Ambient context threaded into every tool call.

    Fields beyond ``workspace_path``/``run_id`` are populated as the owning
    subsystems come online (sandbox in 1.2, workspace/git in 1.6/1.7, budget in
    1.10) and may be ``None`` when a tool does not need them.
    """

    workspace_path: Path
    run_id: str
    sandbox: Sandbox | None = None
    workspace: Workspace | None = None
    tracer: Tracer | None = None
    budget: BudgetTracker | None = None
    #: Retriever + project id for RAG-backed tools (Phase 3); None until indexed.
    retriever: Retriever | None = None
    project_id: str | None = None


class Tool(ABC, Generic[TArgs]):
    """Base class for all tools.

    Subclasses set the class attributes and implement :meth:`run`. ``args_schema``
    is the Pydantic model the raw arguments are validated against.
    """

    name: str
    description: str
    args_schema: type[TArgs]
    #: Whether this tool triggers a human command-approval gate in `semi` autonomy.
    requires_approval: bool = False

    @abstractmethod
    def run(self, args: TArgs, ctx: ToolContext) -> ToolResult:
        """Execute the tool with validated ``args`` and ambient ``ctx``."""

    def spec(self) -> ToolSpec:
        """The provider-facing declaration used for tool-calling."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_schema.model_json_schema(),
        )


class ToolRegistry:
    """A name-indexed collection of tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}

    def register(self, tool: Tool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._tools)) or "<none>"
            raise KeyError(f"Unknown tool '{name}'. Registered: {known}.") from exc

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec() for name in self.names()]


def to_langchain_tool(tool: Tool[Any], ctx: ToolContext) -> Any:
    """Adapt a domain :class:`Tool` to a LangChain ``StructuredTool`` (Phase 2 seam).

    Imported lazily so domain code never hard-depends on LangChain.
    """
    from langchain_core.tools import StructuredTool

    def _run(**kwargs: Any) -> str:
        args = tool.args_schema.model_validate(kwargs)
        result = tool.run(args, ctx)
        return result.output if result.ok else f"ERROR: {result.error}"

    return StructuredTool.from_function(
        func=_run,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )
