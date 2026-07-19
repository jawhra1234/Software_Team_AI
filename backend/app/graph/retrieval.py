"""Capture ``retrieve`` tool hits into graph state (Task 3.12).

A fresh :class:`RetrievalCapture` is created per node invocation (plan/coder),
passed as the agent loop's tool-result observer, and its accumulated chunks are
returned as ``retrieved_context`` in the node's patch. Since that state field
has no reducer (overwrite semantics), each node invocation's capture replaces
the previous one — ``retrieved_context`` is ephemeral per step, not accumulated
across a run (ARCHITECTURE.md §5).
"""

from __future__ import annotations

from app.graph.state import RetrievedChunk
from app.tools.base import ToolResult


class RetrievalCapture:
    """Collects ``RetrievedChunk``s surfaced by ``retrieve`` tool calls in one node run."""

    def __init__(self) -> None:
        self.chunks: list[RetrievedChunk] = []

    def observe(self, tool_name: str, result: ToolResult) -> None:
        if tool_name != "retrieve" or not result.ok:
            return
        for raw in result.meta.get("chunks", []):
            self.chunks.append(RetrievedChunk(**raw))
