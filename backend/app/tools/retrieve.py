"""retrieve tool — RAG-backed grounding for plan/coder (Task 3.8, ADR-0008).

Read-only. Runs hybrid retrieval (vector + BM25 → RRF) over the project's index
and returns the top-k chunks as formatted text for the agent loop, plus the
structured hits in ``meta["chunks"]`` (serialized :class:`RetrievedChunk`) so a
graph node can capture them into ``state["retrieved_context"]`` (Task 3.12)
without re-querying. Degrades gracefully to a clear message when no
index/retriever is bound (e.g. an empty new workspace) so the agent can fall
back to ``search_code``/``read_file``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolContext, ToolResult


class RetrieveArgs(BaseModel):
    query: str = Field(description="What to find — a symbol name, behaviour, or concept.")
    k: int = Field(default=6, ge=1, le=20, description="Max number of code chunks to return.")


class Retrieve(Tool[RetrieveArgs]):
    name = "retrieve"
    description = (
        "Search the indexed codebase for relevant code (hybrid semantic + keyword). "
        "Use this to find where something is defined or how a concept is implemented."
    )
    args_schema = RetrieveArgs

    def run(self, args: RetrieveArgs, ctx: ToolContext) -> ToolResult:
        if ctx.retriever is None or ctx.project_id is None:
            return ToolResult.failure("no code index is available for this project")
        hits = ctx.retriever.retrieve(ctx.project_id, args.query, k=args.k)
        if not hits:
            return ToolResult.success(output="(no matching code found)", matches=0, chunks=[])
        blocks = [
            f"### {h.path}" + (f" — {h.symbol}" if h.symbol else "") + f"\n{h.content}"
            for h in hits
        ]
        return ToolResult.success(
            output="\n\n".join(blocks),
            matches=len(hits),
            chunks=[h.model_dump() for h in hits],
        )
