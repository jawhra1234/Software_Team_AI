"""Entry point for `langgraph dev` / LangGraph Studio (Task 2.9 DoD, 3.12).

Not imported by application code. Studio's CLI loads ``graph`` from this
module using default settings — no checkpointer is passed here because
Studio supplies its own persistence layer when running under ``langgraph dev``.

RAG/episodic memory are opt-in in ``build_graph`` (see its docstring), so this
"real" entry point explicitly builds and wires the stack — construction alone
never touches the network (providers/stores connect lazily per call).
"""

from __future__ import annotations

from app.core.config import get_settings
from app.graph.build_graph import build_graph
from app.rag.factory import build_rag_stack

_settings = get_settings()
_rag = build_rag_stack(_settings)

graph = build_graph(_settings, retriever=_rag.retriever, episodic=_rag.episodic)
