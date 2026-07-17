"""Entry point for `langgraph dev` / LangGraph Studio (Task 2.9 DoD).

Not imported by application code. Studio's CLI loads ``graph`` from this
module using default settings — no checkpointer is passed here because
Studio supplies its own persistence layer when running under ``langgraph dev``.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.graph.build_graph import build_graph

graph = build_graph(get_settings())
