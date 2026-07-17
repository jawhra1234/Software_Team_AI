"""Custom LangGraph state reducers (Task 2.1, ARCHITECTURE.md §5).

Both reducers are called by LangGraph as ``reducer(existing, update)`` each
super-step, where ``existing`` is the value accumulated so far for that channel
(starting from whatever was passed at invoke-time) and ``update`` is the delta
returned by a node this step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.graph.state import FileRef


def merge_by_path(existing: list[FileRef], updates: list[FileRef]) -> list[FileRef]:
    """Merge ``updates`` into ``existing`` by path; the latest write wins."""
    by_path = {f.path: f for f in existing}
    for f in updates:
        by_path[f.path] = f
    return list(by_path.values())


def merge_counts(existing: dict[str, int], updates: dict[str, int]) -> dict[str, int]:
    """Sum per-key counters (e.g. per-node retry counts)."""
    merged = dict(existing)
    for key, value in updates.items():
        merged[key] = merged.get(key, 0) + value
    return merged
