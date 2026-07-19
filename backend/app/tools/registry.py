"""Default tool registry assembly (Task 1.x / 2.2 / 3.8).

Registers the tool set. ``retrieve`` (hybrid RAG) is available to both the coder
and planner from Phase 3; it degrades gracefully when no index is bound, and
``search_code`` gains symbol lookups when a retriever is present.
"""

from __future__ import annotations

from app.tools.base import ToolRegistry
from app.tools.control import FinishTask
from app.tools.fs import EditFile, ListDir, ReadFile, WriteFile
from app.tools.git import GitAdd, GitCommit, GitDiff, GitStatus
from app.tools.retrieve import Retrieve
from app.tools.search import SearchCode
from app.tools.shell import RunCommand


def build_default_registry() -> ToolRegistry:
    """The coder's full tool set (read/write/exec/git/retrieve)."""
    registry = ToolRegistry()
    for tool in (
        ReadFile(),
        WriteFile(),
        EditFile(),
        ListDir(),
        SearchCode(),
        Retrieve(),
        RunCommand(),
        GitStatus(),
        GitDiff(),
        GitAdd(),
        GitCommit(),
        FinishTask(),
    ):
        registry.register(tool)
    return registry


def build_planner_registry() -> ToolRegistry:
    """The planner's read-only grounding tool set (``ARCHITECTURE.md §4.1``: no writes)."""
    registry = ToolRegistry()
    for tool in (ReadFile(), ListDir(), SearchCode(), Retrieve()):
        registry.register(tool)
    return registry
