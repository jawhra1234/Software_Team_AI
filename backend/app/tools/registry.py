"""Default tool registry assembly (Task 1.x).

Registers the Phase-1 tool set. ``retrieve`` (RAG) arrives in Phase 3; the coder
uses ``search_code`` for grounding until then.
"""

from __future__ import annotations

from app.tools.base import ToolRegistry
from app.tools.control import FinishTask
from app.tools.fs import EditFile, ListDir, ReadFile, WriteFile
from app.tools.git import GitAdd, GitCommit, GitDiff, GitStatus
from app.tools.search import SearchCode
from app.tools.shell import RunCommand


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        ReadFile(),
        WriteFile(),
        EditFile(),
        ListDir(),
        SearchCode(),
        RunCommand(),
        GitStatus(),
        GitDiff(),
        GitAdd(),
        GitCommit(),
        FinishTask(),
    ):
        registry.register(tool)
    return registry
