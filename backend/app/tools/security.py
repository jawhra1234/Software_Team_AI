"""Workspace path-jail (Task 1.3).

Ensures every filesystem path a tool touches resolves inside the workspace
directory, defeating traversal (``../``) and absolute-path escapes.
"""

from __future__ import annotations

from pathlib import Path

from app.core.errors import AppError


class PathJailError(AppError):
    """A path resolved outside the workspace boundary."""


def resolve_within(workspace_path: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` against the workspace, rejecting any escape.

    Relative paths are resolved under the workspace; absolute paths must already
    be inside it. Raises :class:`PathJailError` on any escape.
    """
    base = workspace_path.resolve()
    candidate_path = Path(candidate)
    target = (candidate_path if candidate_path.is_absolute() else base / candidate_path).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise PathJailError(
            f"Path '{candidate}' resolves outside the workspace ({target})"
        ) from exc
    return target
