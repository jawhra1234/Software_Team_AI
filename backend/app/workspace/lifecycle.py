"""Workspace lifecycle (Task 1.7, ARCHITECTURE.md §8).

Each project is a git-backed directory under a workspaces root. On create/attach
a ``base_commit`` is recorded; each run works on a ``agent/run-<id>`` branch so
the accumulating diff is both the review artifact and the rollback unit.
"""

from __future__ import annotations

import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.errors import AppError
from app.tools.git import Git


class WorkspaceError(AppError):
    """Invalid workspace operation (e.g. clobbering a non-empty target)."""


@dataclass
class Workspace:
    """A git-backed project directory and its run state."""

    project_id: str
    path: Path
    git: Git
    base_commit: str
    work_branch: str

    def commit_task(self, message: str) -> str:
        """Commit the current changes (one commit per verified task)."""
        return self.git.commit(message)

    def diff_since_base(self) -> str:
        """The cumulative diff from base_commit to HEAD (the review artifact)."""
        if self.git.current_commit() == self.base_commit:
            return self.git.diff()  # nothing committed on the branch yet
        return self.git.diff(f"{self.base_commit}..HEAD")

    def status(self) -> str:
        return self.git.status()


class WorkspaceManager:
    """Allocates and manages git-backed workspaces under a root directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _project_path(self, project_id: str) -> Path:
        path = (self.root / project_id).resolve()
        # Jail: project dirs must live directly under the root.
        if path.parent != self.root:
            raise WorkspaceError(f"invalid project_id: {project_id!r}")
        return path

    @staticmethod
    def _ensure_base_commit(git: Git, message: str) -> str:
        if git.current_commit() is None:
            return git.commit(message)
        if git.has_changes():
            return git.commit(message)
        commit = git.current_commit()
        assert commit is not None  # just established above
        return commit

    def create(self, project_id: str, *, exist_ok: bool = False) -> Workspace:
        """Create a fresh, empty, git-initialized workspace."""
        path = self._project_path(project_id)
        if path.exists() and any(path.iterdir()) and not exist_ok:
            raise WorkspaceError(f"workspace already exists and is not empty: {project_id}")
        path.mkdir(parents=True, exist_ok=True)
        git = Git(path)
        if not git.is_repo():
            git.init()
        base = self._ensure_base_commit(git, "chore: initialize workspace")
        return Workspace(project_id, path, git, base_commit=base, work_branch=git.current_branch())

    def attach(self, project_id: str, source: Path, *, exist_ok: bool = False) -> Workspace:
        """Import an existing project tree into a git-backed workspace."""
        path = self._project_path(project_id)
        if path.exists() and any(path.iterdir()) and not exist_ok:
            raise WorkspaceError(f"workspace already exists and is not empty: {project_id}")
        path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, path, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
        git = Git(path)
        if not git.is_repo():
            git.init()
        base = self._ensure_base_commit(git, "chore: import existing project")
        return Workspace(project_id, path, git, base_commit=base, work_branch=git.current_branch())

    def start_run(self, workspace: Workspace, run_id: str) -> Workspace:
        """Create and switch to the run's work branch off the current HEAD."""
        branch = f"agent/run-{run_id}"
        workspace.git.create_branch(branch)
        workspace.work_branch = branch
        return workspace

    def cleanup(self, project_id: str) -> None:
        """Remove a workspace directory (handles read-only .git files on Windows)."""
        path = self._project_path(project_id)
        if path.exists():
            shutil.rmtree(path, onerror=_force_remove)


def _force_remove(func: Callable[..., Any], path: str, _exc: Any) -> None:
    Path(path).chmod(stat.S_IWRITE)
    func(path)
