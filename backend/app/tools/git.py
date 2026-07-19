"""Git helper and tools (Task 1.6).

A thin :class:`Git` wrapper over the ``git`` CLI, scoped to a workspace, plus
tool wrappers (status/diff/add/commit) the coder can call. The workspace
lifecycle (Task 1.7) uses :class:`Git` for init/branch/commit-per-task and the
reviewer (Phase 4) uses ``diff(base..HEAD)``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.errors import AppError
from app.tools.base import Tool, ToolContext, ToolResult

GIT_AUTHOR_NAME = "AI SWE Agent"
GIT_AUTHOR_EMAIL = "agent@aiswe.local"
DEFAULT_BRANCH = "main"


class GitError(AppError):
    """A git command failed."""


@dataclass
class Git:
    """Git operations scoped to a single workspace directory."""

    workspace_path: Path

    def _run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.workspace_path), *args],
            capture_output=True,
            text=True,
            check=check,
        )

    def _identity(self) -> list[str]:
        return ["-c", f"user.name={GIT_AUTHOR_NAME}", "-c", f"user.email={GIT_AUTHOR_EMAIL}"]

    def init(self) -> None:
        self._run("init", "-b", DEFAULT_BRANCH, check=True)

    def is_repo(self) -> bool:
        return self._run("rev-parse", "--is-inside-work-tree").returncode == 0

    def current_commit(self) -> str | None:
        result = self._run("rev-parse", "HEAD")
        return result.stdout.strip() if result.returncode == 0 else None

    def current_branch(self) -> str:
        return self._run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def has_changes(self) -> bool:
        return bool(self._run("status", "--porcelain").stdout.strip())

    def status(self) -> str:
        return self._run("status", "--short", "--branch").stdout

    def ls_files(self) -> list[str]:
        """Tracked + untracked-but-not-ignored files (respects .gitignore)."""
        result = self._run("ls-files", "--cached", "--others", "--exclude-standard")
        return [line for line in result.stdout.splitlines() if line.strip()]

    def add_all(self) -> None:
        self._run("add", "-A", check=True)

    def commit(self, message: str, *, add_all: bool = True) -> str:
        if add_all:
            self.add_all()
        result = self._run(*self._identity(), "commit", "-m", message, "--allow-empty")
        if result.returncode != 0:
            raise GitError(result.stderr.strip() or "git commit failed")
        commit = self.current_commit()
        if commit is None:
            raise GitError("commit succeeded but HEAD is unresolved")
        return commit

    def create_branch(self, name: str, *, checkout: bool = True) -> None:
        # `switch -c` creates + checks out; `branch` creates only.
        if checkout:
            self._run("switch", "-c", name, check=True)
        else:
            self._run("branch", name, check=True)

    def checkout(self, name: str) -> None:
        self._run("switch", name, check=True)

    def diff(self, ref: str | None = None) -> str:
        if ref is not None:
            return self._run("diff", ref).stdout
        if self.current_commit() is not None:
            return self._run("diff", "HEAD").stdout
        return self._run("diff").stdout

    def diff_name_status(self, ref: str) -> list[tuple[str, str]]:
        """Return (status_letter, path) pairs for changes since ``ref`` (e.g. "A", "M", "D")."""
        output = self._run("diff", "--name-status", ref).stdout
        pairs: list[tuple[str, str]] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            status, path = parts[0], parts[-1]  # renames (R100) carry old\tnew; take new path
            pairs.append((status[0], path))
        return pairs


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------
class _NoArgs(BaseModel):
    pass


class GitStatus(Tool[_NoArgs]):
    name = "git_status"
    description = "Show the working-tree status (short + branch)."
    args_schema = _NoArgs

    def run(self, args: _NoArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.success(output=Git(ctx.workspace_path).status() or "(clean)")


class GitDiffArgs(BaseModel):
    ref: str | None = Field(default=None, description="Optional ref to diff against (else HEAD).")


class GitDiff(Tool[GitDiffArgs]):
    name = "git_diff"
    description = "Show the diff of the working tree (optionally against a ref)."
    args_schema = GitDiffArgs

    def run(self, args: GitDiffArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.success(output=Git(ctx.workspace_path).diff(args.ref) or "(no changes)")


class GitAddArgs(BaseModel):
    paths: list[str] = Field(default_factory=lambda: ["."], description="Paths to stage.")


class GitAdd(Tool[GitAddArgs]):
    name = "git_add"
    description = "Stage changes for commit."
    args_schema = GitAddArgs

    def run(self, args: GitAddArgs, ctx: ToolContext) -> ToolResult:
        git = Git(ctx.workspace_path)
        git._run("add", *args.paths, check=False)
        return ToolResult.success(output=f"staged: {', '.join(args.paths)}")


class GitCommitArgs(BaseModel):
    message: str = Field(description="Commit message.")


class GitCommit(Tool[GitCommitArgs]):
    name = "git_commit"
    description = "Stage all changes and create a commit; returns the commit sha."
    args_schema = GitCommitArgs

    def run(self, args: GitCommitArgs, ctx: ToolContext) -> ToolResult:
        try:
            sha = Git(ctx.workspace_path).commit(args.message)
        except GitError as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success(output=f"committed {sha[:10]}", sha=sha)
