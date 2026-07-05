"""Task 1.6 — Git helper and tools over a real workspace repo."""

from __future__ import annotations

from pathlib import Path

from app.tools.base import ToolContext
from app.tools.git import DEFAULT_BRANCH, Git, GitCommit, GitCommitArgs, GitDiff, GitDiffArgs


def _seed_repo(tmp_path: Path) -> Git:
    git = Git(tmp_path)
    git.init()
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    return git


def test_init_commit_and_current_commit(tmp_path: Path) -> None:
    git = _seed_repo(tmp_path)
    assert git.is_repo()
    assert git.current_commit() is None  # nothing committed yet
    sha = git.commit("initial")
    assert len(sha) == 40
    assert git.current_commit() == sha
    assert git.current_branch() == DEFAULT_BRANCH
    assert not git.has_changes()


def test_diff_reflects_changes(tmp_path: Path) -> None:
    git = _seed_repo(tmp_path)
    base = git.commit("base")
    (tmp_path / "a.txt").write_text("hello world\n", encoding="utf-8")
    assert git.has_changes()
    assert "hello world" in git.diff()
    # Diff against the base commit shows the change too.
    git.commit("update")
    assert "hello world" in git.diff(f"{base}..HEAD")


def test_branch_create_and_checkout(tmp_path: Path) -> None:
    git = _seed_repo(tmp_path)
    git.commit("base")
    git.create_branch("agent/run-1")
    assert git.current_branch() == "agent/run-1"
    git.checkout(DEFAULT_BRANCH)
    assert git.current_branch() == DEFAULT_BRANCH


def test_commit_tool_and_diff_tool(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    ctx = ToolContext(workspace_path=tmp_path, run_id="t")
    commit_result = GitCommit().run(GitCommitArgs(message="via tool"), ctx)
    assert commit_result.ok
    assert len(commit_result.meta["sha"]) == 40
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    diff_result = GitDiff().run(GitDiffArgs(), ctx)
    assert diff_result.ok and "changed" in diff_result.output
