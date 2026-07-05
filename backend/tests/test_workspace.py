"""Task 1.7 — workspace lifecycle: create/attach, base_commit, work branch, diff."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.tools.base import ToolContext
from app.tools.fs import WriteFile, WriteFileArgs
from app.workspace.lifecycle import WorkspaceError, WorkspaceManager


def test_create_initializes_git_and_base_commit(tmp_path: Path) -> None:
    mgr = WorkspaceManager(tmp_path / "workspaces")
    ws = mgr.create("proj1")
    assert ws.path.is_dir()
    assert ws.git.is_repo()
    assert len(ws.base_commit) == 40
    assert ws.git.current_commit() == ws.base_commit


def test_start_run_creates_work_branch(tmp_path: Path) -> None:
    mgr = WorkspaceManager(tmp_path / "workspaces")
    ws = mgr.create("proj1")
    mgr.start_run(ws, "run-42")
    assert ws.work_branch == "agent/run-run-42"
    assert ws.git.current_branch() == "agent/run-run-42"


def test_commit_task_and_diff_since_base(tmp_path: Path) -> None:
    mgr = WorkspaceManager(tmp_path / "workspaces")
    ws = mgr.create("proj1")
    mgr.start_run(ws, "run-1")
    ctx = ToolContext(workspace_path=ws.path, run_id="run-1", workspace=ws)
    WriteFile().run(WriteFileArgs(path="app.py", content="print('hi')\n"), ctx)
    ws.commit_task("feat: add app.py")
    diff = ws.diff_since_base()
    assert "app.py" in diff and "print('hi')" in diff


def test_attach_imports_existing_project(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "sub").mkdir(parents=True)
    (source / "main.py").write_text("x = 1\n", encoding="utf-8")
    (source / "sub" / "util.py").write_text("y = 2\n", encoding="utf-8")

    mgr = WorkspaceManager(tmp_path / "workspaces")
    ws = mgr.attach("imported", source)
    assert (ws.path / "main.py").read_text() == "x = 1\n"
    assert (ws.path / "sub" / "util.py").exists()
    assert len(ws.base_commit) == 40


def test_create_rejects_nonempty_and_invalid_id(tmp_path: Path) -> None:
    mgr = WorkspaceManager(tmp_path / "workspaces")
    mgr.create("proj1")
    with pytest.raises(WorkspaceError):
        mgr.create("proj1")  # already exists, non-empty
    with pytest.raises(WorkspaceError):
        mgr.create("../escape")


def test_cleanup_removes_workspace(tmp_path: Path) -> None:
    mgr = WorkspaceManager(tmp_path / "workspaces")
    ws = mgr.create("proj1")
    mgr.cleanup("proj1")
    assert not ws.path.exists()
