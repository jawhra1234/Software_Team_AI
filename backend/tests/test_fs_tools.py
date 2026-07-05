"""Task 1.4 — filesystem tools: read/write/edit/list, happy + error paths."""

from __future__ import annotations

from pathlib import Path

from app.tools.base import ToolContext
from app.tools.fs import (
    EditFile,
    EditFileArgs,
    ListDir,
    ListDirArgs,
    ReadFile,
    ReadFileArgs,
    WriteFile,
    WriteFileArgs,
)


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path, run_id="t")


def test_write_then_read(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WriteFile().run(WriteFileArgs(path="pkg/mod.py", content="a=1\nb=2\n"), ctx)
    assert (tmp_path / "pkg" / "mod.py").read_text() == "a=1\nb=2\n"
    read = ReadFile().run(ReadFileArgs(path="pkg/mod.py"), ctx)
    assert read.ok and read.output == "a=1\nb=2\n"


def test_read_line_range(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WriteFile().run(WriteFileArgs(path="f.txt", content="l1\nl2\nl3\nl4\n"), ctx)
    read = ReadFile().run(ReadFileArgs(path="f.txt", start_line=2, end_line=3), ctx)
    assert read.output == "l2\nl3"


def test_read_missing_file(tmp_path: Path) -> None:
    assert not ReadFile().run(ReadFileArgs(path="nope.txt"), _ctx(tmp_path)).ok


def test_edit_unique_match(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WriteFile().run(WriteFileArgs(path="f.py", content="value = 1\n"), ctx)
    result = EditFile().run(
        EditFileArgs(path="f.py", old_string="value = 1", new_string="value = 2"), ctx
    )
    assert result.ok
    assert (tmp_path / "f.py").read_text() == "value = 2\n"


def test_edit_not_found(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WriteFile().run(WriteFileArgs(path="f.py", content="x=1\n"), ctx)
    assert (
        not EditFile().run(EditFileArgs(path="f.py", old_string="missing", new_string="y"), ctx).ok
    )


def test_edit_ambiguous_requires_replace_all(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WriteFile().run(WriteFileArgs(path="f.py", content="x\nx\n"), ctx)
    ambiguous = EditFile().run(EditFileArgs(path="f.py", old_string="x", new_string="y"), ctx)
    assert not ambiguous.ok and "matches 2 times" in (ambiguous.error or "")
    ok = EditFile().run(
        EditFileArgs(path="f.py", old_string="x", new_string="y", replace_all=True), ctx
    )
    assert ok.ok and (tmp_path / "f.py").read_text() == "y\ny\n"


def test_list_dir(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    WriteFile().run(WriteFileArgs(path="a.py", content=""), ctx)
    WriteFile().run(WriteFileArgs(path="sub/b.py", content=""), ctx)
    result = ListDir().run(ListDirArgs(path="."), ctx)
    assert result.ok
    assert "a.py" in result.output
    assert "sub/" in result.output
