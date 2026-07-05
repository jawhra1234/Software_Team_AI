"""Task 1.5 — search_code (ripgrep when present, else Python fallback)."""

from __future__ import annotations

from pathlib import Path

from app.tools.base import ToolContext
from app.tools.search import SearchCode, SearchCodeArgs


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path, run_id="t")


def _seed(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def add(x, y):\n    return x + y\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("VALUE = 42\n", encoding="utf-8")


def test_finds_literal_match(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = SearchCode().run(SearchCodeArgs(query="def add"), _ctx(tmp_path))
    assert result.ok
    assert "a.py" in result.output
    assert result.meta["matches"] == 1


def test_no_matches(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = SearchCode().run(SearchCodeArgs(query="nonexistent_symbol"), _ctx(tmp_path))
    assert result.ok
    assert result.output == "(no matches)"
    assert result.meta["matches"] == 0


def test_regex_search(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = SearchCode().run(
        SearchCodeArgs(query=r"VALUE\s*=\s*\d+", is_regex=True), _ctx(tmp_path)
    )
    assert result.ok and "b.py" in result.output


def test_case_insensitive(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = SearchCode().run(SearchCodeArgs(query="value", case_insensitive=True), _ctx(tmp_path))
    assert result.ok and "b.py" in result.output
