"""Tasks 3.8/3.9 — retrieve tool + symbol-backed search_code (hermetic, fake retriever)."""

from __future__ import annotations

from pathlib import Path

from app.graph.state import RetrievedChunk
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry, build_planner_registry
from app.tools.retrieve import Retrieve, RetrieveArgs
from app.tools.search import SearchCode, SearchCodeArgs


class FakeRetriever:
    """Minimal stand-in exposing the two methods the tools call."""

    def __init__(
        self, hits: list[RetrievedChunk], symbols: list[tuple[str, str, int]]
    ) -> None:
        self._hits = hits
        self._symbols = symbols

    def retrieve(self, project_id: str, query: str, k: int = 8) -> list[RetrievedChunk]:
        return self._hits[:k]

    def find_symbols(self, project_id: str, pattern: str, limit: int = 20) -> list[tuple[str, str, int]]:
        return [s for s in self._symbols if pattern.lower() in s[0].lower()][:limit]


def _ctx(tmp_path: Path, retriever: object | None) -> ToolContext:
    return ToolContext(
        workspace_path=tmp_path,
        run_id="t",
        retriever=retriever,  # type: ignore[arg-type]
        project_id="proj" if retriever else None,
    )


# --- retrieve tool ---------------------------------------------------------
def test_retrieve_returns_formatted_hits(tmp_path: Path) -> None:
    retriever = FakeRetriever(
        hits=[RetrievedChunk(path="calc.py", symbol="add", score=0.9, content="def add(): ...")],
        symbols=[],
    )
    result = Retrieve().run(RetrieveArgs(query="add"), _ctx(tmp_path, retriever))
    assert result.ok
    assert "calc.py" in result.output and "def add()" in result.output
    assert result.meta["matches"] == 1


def test_retrieve_without_index_fails_gracefully(tmp_path: Path) -> None:
    result = Retrieve().run(RetrieveArgs(query="add"), _ctx(tmp_path, None))
    assert not result.ok
    assert "no code index" in (result.error or "")


def test_retrieve_no_hits(tmp_path: Path) -> None:
    result = Retrieve().run(RetrieveArgs(query="zzz"), _ctx(tmp_path, FakeRetriever([], [])))
    assert result.ok
    assert result.meta["matches"] == 0


# --- symbol-backed search_code --------------------------------------------
def test_search_code_appends_symbol_section(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    retriever = FakeRetriever(hits=[], symbols=[("add", "calc.py", 1)])
    result = SearchCode().run(SearchCodeArgs(query="add"), _ctx(tmp_path, retriever))
    assert result.ok
    assert "Symbols:" in result.output
    assert "calc.py:1: add" in result.output


def test_search_code_without_retriever_has_no_symbol_section(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    result = SearchCode().run(SearchCodeArgs(query="add"), _ctx(tmp_path, None))
    assert result.ok
    assert "Symbols:" not in result.output  # Phase-1 behaviour preserved


# --- registration ----------------------------------------------------------
def test_retrieve_registered_in_both_registries() -> None:
    assert "retrieve" in build_default_registry()
    assert "retrieve" in build_planner_registry()
