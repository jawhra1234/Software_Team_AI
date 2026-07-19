"""Task 3.4 — BM25 keyword index: code-aware tokenizer + exact-symbol retrieval."""

from __future__ import annotations

from app.rag.keyword_index import KeywordIndex, tokenize
from app.rag.vector_store import StoredChunk


def _chunk(symbol: str, text: str, path: str = "m.py") -> StoredChunk:
    return StoredChunk(
        path=path, language="python", kind="function",
        symbol=symbol, start_line=1, end_line=2, content_hash=symbol, text=text,
    )


def test_tokenize_splits_snake_and_camel() -> None:
    toks = set(tokenize("add_numbers addNumbers"))
    # whole tokens plus subtokens
    assert "add_numbers" in toks
    assert "addnumbers" in toks
    assert "add" in toks
    assert "numbers" in toks


def test_exact_symbol_query_ranks_top() -> None:
    index = KeywordIndex([
        _chunk("calculate_total", "def calculate_total(items): return sum(items)"),
        _chunk("greet", "def greet(name): return 'hi ' + name"),
    ])
    hits = index.query("calculate_total", k=2)
    assert hits
    assert hits[0].chunk.symbol == "calculate_total"


def test_subtoken_query_matches_camel_symbol() -> None:
    index = KeywordIndex([
        _chunk("parseConfig", "function parseConfig(x){ return x; }"),
        _chunk("unrelated", "def foo(): pass"),
    ])
    hits = index.query("parse config", k=2)
    assert hits[0].chunk.symbol == "parseConfig"


def test_no_match_returns_empty() -> None:
    index = KeywordIndex([_chunk("add", "def add(a, b): return a + b")])
    assert index.query("nonexistent_zzzz", k=5) == []


def test_empty_corpus() -> None:
    index = KeywordIndex([])
    assert len(index) == 0
    assert index.query("anything", k=5) == []


def test_respects_k_limit() -> None:
    chunks = [_chunk(f"handler_{i}", f"def handler_{i}(): return {i}") for i in range(10)]
    index = KeywordIndex(chunks)
    hits = index.query("handler", k=3)
    assert len(hits) == 3
