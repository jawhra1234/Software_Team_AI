"""Task 3.1 — structural chunker: intact defs, symbol table, fallback, gitignore-agnostic."""

from __future__ import annotations

from app.rag.chunker import CodeChunk, chunk_source, language_for_path

_PY = '''\
import os

CONST = 1


def top_level(a, b):
    return a + b


class Calc:
    """A calculator."""

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b
'''


def _by_symbol(chunks: list[CodeChunk]) -> dict[str | None, CodeChunk]:
    return {c.symbol: c for c in chunks}


def test_language_detection() -> None:
    assert language_for_path("a/b/calc.py") == "python"
    assert language_for_path("x.ts") == "typescript"
    assert language_for_path("x.jsx") == "javascript"
    assert language_for_path("README.md") is None


def test_python_functions_and_classes_intact() -> None:
    chunks = chunk_source("calc.py", _PY)
    by_sym = _by_symbol(chunks)

    # Top-level function is one intact chunk.
    assert "top_level" in by_sym
    top = by_sym["top_level"]
    assert top.kind == "function"
    assert "def top_level(a, b):" in top.text and "return a + b" in top.text

    # Class is captured, and its methods are qualified Class.method chunks.
    assert "Calc" in by_sym
    assert by_sym["Calc"].kind == "class"
    assert "Calc.add" in by_sym
    add = by_sym["Calc.add"]
    assert add.kind == "method"
    assert "def add(self, a, b):" in add.text
    # No mid-function split: the method chunk contains its whole body.
    assert "return a + b" in add.text


def test_symbol_table_line_mapping() -> None:
    chunks = chunk_source("calc.py", _PY)
    top = _by_symbol(chunks)["top_level"]
    # `def top_level` is on line 6 (1-indexed) in _PY.
    assert top.start_line == 6
    assert top.end_line == 7


def test_module_preamble_captures_imports() -> None:
    chunks = chunk_source("calc.py", _PY)
    module_chunks = [c for c in chunks if c.kind == "module"]
    assert module_chunks
    assert "import os" in module_chunks[0].text


def test_content_hash_populated_and_stable() -> None:
    chunks = chunk_source("calc.py", _PY)
    again = chunk_source("calc.py", _PY)
    assert all(c.content_hash for c in chunks)
    assert [c.content_hash for c in chunks] == [c.content_hash for c in again]


def test_unsupported_language_falls_back_to_windows() -> None:
    text = "\n".join(f"line {i}" for i in range(150))
    chunks = chunk_source("notes.md", text)
    assert chunks
    assert all(c.kind == "window" for c in chunks)
    # 150 lines / 60 per window = 3 windows.
    assert len(chunks) == 3
    assert chunks[0].start_line == 1


def test_unparseable_content_does_not_raise() -> None:
    # Garbage that isn't valid Python still yields chunks (fallback), never an exception.
    chunks = chunk_source("broken.py", "def (((( this is not python @@@@\n" * 5)
    assert isinstance(chunks, list)


def test_empty_file_yields_no_chunks() -> None:
    assert chunk_source("empty.py", "\n\n   \n") == []


def test_javascript_functions_detected() -> None:
    js = "function add(a, b) {\n  return a + b;\n}\n\nclass Calc {\n  mul(a, b) { return a * b; }\n}\n"
    chunks = chunk_source("calc.js", js)
    symbols = {c.symbol for c in chunks}
    assert "add" in symbols
    assert "Calc" in symbols
