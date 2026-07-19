"""Structural code chunking + symbol table (Task 3.1, ADR-0008).

Splits source files on function/class boundaries using tree-sitter, so chunks
are semantically whole (never a mid-function cut). Each chunk carries its symbol
name and line span, which doubles as the symbol table. Unsupported languages or
parse failures fall back to fixed line-window chunks — a parse error must never
break indexing (Phase-3 risk note).

Granularity (v1, deliberate): every definition node is emitted as its own intact
chunk, recursively. A class therefore yields a whole-class chunk *and* a chunk
per method (`Class.method`); the overlap is intentional — it lets retrieval hit
at both class and method granularity. Only exact-duplicate spans are de-duped.
Top-level code outside any definition (imports, constants) is coalesced into a
``module`` chunk.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from tree_sitter import Node

ChunkKind = Literal["function", "class", "method", "module", "window"]

_LANG_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}

# Definition node types per language, mapped to a chunk kind. `class`-typed nodes
# are recursed into so their methods become qualified `Class.method` chunks.
_DEFN_TYPES: dict[str, dict[str, ChunkKind]] = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
}
_NAME_NODE_TYPES = frozenset({"identifier", "property_identifier", "type_identifier"})

_WINDOW_LINES = 60


class CodeChunk(BaseModel):
    """A retrievable unit of source with its location and symbol."""

    model_config = ConfigDict(extra="forbid")

    path: str
    language: str
    kind: ChunkKind
    symbol: str | None = None
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    text: str
    content_hash: str = Field(default="")

    def model_post_init(self, _context: object) -> None:
        if not self.content_hash:
            object.__setattr__(
                self, "content_hash", hashlib.sha1(self.text.encode("utf-8")).hexdigest()
            )


def language_for_path(path: str) -> str | None:
    """Return the tree-sitter language name for a file path, or None if unsupported."""
    for suffix, lang in _LANG_BY_SUFFIX.items():
        if path.endswith(suffix):
            return lang
    return None


def chunk_source(path: str, text: str, language: str | None = None) -> list[CodeChunk]:
    """Chunk ``text`` (from ``path``) on definition boundaries, with a line-window fallback."""
    language = language or language_for_path(path)
    if language is None or language not in _DEFN_TYPES:
        return _line_window_chunks(path, text, language or "text")

    try:
        chunks = _structural_chunks(path, text, language)
    except Exception:
        return _line_window_chunks(path, text, language)
    return chunks or _line_window_chunks(path, text, language)


def _structural_chunks(path: str, text: str, language: str) -> list[CodeChunk]:
    from tree_sitter_language_pack import get_parser

    defn_types = _DEFN_TYPES[language]
    root = get_parser(language).parse(text.encode("utf-8")).root_node
    lines = text.splitlines()

    chunks: list[CodeChunk] = []
    covered_top_level: list[tuple[int, int]] = []

    def visit(node: Node, class_stack: list[str], top_level: bool) -> None:
        for child in node.children:
            kind = defn_types.get(child.type)
            if kind is None:
                visit(child, class_stack, top_level)
                continue
            # A function nested inside a class is a method, regardless of language
            # (Python has no distinct method node type — it's a function_definition).
            if kind == "function" and class_stack:
                kind = "method"
            name = _name_of(child)
            qualified = ".".join([*class_stack, name]) if name else None
            start, end = child.start_point[0] + 1, child.end_point[0] + 1
            chunks.append(
                CodeChunk(
                    path=path,
                    language=language,
                    kind=kind,
                    symbol=qualified,
                    start_line=start,
                    end_line=end,
                    text="\n".join(lines[start - 1 : end]),
                )
            )
            if top_level:
                covered_top_level.append((start, end))
            next_stack = [*class_stack, name] if kind == "class" and name else class_stack
            visit(child, next_stack, top_level=False)

    visit(root, [], top_level=True)

    preamble = _module_preamble(path, language, lines, covered_top_level)
    if preamble is not None:
        chunks.insert(0, preamble)
    return _dedupe(chunks)


def _name_of(node: Node) -> str | None:
    for child in node.children:
        if child.type in _NAME_NODE_TYPES:
            return child.text.decode("utf-8") if child.text is not None else None
    return None


def _module_preamble(
    path: str, language: str, lines: list[str], covered: list[tuple[int, int]]
) -> CodeChunk | None:
    """Coalesce top-level lines outside any definition into a single module chunk."""
    covered_lines = {ln for start, end in covered for ln in range(start, end + 1)}
    kept = [i + 1 for i, line in enumerate(lines) if (i + 1) not in covered_lines and line.strip()]
    if not kept:
        return None
    start, end = kept[0], kept[-1]
    return CodeChunk(
        path=path,
        language=language,
        kind="module",
        symbol=None,
        start_line=start,
        end_line=end,
        text="\n".join(lines[start - 1 : end]),
    )


def _line_window_chunks(path: str, text: str, language: str) -> list[CodeChunk]:
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return []
    chunks: list[CodeChunk] = []
    for start in range(0, len(lines), _WINDOW_LINES):
        window = lines[start : start + _WINDOW_LINES]
        if not any(line.strip() for line in window):
            continue
        chunks.append(
            CodeChunk(
                path=path,
                language=language,
                kind="window",
                symbol=None,
                start_line=start + 1,
                end_line=start + len(window),
                text="\n".join(window),
            )
        )
    return chunks


def _dedupe(chunks: list[CodeChunk]) -> list[CodeChunk]:
    seen: set[tuple[int, int, str]] = set()
    unique: list[CodeChunk] = []
    for chunk in chunks:
        key = (chunk.start_line, chunk.end_line, chunk.content_hash)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique
