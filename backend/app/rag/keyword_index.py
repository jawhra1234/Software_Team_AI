"""BM25 keyword index — the lexical arm of hybrid retrieval (Task 3.4, ADR-0008).

Pure-Python (``rank-bm25``), in-memory, per project. Keyword search matters a
lot for code because exact symbol names dominate queries; the tokenizer is
code-aware — it splits ``snake_case`` and ``camelCase`` into subtokens (and keeps
the whole token), so a query for ``add`` matches ``add_numbers`` and
``addNumbers``, and the symbol name is indexed alongside the body.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Plus

from app.rag.vector_store import StoredChunk

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


class KeywordHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: StoredChunk
    score: float


def tokenize(text: str) -> list[str]:
    """Code-aware tokenizer: whole identifiers + their snake/camel subtokens, lowercased."""
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        lowered = word.lower()
        tokens.append(lowered)
        for part in word.split("_"):
            for sub in _CAMEL_RE.findall(part):
                sub_lower = sub.lower()
                if sub_lower != lowered:
                    tokens.append(sub_lower)
    return tokens


class KeywordIndex:
    """In-memory BM25 index over a project's chunks."""

    def __init__(self, chunks: Sequence[StoredChunk]) -> None:
        self._chunks = list(chunks)
        corpus = [tokenize(_doc(c)) for c in self._chunks]
        # BM25Plus (not Okapi): strictly-positive IDF avoids the Okapi degeneracy
        # where a term in 1-of-2 docs scores 0, which matters for tiny local repos.
        self._bm25 = BM25Plus(corpus) if any(corpus) else None

    def query(self, text: str, k: int = 8) -> list[KeywordHit]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [
            KeywordHit(chunk=chunk, score=float(score))
            for chunk, score in ranked[:k]
            if score > 0
        ]

    def __len__(self) -> int:
        return len(self._chunks)


def _doc(chunk: StoredChunk) -> str:
    """The text BM25 indexes: symbol name (high-signal) prepended to the body."""
    return f"{chunk.symbol}\n{chunk.text}" if chunk.symbol else chunk.text
