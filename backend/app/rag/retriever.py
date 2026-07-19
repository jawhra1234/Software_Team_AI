"""Hybrid retriever: vector + BM25 fused via RRF (Task 3.5, ADR-0008).

Runs the semantic (pgvector) and lexical (BM25) arms independently and fuses
their ranked lists with Reciprocal Rank Fusion — **no cross-encoder reranker**
(ADR-0008). Keyword matters because exact symbol names dominate code queries;
vector matters for paraphrase/semantic queries; RRF combines both without tuning
score scales. Returns the ephemeral :class:`RetrievedChunk` the graph nodes put
in ``retrieved_context``.

The BM25 index is rebuilt in-memory per project from the vector store's chunks
and cached; ``invalidate(project_id)`` drops it after a reindex.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

from app.graph.state import RetrievedChunk
from app.rag.embeddings import ChunkEmbedder
from app.rag.keyword_index import KeywordIndex
from app.rag.vector_store import StoredChunk, VectorStore

K = TypeVar("K", bound=Hashable)

_ChunkKey = tuple[str, int, int, str]


def rrf_scores(ranked_lists: Sequence[Sequence[K]], rrf_k: int = 60) -> dict[K, float]:
    """Reciprocal Rank Fusion: sum 1/(rrf_k + rank) across lists (rank is 1-indexed)."""
    scores: dict[K, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def _key(chunk: StoredChunk) -> _ChunkKey:
    return (chunk.path, chunk.start_line, chunk.end_line, chunk.content_hash)


class Retriever:
    """Hybrid (vector + BM25 → RRF) retrieval over one project's index."""

    def __init__(
        self,
        store: VectorStore,
        embedder: ChunkEmbedder,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._keyword_cache: dict[str, KeywordIndex] = {}

    def invalidate(self, project_id: str) -> None:
        self._keyword_cache.pop(project_id, None)

    def find_symbols(
        self, project_id: str, pattern: str, limit: int = 20
    ) -> list[tuple[str, str, int]]:
        """Symbol-name lookup for the symbol-backed search: (symbol, path, start_line)."""
        return self._store.find_symbols(project_id, pattern, limit=limit)

    def _keyword_index(self, project_id: str) -> KeywordIndex:
        index = self._keyword_cache.get(project_id)
        if index is None:
            index = KeywordIndex(self._store.all_chunks(project_id))
            self._keyword_cache[project_id] = index
        return index

    def retrieve(self, project_id: str, query: str, k: int = 8) -> list[RetrievedChunk]:
        query_vec = self._embedder.embed_query(query)
        vector_keys: list[_ChunkKey] = []
        by_key: dict[_ChunkKey, StoredChunk] = {}
        if query_vec:
            for vhit in self._store.query(project_id, query_vec, k=self._candidate_k):
                key = _key(vhit.chunk)
                by_key.setdefault(key, vhit.chunk)
                vector_keys.append(key)

        keyword_keys: list[_ChunkKey] = []
        for khit in self._keyword_index(project_id).query(query, k=self._candidate_k):
            key = _key(khit.chunk)
            by_key.setdefault(key, khit.chunk)
            keyword_keys.append(key)

        fused = rrf_scores([vector_keys, keyword_keys], self._rrf_k)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            RetrievedChunk(
                path=by_key[key].path,
                symbol=by_key[key].symbol,
                score=score,
                content=by_key[key].text,
            )
            for key, score in top
        ]
