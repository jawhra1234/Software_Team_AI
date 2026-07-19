"""Task 3.2 — chunk embedder: batching, content-hash caching, dedup within call."""

from __future__ import annotations

from collections.abc import Sequence

from app.providers.base import Vector
from app.rag.chunker import CodeChunk
from app.rag.embeddings import ChunkEmbedder

from tests.fakes import FakeProvider


class CountingEmbedProvider(FakeProvider):
    """FakeProvider that records embedded texts and returns deterministic vectors."""

    def __init__(self) -> None:
        super().__init__(model="fake-embed")
        self.embedded: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.embedded.extend(texts)
        return [[float(len(t)), float(sum(map(ord, t)) % 997)] for t in texts]


def _chunk(text: str, symbol: str) -> CodeChunk:
    return CodeChunk(
        path="m.py", language="python", kind="function",
        symbol=symbol, start_line=1, end_line=1, text=text,
    )


def test_embeds_all_chunks() -> None:
    embedder = ChunkEmbedder(provider=CountingEmbedProvider())
    chunks = [_chunk("def a(): pass", "a"), _chunk("def b(): pass", "b")]
    result = embedder.embed_chunks(chunks)
    assert len(result) == 2
    assert all(len(e.vector) == 2 for e in result)
    assert result[0].chunk.symbol == "a"


def test_content_hash_cache_avoids_recompute_across_calls() -> None:
    provider = CountingEmbedProvider()
    embedder = ChunkEmbedder(provider=provider)
    chunks = [_chunk("def a(): pass", "a")]
    embedder.embed_chunks(chunks)
    embedder.embed_chunks(chunks)  # same content_hash -> cache hit
    assert len(provider.embedded) == 1
    assert embedder.cache_size == 1


def test_identical_bodies_embedded_once_within_call() -> None:
    provider = CountingEmbedProvider()
    embedder = ChunkEmbedder(provider=provider)
    # Two chunks, different symbols, identical text -> one content_hash.
    chunks = [_chunk("return 1", "a"), _chunk("return 1", "b")]
    result = embedder.embed_chunks(chunks)
    assert len(provider.embedded) == 1
    assert result[0].vector == result[1].vector


def test_batching_respected() -> None:
    provider = CountingEmbedProvider()
    embedder = ChunkEmbedder(provider=provider, batch_size=2)
    chunks = [_chunk(f"def f{i}(): return {i}", f"f{i}") for i in range(5)]
    result = embedder.embed_chunks(chunks)
    assert len(result) == 5
    assert len(provider.embedded) == 5


def test_empty_input() -> None:
    embedder = ChunkEmbedder(provider=CountingEmbedProvider())
    assert embedder.embed_chunks([]) == []
    assert embedder.embed_texts([]) == []
