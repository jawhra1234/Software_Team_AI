"""Chunk embedding with content-hash caching (Task 3.2, ADR-0004).

Embeds :class:`CodeChunk` text via the configured provider's ``embed()`` (the
``embed`` role → ``nomic-embed-text`` locally). Embeddings are cached by the
chunk's ``content_hash`` so re-indexing an unchanged chunk costs nothing — the
key enabler of cheap incremental reindex on CPU (Phase-3 risk note).

The provider is embedded-role-agnostic: any :class:`LLMProvider` with a working
``embed()`` works, so this is unit-testable with a fake and swappable to a
hosted embedder by config alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.providers.base import LLMProvider, Vector
from app.rag.chunker import CodeChunk

# nomic-embed-text is asymmetric: it expects a task prefix on both sides so query
# and document embeddings land in the same space (ADR-0004). Missing prefixes
# still rank roughly right but with much smaller margins (observed in live
# validation). Harmless for other embedders — they just see leading text.
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


@dataclass
class EmbeddedChunk:
    """A chunk paired with its embedding vector."""

    chunk: CodeChunk
    vector: Vector


@dataclass
class ChunkEmbedder:
    """Embeds chunks in batches, caching by content hash.

    The cache is per-instance (one embedder lives for the duration of an index
    build / reindex), keyed by ``content_hash`` so identical chunk bodies — and
    unchanged files across reindex — are embedded at most once.
    """

    provider: LLMProvider
    batch_size: int = 32
    _cache: dict[str, Vector] = field(default_factory=dict, repr=False)

    def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        """Embed raw texts with no task prefix (generic pass-through, no caching)."""
        if not texts:
            return []
        return self.provider.embed(list(texts))

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed texts as documents (``search_document:`` prefix)."""
        return self.embed_texts([f"{_DOC_PREFIX}{t}" for t in texts])

    def embed_query(self, text: str) -> Vector:
        """Embed a single query (``search_query:`` prefix)."""
        return self.embed_texts([f"{_QUERY_PREFIX}{text}"])[0]

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> list[EmbeddedChunk]:
        """Embed chunks as documents, reusing cached vectors and batching misses.

        De-duplicates by ``content_hash`` both against the cache *and* within
        this call, so identical chunk bodies are embedded at most once. Chunk
        text is embedded with the ``search_document:`` prefix.
        """
        misses: dict[str, CodeChunk] = {}
        for chunk in chunks:
            if chunk.content_hash not in self._cache and chunk.content_hash not in misses:
                misses[chunk.content_hash] = chunk
        pending = list(misses.values())
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors = self.provider.embed([f"{_DOC_PREFIX}{c.text}" for c in batch])
            if len(vectors) != len(batch):
                raise ValueError(
                    f"embed() returned {len(vectors)} vectors for {len(batch)} inputs"
                )
            for chunk, vector in zip(batch, vectors, strict=True):
                self._cache[chunk.content_hash] = vector
        return [EmbeddedChunk(chunk=c, vector=self._cache[c.content_hash]) for c in chunks]

    @property
    def cache_size(self) -> int:
        return len(self._cache)
