"""Construct the RAG stack (store, embedder, retriever, indexer) from Settings.

Single seam so ``build_graph`` (and tests) don't hand-assemble pgvector
connections — mirrors the existing ``providers.factory``/``checkpointer``
pattern. The ``embed`` role resolves the embedding provider (``nomic-embed-text``
by default, ADR-0004); everything else is config-driven (``RagSettings``).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.memory.episodic import EpisodicMemory
from app.memory.long_term import LongTermMemory
from app.providers.factory import get_provider
from app.rag.embeddings import ChunkEmbedder
from app.rag.indexer import Indexer
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


@dataclass
class RagStack:
    """The assembled RAG + memory components for one process."""

    embedder: ChunkEmbedder
    store: VectorStore
    retriever: Retriever
    indexer: Indexer
    long_term: LongTermMemory
    episodic: EpisodicMemory


def build_rag_stack(settings: Settings) -> RagStack:
    embed_provider = get_provider("embed", settings)
    embedder = ChunkEmbedder(provider=embed_provider, batch_size=settings.rag.embed_batch_size)
    store = VectorStore(settings.postgres.dsn, dim=settings.rag.embedding_dim)
    retriever = Retriever(
        store, embedder, candidate_k=settings.rag.candidate_k, rrf_k=settings.rag.rrf_k
    )
    indexer = Indexer(store, embedder)
    long_term = LongTermMemory(settings.postgres.dsn, embedder, dim=settings.rag.embedding_dim)
    episodic = EpisodicMemory(settings.postgres.dsn)
    return RagStack(
        embedder=embedder,
        store=store,
        retriever=retriever,
        indexer=indexer,
        long_term=long_term,
        episodic=episodic,
    )
