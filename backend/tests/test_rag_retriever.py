"""Task 3.5 — hybrid retriever: RRF fusion (hermetic) + full retrieve (integration)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence

import pytest
from app.core.config import Settings
from app.providers.base import Vector
from app.rag.chunker import CodeChunk
from app.rag.embeddings import ChunkEmbedder, EmbeddedChunk
from app.rag.retriever import Retriever, rrf_scores
from app.rag.vector_store import VectorStore

from tests.fakes import FakeProvider


# ---------------------------------------------------------------------------
# RRF fusion — pure function, hermetic
# ---------------------------------------------------------------------------
def test_rrf_rewards_items_high_in_both_lists() -> None:
    vector = ["a", "b", "c"]
    keyword = ["b", "a", "d"]
    scores = rrf_scores([vector, keyword], rrf_k=60)
    # 'a' (ranks 1,2) and 'b' (ranks 2,1) beat singletons 'c'/'d'.
    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
    assert set(ranked[:2]) == {"a", "b"}
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]


def test_rrf_single_list() -> None:
    scores = rrf_scores([["x", "y"]], rrf_k=60)
    assert scores["x"] > scores["y"]


def test_rrf_empty() -> None:
    assert rrf_scores([[], []]) == {}


# ---------------------------------------------------------------------------
# Full hybrid retrieve — integration (Postgres); deterministic fake embedder
# ---------------------------------------------------------------------------
_DSN = Settings(_env_file=None).postgres.dsn


def _postgres_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


_SKIP = not _postgres_reachable(_DSN)
_DIM = 3


class DirEmbedProvider(FakeProvider):
    """Embeds by keyword direction: 'add' -> x-axis, 'sub' -> y-axis, else z."""

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for t in texts:
            low = t.lower()
            if "add" in low:
                out.append([1.0, 0.0, 0.0])
            elif "sub" in low:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return out


def _emb(provider: DirEmbedProvider, symbol: str, text: str) -> EmbeddedChunk:
    chunk = CodeChunk(
        path=f"{symbol}.py", language="python", kind="function",
        symbol=symbol, start_line=1, end_line=2, text=text,
    )
    return EmbeddedChunk(chunk=chunk, vector=provider.embed([text])[0])


@pytest.fixture
def indexed() -> Iterator[tuple[Retriever, str]]:
    provider = DirEmbedProvider()
    store = VectorStore(_DSN, dim=_DIM, table="rag_chunks_test")
    store.ensure_schema()
    project = f"proj-{uuid.uuid4().hex[:8]}"
    store.add_chunks(project, [
        _emb(provider, "add_numbers", "def add_numbers(a, b): return a + b"),
        _emb(provider, "subtract", "def subtract(a, b): return a - b"),
        _emb(provider, "greet", "def greet(name): return name"),
    ])
    retriever = Retriever(store, ChunkEmbedder(provider=provider))
    try:
        yield retriever, project
    finally:
        store.clear_project(project)


@pytest.mark.integration
@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_exact_symbol_retrieved_via_keyword_arm(indexed: tuple[Retriever, str]) -> None:
    retriever, project = indexed
    hits = retriever.retrieve(project, "add_numbers", k=3)
    assert hits
    assert hits[0].symbol == "add_numbers"


@pytest.mark.integration
@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_semantic_query_retrieved_via_vector_arm(indexed: tuple[Retriever, str]) -> None:
    retriever, project = indexed
    # Query text embeds to the 'add' direction; symbol wording differs from query.
    hits = retriever.retrieve(project, "addition helper", k=3)
    assert any(h.symbol == "add_numbers" for h in hits)


@pytest.mark.integration
@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_retrieve_respects_k(indexed: tuple[Retriever, str]) -> None:
    retriever, project = indexed
    hits = retriever.retrieve(project, "add", k=1)
    assert len(hits) == 1
