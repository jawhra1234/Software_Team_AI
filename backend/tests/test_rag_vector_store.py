"""Task 3.3 — pgvector store (integration; needs live Postgres + pgvector).

Verifies upsert/query round-trip, cosine ranking, namespace isolation, and the
incremental-reindex helpers. Uses a unique project_id per test and cleans up, so
runs are independent and don't collide with other data in the shared DB.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from app.core.config import Settings
from app.rag.chunker import CodeChunk
from app.rag.embeddings import EmbeddedChunk
from app.rag.vector_store import VectorStore

pytestmark = pytest.mark.integration

_DIM = 3


def _postgres_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


_DSN = Settings(_env_file=None).postgres.dsn
_SKIP = not _postgres_reachable(_DSN)


def _emb(symbol: str, text: str, vec: list[float], path: str = "m.py") -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=CodeChunk(
            path=path, language="python", kind="function",
            symbol=symbol, start_line=1, end_line=2, text=text,
        ),
        vector=vec,
    )


@pytest.fixture
def store() -> Iterator[VectorStore]:
    s = VectorStore(_DSN, dim=_DIM, table="rag_chunks_test")
    s.ensure_schema()
    project = f"proj-{uuid.uuid4().hex[:8]}"
    # expose the project id via attribute for the test to use
    s._test_project = project  # type: ignore[attr-defined]
    try:
        yield s
    finally:
        s.clear_project(project)


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_upsert_and_cosine_query(store: VectorStore) -> None:
    project = store._test_project  # type: ignore[attr-defined]
    store.add_chunks(project, [
        _emb("add", "def add(a, b): return a + b", [1.0, 0.0, 0.0]),
        _emb("sub", "def sub(a, b): return a - b", [0.0, 1.0, 0.0]),
    ])
    assert store.count(project) == 2

    hits = store.query(project, [0.9, 0.1, 0.0], k=2)
    assert hits[0].chunk.symbol == "add"  # closest to the [1,0,0] direction
    assert hits[0].score > hits[1].score


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_namespace_isolation(store: VectorStore) -> None:
    project_a = store._test_project  # type: ignore[attr-defined]
    project_b = f"proj-{uuid.uuid4().hex[:8]}"
    store.add_chunks(project_a, [_emb("a", "in A", [1.0, 0.0, 0.0])])
    store.add_chunks(project_b, [_emb("b", "in B", [1.0, 0.0, 0.0])])
    try:
        hits = store.query(project_a, [1.0, 0.0, 0.0], k=10)
        symbols = {h.chunk.symbol for h in hits}
        assert symbols == {"a"}  # never returns project B's chunk
    finally:
        store.clear_project(project_b)


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_replace_file_incremental(store: VectorStore) -> None:
    project = store._test_project  # type: ignore[attr-defined]
    store.add_chunks(project, [_emb("old", "old body", [1.0, 0.0, 0.0], path="x.py")])
    store.replace_file(project, "x.py", [_emb("new", "new body", [0.0, 1.0, 0.0], path="x.py")])
    hits = store.query(project, [0.0, 1.0, 0.0], k=10)
    symbols = {h.chunk.symbol for h in hits}
    assert symbols == {"new"}  # old chunk replaced


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_hashes_by_path(store: VectorStore) -> None:
    project = store._test_project  # type: ignore[attr-defined]
    store.add_chunks(project, [
        _emb("a", "body a", [1.0, 0.0, 0.0], path="a.py"),
        _emb("b", "body b", [0.0, 1.0, 0.0], path="b.py"),
    ])
    mapping = store.hashes_by_path(project)
    assert set(mapping.keys()) == {"a.py", "b.py"}
    assert all(len(hashes) == 1 for hashes in mapping.values())
