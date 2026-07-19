"""Tasks 3.6/3.7 — indexer: full index + incremental reindex (integration, Postgres)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from app.core.config import Settings
from app.rag.embeddings import ChunkEmbedder
from app.rag.indexer import Indexer
from app.rag.vector_store import VectorStore
from app.tools.git import Git

from tests.fakes import FakeProvider

pytestmark = pytest.mark.integration

_DSN = Settings(_env_file=None).postgres.dsn


def _postgres_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


_SKIP = not _postgres_reachable(_DSN)
_DIM = 3  # FakeProvider default embed_dim


@pytest.fixture
def project_env(tmp_path: Path) -> Iterator[tuple[Indexer, VectorStore, str, Path]]:
    store = VectorStore(_DSN, dim=_DIM, table="rag_chunks_test")
    store.ensure_schema()
    indexer = Indexer(store, ChunkEmbedder(provider=FakeProvider()))
    project = f"proj-{uuid.uuid4().hex[:8]}"
    repo = tmp_path / "repo"
    repo.mkdir()
    git = Git(repo)
    git.init()
    try:
        yield indexer, store, project, repo
    finally:
        store.clear_project(project)


def _write(repo: Path, rel: str, text: str) -> None:
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text, encoding="utf-8")


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_index_project_indexes_source_and_skips_non_code(
    project_env: tuple[Indexer, VectorStore, str, Path],
) -> None:
    indexer, store, project, repo = project_env
    _write(repo, "calc.py", "def add(a, b):\n    return a + b\n")
    _write(repo, "util.py", "def greet(n):\n    return n\n")
    _write(repo, "README.md", "# not indexed as code\n")  # unsupported -> not indexed
    Git(repo).commit("init")

    stats = indexer.index_project(project, repo)
    assert stats.files_indexed == 2  # calc.py + util.py, not README.md
    assert stats.chunks_indexed >= 2
    assert store.count(project) == stats.chunks_indexed
    assert set(store.hashes_by_path(project).keys()) == {"calc.py", "util.py"}


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_reindex_changed_only_touches_modified_file(
    project_env: tuple[Indexer, VectorStore, str, Path],
) -> None:
    indexer, store, project, repo = project_env
    _write(repo, "a.py", "def a():\n    return 1\n")
    _write(repo, "b.py", "def b():\n    return 2\n")
    Git(repo).commit("init")
    indexer.index_project(project, repo)
    before = store.hashes_by_path(project)

    # Modify only a.py.
    _write(repo, "a.py", "def a():\n    return 99\n")
    Git(repo).commit("change a")
    stats = indexer.reindex_changed(project, repo)

    assert stats.files_indexed == 1  # only a.py re-embedded
    after = store.hashes_by_path(project)
    assert after["b.py"] == before["b.py"]  # b.py untouched
    assert after["a.py"] != before["a.py"]  # a.py updated


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_reindex_removes_deleted_files(
    project_env: tuple[Indexer, VectorStore, str, Path],
) -> None:
    indexer, store, project, repo = project_env
    _write(repo, "keep.py", "def keep():\n    return 1\n")
    _write(repo, "gone.py", "def gone():\n    return 2\n")
    Git(repo).commit("init")
    indexer.index_project(project, repo)

    (repo / "gone.py").unlink()
    Git(repo).commit("remove gone")
    stats = indexer.reindex_changed(project, repo)

    assert stats.files_deleted == 1
    assert set(store.hashes_by_path(project).keys()) == {"keep.py"}
