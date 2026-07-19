"""Task 3.13 — precision@k retrieval-quality harness (integration, Postgres).

Uses a small synthetic "known repo" + a fixed query set, mixing exact-symbol
queries (keyword arm) and paraphrase queries (vector arm, via a deterministic
direction-based fake embedder) so both arms of the hybrid retriever are
exercised. Tracked as the Phase-3 regression baseline (feeds Phase 5's evals).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest
from app.core.config import Settings
from app.providers.base import Vector
from app.rag.embeddings import ChunkEmbedder
from app.rag.evaluation import RetrievalCase, index_and_evaluate
from app.rag.indexer import Indexer
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.tools.git import Git

from tests.fakes import FakeProvider

pytestmark = pytest.mark.integration

_DSN = Settings(_env_file=None).postgres.dsn
_DIM = 4


def _postgres_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


_SKIP = not _postgres_reachable(_DSN)

# Each concept gets its own axis so exact terms AND paraphrases route to the
# same vector, letting semantic queries succeed via the vector arm alone.
_AXES: dict[str, Vector] = {
    "add": [1.0, 0.0, 0.0, 0.0],
    "auth": [0.0, 1.0, 0.0, 0.0],
    "parse": [0.0, 0.0, 1.0, 0.0],
    "greet": [0.0, 0.0, 0.0, 1.0],
}


class ConceptEmbedProvider(FakeProvider):
    """Embeds by whichever concept keyword appears in the text (order matters)."""

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for t in texts:
            low = t.lower()
            vec = next((v for key, v in _AXES.items() if key in low), [0.0, 0.0, 0.0, 0.0])
            out.append(vec)
        return out


_KNOWN_REPO = {
    "calc.py": (
        "def add_numbers(a, b):\n"
        "    \"\"\"Return the sum of a and b.\"\"\"\n"
        "    return a + b\n"
    ),
    "auth.py": (
        "def authenticate_user(username, password):\n"
        "    \"\"\"Check credentials and return a session token.\"\"\"\n"
        "    return check_credentials(username, password)\n"
    ),
    "parsing.py": (
        "def parse_config(path):\n"
        "    \"\"\"Read and parse a YAML config file.\"\"\"\n"
        "    return load_yaml(path)\n"
    ),
    "greeting.py": (
        "def greet_user(name):\n"
        "    \"\"\"Return a friendly greeting for name.\"\"\"\n"
        "    return f'hello {name}'\n"
    ),
}

# Fixed query set: exact-symbol queries (keyword arm) + paraphrases (vector arm).
_CASES = [
    RetrievalCase(query="add_numbers", expected_symbol="add_numbers"),
    RetrievalCase(query="authenticate_user", expected_symbol="authenticate_user"),
    RetrievalCase(query="parse_config", expected_symbol="parse_config"),
    RetrievalCase(query="greet_user", expected_symbol="greet_user"),
    RetrievalCase(query="how do I add two numbers together", expected_symbol="add_numbers"),
    RetrievalCase(query="check a user's login credentials", expected_symbol="authenticate_user"),
    RetrievalCase(query="read a yaml configuration file", expected_symbol="parse_config"),
    RetrievalCase(query="say hello to someone", expected_symbol="greet_user"),
]

_BASELINE_PRECISION = 0.75  # regression floor; ADR-0008: tune fusion, not a reranker


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_precision_at_k_meets_baseline(tmp_path: Path) -> None:
    repo = tmp_path / "known_repo"
    repo.mkdir()
    for name, content in _KNOWN_REPO.items():
        (repo / name).write_text(content, encoding="utf-8")
    git = Git(repo)
    git.init()
    git.commit("known repo")

    project_id = f"eval-{uuid.uuid4().hex[:8]}"
    provider = ConceptEmbedProvider()
    # A dedicated table (not the shared `rag_chunks_test`): this suite's 4-axis
    # embedder needs dim=4, which would collide with other tests' dim=3 table.
    store = VectorStore(_DSN, dim=_DIM, table="rag_chunks_eval_test")
    embedder = ChunkEmbedder(provider=provider)
    indexer = Indexer(store, embedder)
    retriever = Retriever(store, embedder)

    try:
        report = index_and_evaluate(indexer, retriever, project_id, repo, _CASES, k=5)

        assert report.precision_at_k >= _BASELINE_PRECISION, (
            f"precision@5={report.precision_at_k:.2f} below baseline "
            f"{_BASELINE_PRECISION}; misses={report.misses()}"
        )
        # Exact-symbol queries (keyword arm) should be essentially perfect.
        exact_results = report.results[:4]
        assert all(r.hit for r in exact_results), [r.case for r in exact_results if not r.hit]
    finally:
        store.clear_project(project_id)
