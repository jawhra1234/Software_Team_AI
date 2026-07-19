"""Tasks 3.10/3.11 — long-term (semantic) + episodic memory (integration, Postgres)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence

import pytest
from app.core.config import Settings
from app.memory.episodic import EpisodicMemory, RunRecord, _relevance, _tokenize
from app.memory.long_term import LongTermMemory
from app.providers.base import Vector
from app.rag.embeddings import ChunkEmbedder

from tests.fakes import FakeProvider

_DSN = Settings(_env_file=None).postgres.dsn


def _postgres_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


_SKIP = not _postgres_reachable(_DSN)
_DIM = 2

# Isolate test memory in dedicated tables. The integration tests use a 2-dim fake
# embedder, while the real app dimensions `memory_semantic` at 768; sharing one
# table would let whichever ran first fix the pgvector column width and break the
# other (`CREATE TABLE IF NOT EXISTS` never reconciles). Scoped names keep tests
# fully independent of the app's tables in the same dev database.
_LT_TABLE = "memory_semantic_test"
_EP_TABLE = "memory_episodic_test"


class DirEmbedProvider(FakeProvider):
    """'python' -> x-axis, 'pnpm' -> y-axis, else midpoint — enough to separate memories."""

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for t in texts:
            low = t.lower()
            if "python" in low:
                out.append([1.0, 0.0])
            elif "pnpm" in low:
                out.append([0.0, 1.0])
            else:
                out.append([0.5, 0.5])
        return out


# ---------------------------------------------------------------------------
# Long-term (semantic) memory
# ---------------------------------------------------------------------------
@pytest.fixture
def long_term() -> Iterator[tuple[LongTermMemory, str]]:
    mem = LongTermMemory(_DSN, ChunkEmbedder(provider=DirEmbedProvider()), dim=_DIM, table=_LT_TABLE)
    mem.ensure_schema()
    project = f"proj-{uuid.uuid4().hex[:8]}"
    try:
        yield mem, project
    finally:
        mem.clear_project(project)


@pytest.mark.integration
@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_long_term_write_and_semantic_search(long_term: tuple[LongTermMemory, str]) -> None:
    mem, project = long_term
    mem.write(project, "This repo uses Python 3.11 and ruff.", kind="convention")
    mem.write(project, "The frontend uses pnpm for packages.", kind="convention")

    hits = mem.search(project, "which python version", k=1)
    assert hits
    assert "Python" in hits[0].text  # semantic match to the python memory


@pytest.mark.integration
@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_long_term_namespace_isolation(long_term: tuple[LongTermMemory, str]) -> None:
    mem, project_a = long_term
    project_b = f"proj-{uuid.uuid4().hex[:8]}"
    mem.write(project_a, "python note in A", kind="note")
    mem.write(project_b, "python note in B", kind="note")
    try:
        hits = mem.search(project_a, "python", k=10)
        assert all("in A" in h.text for h in hits)
    finally:
        mem.clear_project(project_b)


# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_episodic_record_and_recent() -> None:
    mem = EpisodicMemory(_DSN, table=_EP_TABLE)
    mem.ensure_schema()
    project = f"proj-{uuid.uuid4().hex[:8]}"
    mem.record(RunRecord(run_id="r1", project_id=project, status="succeeded",
                         summary="built calc", tasks_total=2, tasks_done=2))
    mem.record(RunRecord(run_id="r2", project_id=project, status="failed",
                         summary="broke", tasks_total=1, tasks_done=0))
    recent = mem.recent(project, limit=10)
    assert {r.run_id for r in recent} == {"r1", "r2"}
    assert recent[0].run_id == "r2"  # most recent first


def test_episodic_record_never_raises_on_bad_dsn() -> None:
    # Best-effort contract: a broken DSN must not raise (finalize is terminal).
    mem = EpisodicMemory("postgresql://bad:bad@127.0.0.1:1/nope", table=_EP_TABLE)
    mem.record(RunRecord(run_id="r", project_id="p", status="succeeded"))  # no exception


# ---------------------------------------------------------------------------
# Episodic relevance ranking (lexical) — hermetic scorer tests
# ---------------------------------------------------------------------------
def _rec(status: str, summary: str) -> RunRecord:
    return RunRecord(run_id="x", project_id="p", status=status, summary=summary)


def test_relevance_prefers_lexical_overlap() -> None:
    query = _tokenize("fix apply_levy rounding in checkout.py")
    relevant = _relevance(_rec("succeeded", "edited checkout.py apply_levy"), query, recency=0.0)
    unrelated = _relevance(_rec("succeeded", "wrote unrelated readme docs"), query, recency=0.0)
    assert relevant > unrelated


def test_relevance_failure_bonus_breaks_ties() -> None:
    query = _tokenize("apply_levy checkout")
    # Same lexical overlap; the failed run should outrank the successful one.
    failed = _relevance(_rec("failed", "apply_levy checkout"), query, recency=0.0)
    ok = _relevance(_rec("succeeded", "apply_levy checkout"), query, recency=0.0)
    assert failed > ok


def test_relevance_recency_is_only_a_tiebreak() -> None:
    query = _tokenize("apply_levy checkout")
    # A strongly-relevant older run beats a recent irrelevant one (recency <= 0.25).
    old_relevant = _relevance(_rec("succeeded", "apply_levy checkout total"), query, recency=0.0)
    recent_noise = _relevance(_rec("succeeded", "totally unrelated"), query, recency=1.0)
    assert old_relevant > recent_noise


@pytest.mark.integration
@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_episodic_relevant_beats_blind_recency() -> None:
    mem = EpisodicMemory(_DSN, table=_EP_TABLE)
    mem.ensure_schema()
    project = f"proj-{uuid.uuid4().hex[:8]}"
    try:
        # Oldest: a relevant failure. Newest: an irrelevant success.
        mem.record(RunRecord(run_id="old", project_id=project, status="failed",
                             summary="apply_levy checkout.py rounding broke verify"))
        mem.record(RunRecord(run_id="mid", project_id=project, status="succeeded",
                             summary="unrelated docs update"))
        mem.record(RunRecord(run_id="new", project_id=project, status="succeeded",
                             summary="bumped the changelog"))

        hits = mem.relevant(project, "fix apply_levy rounding in checkout", k=1)
        assert hits and hits[0].run_id == "old"  # relevance beats recency
        assert mem.recent(project, limit=1)[0].run_id == "new"  # recency would pick 'new'
    finally:
        with mem._connect() as conn:  # test cleanup
            conn.execute(f"DELETE FROM {mem._table} WHERE project_id = %s", (project,))
