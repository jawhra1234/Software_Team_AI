"""Task 3.13 — ADR ingestion into long-term memory (hermetic + integration)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from app.core.config import Settings
from app.memory.ingest import ingest_adrs, parse_adr
from app.memory.long_term import LongTermMemory, MemoryKind
from app.rag.embeddings import ChunkEmbedder

from tests.fakes import FakeProvider
from tests.test_memory import _LT_TABLE, DirEmbedProvider  # reuse the fake embedder + test table

_ADR = """\
# ADR-0007: Sandboxed execution

**Status:** Accepted

## Context
Agents run arbitrary commands.

## Decision
Run every command in a Docker sandbox with --network=none and resource limits.

## Consequences
- Safe by default.
"""


# ---------------------------------------------------------------------------
# parse_adr (pure)
# ---------------------------------------------------------------------------
def test_parse_adr_extracts_title_and_decision() -> None:
    fact = parse_adr(_ADR)
    assert fact is not None
    assert "ADR-0007: Sandboxed execution" in fact
    assert "Docker sandbox" in fact  # the Decision section
    assert "Safe by default" not in fact  # Consequences excluded


def test_parse_adr_falls_back_without_decision_section() -> None:
    fact = parse_adr("# Title only\n\nsome body text")
    assert fact is not None
    assert "Title only" in fact and "some body text" in fact


def test_parse_adr_returns_none_without_title() -> None:
    assert parse_adr("no heading here") is None


# ---------------------------------------------------------------------------
# ingest_adrs (hermetic — fake store records writes, no Postgres)
# ---------------------------------------------------------------------------
class _RecordingLongTerm(LongTermMemory):
    """A LongTermMemory that records writes/clears instead of hitting Postgres."""

    def __init__(self) -> None:  # deliberately skips the DB-bound base __init__
        self.written: list[tuple[MemoryKind, str]] = []
        self.cleared: list[MemoryKind | None] = []

    def clear_project(self, project_id: str, kind: MemoryKind | None = None) -> None:
        self.cleared.append(kind)

    def write_many(self, project_id: str, items: Sequence[tuple[MemoryKind, str]]) -> int:
        self.written.extend(items)
        return len(items)


def test_ingest_adrs_writes_decisions_and_skips_readme(tmp_path: Path) -> None:
    (tmp_path / "0001-a.md").write_text(_ADR, encoding="utf-8")
    (tmp_path / "0002-b.md").write_text("# ADR-0002: Another\n\n## Decision\nDo B.\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Index\n\n## Decision\nnot a real ADR\n", encoding="utf-8")

    mem = _RecordingLongTerm()
    count = ingest_adrs(mem, "proj", tmp_path)

    assert count == 2  # README.md skipped
    assert all(kind == "decision" for kind, _ in mem.written)
    assert mem.cleared == ["decision"]  # kind-scoped, idempotent
    assert any("Another" in text for _, text in mem.written)


def test_ingest_adrs_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "0001-a.md").write_text(_ADR, encoding="utf-8")
    mem = _RecordingLongTerm()
    ingest_adrs(mem, "proj", tmp_path)
    ingest_adrs(mem, "proj", tmp_path)
    # cleared before each run -> no unbounded growth semantics leaking to callers
    assert mem.cleared == ["decision", "decision"]


# ---------------------------------------------------------------------------
# Integration — real pgvector round-trip
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


@pytest.fixture
def long_term() -> Iterator[tuple[LongTermMemory, str]]:
    mem = LongTermMemory(_DSN, ChunkEmbedder(provider=DirEmbedProvider()), dim=2, table=_LT_TABLE)
    mem.ensure_schema()
    project = f"proj-{uuid.uuid4().hex[:8]}"
    try:
        yield mem, project
    finally:
        mem.clear_project(project)


@pytest.mark.integration
@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_ingest_then_search_roundtrip(tmp_path: Path, long_term: tuple[LongTermMemory, str]) -> None:
    mem, project = long_term
    (tmp_path / "0001-python.md").write_text(
        "# ADR: Python toolchain\n\n## Decision\nUse python 3.11 and ruff.\n", encoding="utf-8"
    )
    written = ingest_adrs(mem, project, tmp_path)
    assert written == 1

    hits = mem.search(project, "which python version", k=1)
    assert hits and "python" in hits[0].text.lower()
    assert hits[0].kind == "decision"

    # Re-ingest is idempotent: still exactly one decision, not two.
    ingest_adrs(mem, project, tmp_path)
    assert len(mem.search(project, "python", k=10)) == 1


def test_ingest_uses_fake_embedder_offline() -> None:
    # Guards the import surface: DirEmbedProvider is a FakeProvider (no network).
    assert isinstance(DirEmbedProvider(), FakeProvider)
