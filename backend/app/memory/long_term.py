"""Long-term / semantic memory (Task 3.10, ARCHITECTURE.md §Memory, ADR-0010).

A namespaced pgvector store of durable facts — decisions/ADRs and learned repo
conventions ("this repo uses pnpm") — distinct from the code index (Task 3.3).
Written across runs, retrieved by semantic similarity to inform planning. Kept
deliberately small and separate so its lifecycle (human-curated / decision-level)
doesn't mix with the churn of code chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict

from app.rag.embeddings import ChunkEmbedder

MemoryKind = Literal["decision", "convention", "note"]

_TABLE = "memory_semantic"


class MemoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: MemoryKind
    text: str
    score: float = 0.0


class LongTermMemory:
    """Namespaced semantic memory over Postgres + pgvector."""

    def __init__(self, dsn: str, embedder: ChunkEmbedder, dim: int, table: str = _TABLE) -> None:
        self._dsn = dsn
        self._embedder = embedder
        self._dim = dim
        self._table = table

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        conn = psycopg.connect(
            self._dsn, autocommit=True, row_factory=dict_row, connect_timeout=5
        )
        register_vector(conn)
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id         BIGSERIAL PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    kind       TEXT NOT NULL,
                    text       TEXT NOT NULL,
                    embedding  vector({self._dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_project_idx "
                f"ON {self._table} (project_id)"
            )

    def write(self, project_id: str, text: str, kind: MemoryKind = "note") -> None:
        vector = self._embedder.embed_documents([text])[0]
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {self._table} (project_id, kind, text, embedding) "
                f"VALUES (%s, %s, %s, %s)",
                (project_id, kind, text, vector),
            )

    def write_many(self, project_id: str, items: Sequence[tuple[MemoryKind, str]]) -> int:
        if not items:
            return 0
        vectors = self._embedder.embed_documents([text for _, text in items])
        rows = [
            (project_id, kind, text, vec)
            for (kind, text), vec in zip(items, vectors, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self._table} (project_id, kind, text, embedding) "
                f"VALUES (%s, %s, %s, %s)",
                rows,
            )
        return len(items)

    def search(self, project_id: str, query: str, k: int = 5) -> list[MemoryItem]:
        vector = self._embedder.embed_query(query)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT kind, text, 1 - (embedding <=> %s::vector) AS score
                FROM {self._table}
                WHERE project_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, project_id, vector, k),
            ).fetchall()
        return [
            MemoryItem(kind=r["kind"], text=str(r["text"]), score=float(r["score"]))
            for r in rows
        ]

    def clear_project(self, project_id: str, kind: MemoryKind | None = None) -> None:
        """Delete a project's memories; scope to a single ``kind`` when given.

        A kind-scoped clear lets a deterministic writer (e.g. ADR ingestion,
        which owns the ``decision`` kind) re-run idempotently without wiping
        memories written by other sources/kinds.
        """
        with self._connect() as conn:
            if kind is None:
                conn.execute(f"DELETE FROM {self._table} WHERE project_id = %s", (project_id,))
            else:
                conn.execute(
                    f"DELETE FROM {self._table} WHERE project_id = %s AND kind = %s",
                    (project_id, kind),
                )
