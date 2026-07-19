"""pgvector-backed chunk store, namespaced per project (Task 3.3, ADR-0008/0010).

Persists embedded code chunks in Postgres + pgvector and serves cosine-similarity
queries. Every row carries ``project_id`` and every query filters on it, so
project namespaces are isolated at the store layer (Phase-3 risk note).

Connections are short-lived per call (autocommit) — simple and correct for the
local single-workspace scale; ANN indexing (IVFFlat/HNSW) is deferred (exact
scan is instant at this scale, and ADR-0008 forbids a reranker, so retrieval
quality is tuned via fusion, not the index).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict

from app.providers.base import Vector
from app.rag.chunker import ChunkKind, CodeChunk
from app.rag.embeddings import EmbeddedChunk

_DEFAULT_TABLE = "rag_chunks"


class StoredChunk(BaseModel):
    """A chunk as persisted/retrieved (no embedding vector)."""

    model_config = ConfigDict(extra="forbid")

    path: str
    language: str
    kind: ChunkKind
    symbol: str | None
    start_line: int
    end_line: int
    content_hash: str
    text: str


class VectorHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: StoredChunk
    score: float  # cosine similarity in [-1, 1]; higher = closer


class VectorStore:
    """CRUD + similarity query over ``rag_chunks`` for one Postgres database."""

    def __init__(self, dsn: str, dim: int, table: str = _DEFAULT_TABLE) -> None:
        self._dsn = dsn
        self._dim = dim
        self._table = table

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        conn = psycopg.connect(self._dsn, autocommit=True, row_factory=dict_row)
        register_vector(conn)
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id           BIGSERIAL PRIMARY KEY,
                    project_id   TEXT NOT NULL,
                    path         TEXT NOT NULL,
                    language     TEXT NOT NULL,
                    kind         TEXT NOT NULL,
                    symbol       TEXT,
                    start_line   INTEGER NOT NULL,
                    end_line     INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    text         TEXT NOT NULL,
                    embedding    vector({self._dim}) NOT NULL
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_project_path_idx "
                f"ON {self._table} (project_id, path)"
            )

    def add_chunks(self, project_id: str, embedded: Sequence[EmbeddedChunk]) -> int:
        if not embedded:
            return 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self._table}
                    (project_id, path, language, kind, symbol,
                     start_line, end_line, content_hash, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [self._row(project_id, e) for e in embedded],
            )
        return len(embedded)

    def replace_file(self, project_id: str, path: str, embedded: Sequence[EmbeddedChunk]) -> None:
        """Atomically replace all chunks for one file (incremental reindex)."""
        with self._connect() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE project_id = %s AND path = %s",
                (project_id, path),
            )
            if embedded:
                cur.executemany(
                    f"""
                    INSERT INTO {self._table}
                        (project_id, path, language, kind, symbol,
                         start_line, end_line, content_hash, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [self._row(project_id, e) for e in embedded],
                )

    def delete_paths(self, project_id: str, paths: Iterable[str]) -> None:
        path_list = list(paths)
        if not path_list:
            return
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {self._table} WHERE project_id = %s AND path = ANY(%s)",
                (project_id, path_list),
            )

    def clear_project(self, project_id: str) -> None:
        with self._connect() as conn:
            conn.execute(f"DELETE FROM {self._table} WHERE project_id = %s", (project_id,))

    def count(self, project_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM {self._table} WHERE project_id = %s", (project_id,)
            ).fetchone()
        return int(row["n"]) if row else 0

    def hashes_by_path(self, project_id: str) -> dict[str, set[str]]:
        """Return {path: {content_hash, ...}} — used to decide incremental reindex."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT path, content_hash FROM {self._table} WHERE project_id = %s",
                (project_id,),
            ).fetchall()
        out: dict[str, set[str]] = {}
        for row in rows:
            out.setdefault(str(row["path"]), set()).add(str(row["content_hash"]))
        return out

    def find_symbols(
        self, project_id: str, pattern: str, limit: int = 20
    ) -> list[tuple[str, str, int]]:
        """Return (symbol, path, start_line) for symbols matching ``pattern`` (case-insensitive)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT symbol, path, start_line FROM {self._table}
                WHERE project_id = %s AND symbol ILIKE %s
                ORDER BY symbol
                LIMIT %s
                """,
                (project_id, f"%{pattern}%", limit),
            ).fetchall()
        return [(str(r["symbol"]), str(r["path"]), int(r["start_line"])) for r in rows]

    def all_chunks(self, project_id: str) -> list[StoredChunk]:
        """Return every stored chunk for a project (used to build the BM25 index)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT path, language, kind, symbol, start_line, end_line, content_hash, text
                FROM {self._table} WHERE project_id = %s
                """,
                (project_id,),
            ).fetchall()
        return [
            StoredChunk(
                path=str(r["path"]),
                language=str(r["language"]),
                kind=r["kind"],
                symbol=r["symbol"],
                start_line=int(r["start_line"]),
                end_line=int(r["end_line"]),
                content_hash=str(r["content_hash"]),
                text=str(r["text"]),
            )
            for r in rows
        ]

    def query(self, project_id: str, vector: Vector, k: int = 8) -> list[VectorHit]:
        """Top-k chunks for ``project_id`` by cosine similarity (namespace-isolated)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT path, language, kind, symbol, start_line, end_line, content_hash, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self._table}
                WHERE project_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, project_id, vector, k),
            ).fetchall()
        return [
            VectorHit(
                chunk=StoredChunk(
                    path=str(r["path"]),
                    language=str(r["language"]),
                    kind=r["kind"],
                    symbol=r["symbol"],
                    start_line=int(r["start_line"]),
                    end_line=int(r["end_line"]),
                    content_hash=str(r["content_hash"]),
                    text=str(r["text"]),
                ),
                score=float(r["score"]),
            )
            for r in rows
        ]

    @staticmethod
    def _row(project_id: str, e: EmbeddedChunk) -> tuple[object, ...]:
        c: CodeChunk = e.chunk
        return (
            project_id, c.path, c.language, c.kind, c.symbol,
            c.start_line, c.end_line, c.content_hash, c.text, e.vector,
        )
