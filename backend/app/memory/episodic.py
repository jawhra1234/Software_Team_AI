"""Episodic memory (Task 3.11, ARCHITECTURE.md §Memory, ADR-0002).

Records run outcomes to a relational Postgres table — what a run did and how it
ended — realizing the ``finalize`` node's memory hook (a stub since Phase 2).
Relational, not vector: episodic history is queried by project/time/status, not
by similarity. Writes are best-effort — a memory-write failure must never fail a
run (finalize is a terminal, best-effort node).
"""

from __future__ import annotations

import re
from typing import Any

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict

from app.core.logging import get_logger

log = get_logger("memory.episodic")

_TABLE = "memory_episodic"

_TOKEN = re.compile(r"[a-z0-9_]+")
#: Run statuses that count as "went fine" — failures/escalations score higher for
#: planning because they carry the lesson ("last time verify failed on X").
_SUCCESS_STATUSES = frozenset({"succeeded"})


def _tokenize(text: str) -> set[str]:
    """Lexical tokens (symbols, filenames, tool names, error words), len >= 2."""
    return {t for t in _TOKEN.findall(text.lower()) if len(t) >= 2}


def _relevance(record: RunRecord, query_tokens: set[str], *, recency: float) -> float:
    """Lexical relevance of a past run to the current request.

    Episodic summaries are mostly filenames/symbols/tool names/errors, so a plain
    token overlap is enough (ADR-0008-style "keyword matters for code"). A failed
    run gets a bonus (failures are the useful lesson); recency (0..1) is a small
    tiebreak so it never dominates genuine relevance.
    """
    summary_tokens = _tokenize(f"{record.summary} {record.status}")
    overlap = float(len(summary_tokens & query_tokens))
    status_bonus = 0.0 if record.status in _SUCCESS_STATUSES else 0.75
    return overlap + status_bonus + 0.25 * recency


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    project_id: str
    status: str
    summary: str = ""
    tasks_total: int = 0
    tasks_done: int = 0


class EpisodicMemory:
    """Append-only run-outcome log in Postgres."""

    def __init__(self, dsn: str, table: str = _TABLE) -> None:
        self._dsn = dsn
        self._table = table

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        # Short timeout: a dead/unreachable DB must fail fast, never hang finalize.
        return psycopg.connect(
            self._dsn, autocommit=True, row_factory=dict_row, connect_timeout=5
        )

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id          BIGSERIAL PRIMARY KEY,
                    run_id      TEXT NOT NULL,
                    project_id  TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    summary     TEXT NOT NULL DEFAULT '',
                    tasks_total INTEGER NOT NULL DEFAULT 0,
                    tasks_done  INTEGER NOT NULL DEFAULT 0,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_project_idx "
                f"ON {self._table} (project_id, created_at)"
            )

    def record(self, record: RunRecord) -> None:
        """Best-effort insert of a run outcome (never raises)."""
        try:
            self.ensure_schema()
            with self._connect() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {self._table}
                        (run_id, project_id, status, summary, tasks_total, tasks_done)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.run_id, record.project_id, record.status,
                        record.summary, record.tasks_total, record.tasks_done,
                    ),
                )
        except Exception as exc:  # memory write must never fail a run
            log.warning("episodic_write_failed", run_id=record.run_id, error=str(exc))

    def relevant(
        self, project_id: str, query: str, k: int = 3, *, candidate_window: int = 50
    ) -> list[RunRecord]:
        """Top-k past runs most relevant to ``query`` (not just the most recent).

        Fetches a bounded recent candidate window (kept relational — episodic is
        queried by project/time, not embedded), then re-ranks it in memory by
        lexical relevance + a failure bonus + a recency tiebreak. Deterministic
        and cheap; upgrade to embeddings only if summaries stop being lexical.
        """
        candidates = self.recent(project_id, limit=candidate_window)
        if not candidates:
            return []
        query_tokens = _tokenize(query)
        n = len(candidates)
        ranked = sorted(
            enumerate(candidates),
            key=lambda ir: _relevance(ir[1], query_tokens, recency=(n - ir[0]) / n),
            reverse=True,
        )
        return [record for _, record in ranked[:k]]

    def recent(self, project_id: str, limit: int = 10) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT run_id, project_id, status, summary, tasks_total, tasks_done
                FROM {self._table} WHERE project_id = %s
                ORDER BY created_at DESC LIMIT %s
                """,
                (project_id, limit),
            ).fetchall()
        return [
            RunRecord(
                run_id=str(r["run_id"]),
                project_id=str(r["project_id"]),
                status=str(r["status"]),
                summary=str(r["summary"]),
                tasks_total=int(r["tasks_total"]),
                tasks_done=int(r["tasks_done"]),
            )
            for r in rows
        ]
