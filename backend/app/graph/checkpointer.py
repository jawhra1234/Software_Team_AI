"""Checkpointer construction (Task 2.11, ADR-0010).

SQLite is the clone-and-run local default; Postgres is the production-realism
path — selected by ``settings.checkpointer.backend`` alone, no code changes.
Both are context managers over a live connection: call ``build_checkpointer``
in a ``with`` block for the lifetime of graph usage, so the connection outlives
every ``invoke``/``get_state``/resume call against that checkpointer.

The custom serde registers our Pydantic state models by type so they survive
round-trips without LangGraph's "unregistered type" deserialization warning
(and the stricter blocking behavior newer LangGraph versions default toward).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from app.core.config import Settings
from app.graph.state import (
    Budget,
    CheckResult,
    ErrorRecord,
    FileRef,
    HITLRequest,
    HITLResponse,
    Plan,
    RetrievedChunk,
    Review,
    ReviewIssue,
    Task,
    VerifyResult,
)

_STATE_MODELS = [
    Budget, FileRef, Task, Plan, CheckResult, VerifyResult,
    ReviewIssue, Review, RetrievedChunk, HITLRequest, HITLResponse, ErrorRecord,
]


def build_serde() -> JsonPlusSerializer:
    """Serializer that recognizes our Pydantic state models by type (no warnings)."""
    return JsonPlusSerializer(allowed_msgpack_modules=list(_STATE_MODELS))


@contextmanager
def _sqlite_checkpointer(sqlite_path: str, serde: JsonPlusSerializer) -> Iterator[SqliteSaver]:
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        saver = SqliteSaver(conn, serde=serde)
        saver.setup()
        yield saver
    finally:
        conn.close()


@contextmanager
def _postgres_checkpointer(dsn: str, serde: JsonPlusSerializer) -> Iterator[PostgresSaver]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.Connection.connect(
        dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        saver = PostgresSaver(conn, serde=serde)
        saver.setup()
        yield saver


@contextmanager
def build_checkpointer(settings: Settings) -> Iterator[BaseCheckpointSaver[str]]:
    """Yield a configured, ready-to-use checkpointer per ``settings.checkpointer.backend``."""
    cfg = settings.checkpointer
    serde = build_serde()

    if cfg.backend == "sqlite":
        with _sqlite_checkpointer(cfg.sqlite_path, serde) as saver:
            yield saver
    else:
        with _postgres_checkpointer(settings.postgres.dsn, serde) as saver:
            yield saver
