"""Tiny time helper shared by graph nodes and run bootstrapping."""

from __future__ import annotations

from datetime import UTC, datetime


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
