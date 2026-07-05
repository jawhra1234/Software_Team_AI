"""Task 0.8 — infra: docker-compose declares a healthy pgvector + Langfuse stack.

This validates the compose *configuration* (no Docker required). Live
reachability is exercised by `scripts/bootstrap.*` after `docker compose up`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "infra" / "docker-compose.yml"
INIT_DIR = Path(__file__).resolve().parents[2] / "infra" / "postgres" / "init"


def _load() -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(COMPOSE.read_text(encoding="utf-8")))


def test_compose_file_exists() -> None:
    assert COMPOSE.is_file()


def test_postgres_uses_pgvector_image_with_healthcheck() -> None:
    services = _load()["services"]
    assert "postgres" in services
    assert "pgvector" in services["postgres"]["image"]
    assert "healthcheck" in services["postgres"]


def test_langfuse_service_present_and_depends_on_postgres() -> None:
    services = _load()["services"]
    assert "langfuse" in services
    assert "postgres" in services["langfuse"]["depends_on"]


def test_ollama_is_opt_in_profile() -> None:
    # ADR-0004: Ollama runs natively by default; container is behind a profile.
    services = _load()["services"]
    assert services["ollama"]["profiles"] == ["with-ollama"]


def test_pgvector_extension_init_script_present() -> None:
    sql = (INIT_DIR / "00-extensions.sql").read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
