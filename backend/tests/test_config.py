"""Task 0.2 — configuration: defaults, per-role model block, env overrides."""

from __future__ import annotations

import pytest
from app.core.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    ModelSettings,
    Settings,
)


def test_defaults_use_single_local_chat_model(settings: Settings) -> None:
    # ADR-0004: all chat roles resolve to one primary model locally.
    assert settings.models.planner.model == DEFAULT_CHAT_MODEL
    assert settings.models.coder.model == DEFAULT_CHAT_MODEL
    assert settings.models.reviewer.model == DEFAULT_CHAT_MODEL
    assert settings.models.embed.model == DEFAULT_EMBED_MODEL
    assert settings.provider == "ollama"


def test_embed_role_capabilities(settings: Settings) -> None:
    embed = settings.models.embed
    assert embed.supports_tools is False
    assert embed.supports_json is False


def test_for_role_typed_accessor(settings: Settings) -> None:
    assert settings.models.for_role("coder") is settings.models.coder
    assert settings.models.for_role("embed") is settings.models.embed


def test_postgres_dsn(settings: Settings) -> None:
    assert settings.postgres.dsn == "postgresql://appuser:apppassword@localhost:5432/aiswe"


def test_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELS__CODER__MODEL", "llama3.1:8b")
    monkeypatch.setenv("MODELS__CODER__TEMPERATURE", "0.7")
    monkeypatch.setenv("OLLAMA__BASE_URL", "http://ollama.internal:11434")

    s = Settings(_env_file=None)
    assert s.models.coder.model == "llama3.1:8b"
    assert s.models.coder.temperature == pytest.approx(0.7)
    assert s.ollama.base_url == "http://ollama.internal:11434"
    # Other roles keep their defaults.
    assert s.models.planner.model == DEFAULT_CHAT_MODEL


def test_model_settings_partial_override_keeps_defaults() -> None:
    from app.core.config import RoleModelConfig

    models = ModelSettings(coder=RoleModelConfig(model="custom:latest"))
    assert models.coder.model == "custom:latest"
    assert models.planner.model == DEFAULT_CHAT_MODEL
