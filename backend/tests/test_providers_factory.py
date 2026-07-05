"""Task 0.5 — factory: role resolution and config-only provider/model swap."""

from __future__ import annotations

from app.core.config import ModelSettings, RoleModelConfig, Settings
from app.providers.base import LLMProvider, Role
from app.providers.factory import get_provider, register_provider

from tests.fakes import FakeProvider


def _register_fake() -> None:
    def build(role_config: RoleModelConfig, settings: Settings) -> LLMProvider:
        return FakeProvider(model=role_config.model)

    register_provider("fake", build)


def test_role_model_resolution_without_network() -> None:
    # The ollama builder resolves the model without contacting a server.
    settings = Settings(_env_file=None)
    provider = get_provider("coder", settings)
    assert provider.model == settings.models.coder.model


def test_provider_swap_is_config_only() -> None:
    _register_fake()
    settings = Settings(_env_file=None, provider="fake")
    for role in ("planner", "coder", "reviewer", "embed"):
        provider = get_provider(role, settings)
        assert isinstance(provider, FakeProvider)


def test_model_swap_changes_resolved_model() -> None:
    _register_fake()
    settings = Settings(
        _env_file=None,
        provider="fake",
        models=ModelSettings(coder=RoleModelConfig(model="swapped-model:latest")),
    )
    assert get_provider("coder", settings).model == "swapped-model:latest"


def test_per_role_provider_override() -> None:
    # ADR-0003: a single role can point at a different provider by config alone.
    _register_fake()
    settings = Settings(
        _env_file=None,
        provider="ollama",
        models=ModelSettings(
            planner=RoleModelConfig(model="cloud-planner", provider="fake"),
        ),
    )
    planner: Role = "planner"
    assert isinstance(get_provider(planner, settings), FakeProvider)
    # Other roles still use the global provider (ollama), resolved lazily.
    assert get_provider("coder", settings).model == settings.models.coder.model


def test_unknown_provider_raises() -> None:
    import pytest
    from app.core.errors import ConfigError

    settings = Settings(_env_file=None, provider="does-not-exist")
    with pytest.raises(ConfigError):
        get_provider("coder", settings)
