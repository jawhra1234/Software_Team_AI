"""Provider factory and registry (Task 0.5).

Resolves a :class:`LLMProvider` for a given role from configuration alone
(ADR-0003). The registry is extensible: new providers register a builder under
a name, and a role selects a provider via its own ``provider`` override or the
global ``Settings.provider``. Swapping models/providers is therefore a config
edit with zero code change.
"""

from __future__ import annotations

from collections.abc import Callable

from app.core.config import RoleModelConfig, Settings, get_settings
from app.core.errors import ConfigError
from app.providers.base import LLMProvider, Role

#: A provider builder maps (role config, settings) to a ready provider instance.
ProviderBuilder = Callable[[RoleModelConfig, Settings], LLMProvider]

_REGISTRY: dict[str, ProviderBuilder] = {}
_builtins_loaded = False


def register_provider(name: str, builder: ProviderBuilder) -> None:
    """Register (or override) a provider builder under ``name``."""
    _REGISTRY[name] = builder


def _ensure_builtins() -> None:
    """Lazily register built-in providers (imported here to avoid a cycle)."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    from app.providers.ollama import build_ollama_provider

    _REGISTRY.setdefault("ollama", build_ollama_provider)
    _builtins_loaded = True


def get_provider(role: Role, settings: Settings | None = None) -> LLMProvider:
    """Resolve the provider for ``role`` from configuration.

    The provider name is the role's own override if set, else the global
    ``Settings.provider``. The model and capabilities come from the role config.
    """
    settings = settings or get_settings()
    _ensure_builtins()

    role_config = settings.models.for_role(role)
    provider_name = role_config.provider or settings.provider

    try:
        builder = _REGISTRY[provider_name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ConfigError(
            f"Unknown provider '{provider_name}' for role '{role}'. Registered: {known}."
        ) from exc

    return builder(role_config, settings)
