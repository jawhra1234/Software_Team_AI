"""Application exception hierarchy (Phase 0).

All domain errors derive from :class:`AppError` so callers can catch the whole
family without importing every concrete type.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors."""


class ConfigError(AppError):
    """Configuration is missing or invalid."""


class ProviderError(AppError):
    """An LLM provider failed to produce a usable response."""


class StructuredOutputError(ProviderError):
    """A structured (schema-validated) call failed after all repair attempts."""

    def __init__(self, schema_name: str, attempts: int, last_error: Exception | None) -> None:
        self.schema_name = schema_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Failed to obtain a valid '{schema_name}' after {attempts} attempt(s): {last_error}"
        )
