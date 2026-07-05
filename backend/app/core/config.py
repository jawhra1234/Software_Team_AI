"""Application configuration (Task 0.2).

Env/file-driven settings via ``pydantic-settings``. The per-role model config
block (``models``) is the mechanism behind ADR-0003/0004: locally every chat
role resolves to one Ollama model to avoid reload thrash; each role can be
pointed at a different model/provider in the cloud by config alone.

Nested settings are populated from the environment using the ``__`` delimiter,
e.g. ``OLLAMA__BASE_URL`` or ``MODELS__CODER__MODEL``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.providers.base import Role

# Default local models (ADR-0004): one primary coder model for all chat roles,
# a small dedicated embedder that co-resides without model-reload thrash.
DEFAULT_CHAT_MODEL = "qwen2.5-coder:7b-instruct"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class RoleModelConfig(BaseModel):
    """Model configuration for a single logical role."""

    # `protected_namespaces=()` allows a field literally named `model`.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    #: Optional per-role provider override; falls back to ``Settings.provider``.
    provider: str | None = None
    model: str
    temperature: float = 0.2
    #: Context window for this role; falls back to ``OllamaSettings.default_num_ctx``.
    num_ctx: int | None = None
    supports_tools: bool = True
    supports_json: bool = True
    max_context: int = 32768


def _chat_role(temperature: float) -> RoleModelConfig:
    return RoleModelConfig(model=DEFAULT_CHAT_MODEL, temperature=temperature)


def _embed_role() -> RoleModelConfig:
    return RoleModelConfig(
        model=DEFAULT_EMBED_MODEL,
        temperature=0.0,
        supports_tools=False,
        supports_json=False,
        max_context=8192,
    )


class ModelSettings(BaseModel):
    """Per-role model configuration block."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    planner: RoleModelConfig = Field(default_factory=lambda: _chat_role(0.1))
    coder: RoleModelConfig = Field(default_factory=lambda: _chat_role(0.2))
    reviewer: RoleModelConfig = Field(default_factory=lambda: _chat_role(0.0))
    embed: RoleModelConfig = Field(default_factory=_embed_role)

    def for_role(self, role: Role) -> RoleModelConfig:
        """Return the configuration for ``role`` (typed accessor)."""
        mapping: dict[Role, RoleModelConfig] = {
            "planner": self.planner,
            "coder": self.coder,
            "reviewer": self.reviewer,
            "embed": self.embed,
        }
        return mapping[role]


class OllamaSettings(BaseModel):
    """Connection and default runtime options for the local Ollama server."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://localhost:11434"
    #: Keep models resident to avoid reload thrash between calls (ADR-0004).
    keep_alive: str = "30m"
    request_timeout_s: float = 120.0
    #: Default context window when a role does not override ``num_ctx``.
    default_num_ctx: int = 8192


class LangfuseSettings(BaseModel):
    """Self-hosted Langfuse tracing (disabled by default; graceful no-op)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    host: str = "http://localhost:3000"
    public_key: str | None = None
    secret_key: str | None = None


class PostgresSettings(BaseModel):
    """Postgres + pgvector connection (ADR-0010). Consumed by infra and later phases."""

    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    port: int = 5432
    user: str = "appuser"
    password: str = "apppassword"
    database: str = "aiswe"

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# Default command policy for run_command (Task 1.2/1.3). Allowlist is the primary
# control; deny_substrings is defense-in-depth against obviously destructive input.
_DEFAULT_ALLOW_COMMANDS = [
    "python",
    "python3",
    "pytest",
    "pip",
    "node",
    "npm",
    "npx",
    "ls",
    "cat",
    "echo",
    "pwd",
    "mkdir",
    "head",
    "tail",
    "true",
    "false",
]
_DEFAULT_DENY_SUBSTRINGS = [
    "rm -rf",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "curl",
    "wget",
    ":(){",
    "> /dev/sd",
    "chmod 777 /",
]


class SandboxSettings(BaseModel):
    """Command-execution sandbox (ADR-0007).

    ``backend='docker'`` is the security boundary; ``'subprocess'`` is the
    explicit, documented fallback for machines without Docker (cwd-jail +
    timeout + allowlist only — NO network isolation).
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["docker", "subprocess"] = "docker"
    image: str = "aiswe-sandbox:latest"
    network: str = "none"
    mem_limit: str = "1g"
    cpus: float = 2.0
    pids_limit: int = 256
    default_timeout_s: float = 60.0
    allow_commands: list[str] = Field(default_factory=lambda: list(_DEFAULT_ALLOW_COMMANDS))
    deny_substrings: list[str] = Field(default_factory=lambda: list(_DEFAULT_DENY_SUBSTRINGS))


class CoderSettings(BaseModel):
    """Budget/loop guards for the coder ReAct loop (Task 1.10)."""

    model_config = ConfigDict(extra="forbid")

    max_steps_per_task: int = 20
    max_wall_clock_s: float = 900.0
    max_tokens: int | None = None
    #: Consecutive no-progress steps tolerated before failing the task.
    no_progress_limit: int = 3
    #: Max characters retained per command output tail in state/observations.
    output_tail_chars: int = 4000
    #: Per-check timeout for the verify runner (a timeout counts as a failure).
    check_timeout_s: float = 120.0


class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    #: Default provider for roles that do not set their own ``provider``.
    provider: str = "ollama"

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    coder: CoderSettings = Field(default_factory=CoderSettings)


@lru_cache
def get_settings() -> Settings:
    """Return process-wide settings (cached). Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
