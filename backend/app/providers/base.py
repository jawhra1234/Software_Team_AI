"""Provider abstraction — interface and value objects (Task 0.3).

Implements the ``LLMProvider`` contract from ``ARCHITECTURE.md §7``. Domain code
depends only on the types defined here; concrete adapters (Ollama today,
OpenRouter/Gemini/Groq/OpenAI later) live behind this interface and are selected
by configuration alone (ADR-0003).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # avoid an import cycle: base's structured() calls into structured.py
    pass

#: Logical role an LLM is invoked under. Model selection is per-role (ADR-0003/0004).
Role = Literal["planner", "coder", "reviewer", "embed"]

#: Role of a single chat message.
MessageRole = Literal["system", "user", "assistant", "tool"]

#: A dense embedding vector.
Vector = list[float]

T = TypeVar("T", bound=BaseModel)


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """A single message in a chat exchange, provider-agnostic."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str
    #: On ``assistant`` messages: tool calls the model requested (multi-turn tool use).
    tool_calls: list[ToolCall] | None = None
    #: Present on ``tool`` messages: the id of the tool call being answered.
    tool_call_id: str | None = None
    #: Optional name (e.g. the tool name for ``tool`` messages).
    name: str | None = None


class ToolSpec(BaseModel):
    """Declaration of a callable tool exposed to the model.

    ``parameters`` is a JSON Schema object. This is the *shape* used by the
    provider interface; the executable tool layer arrives in Phase 1.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Token accounting for a single call, when the provider reports it."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatResponse(BaseModel):
    """Normalized result of a (non-streaming) chat call."""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    #: Provider-specific payload, retained for debugging/tracing only.
    raw: dict[str, Any] | None = None


class Chunk(BaseModel):
    """A single streamed delta."""

    model_config = ConfigDict(extra="forbid")

    delta: str = ""
    done: bool = False


class Capabilities(BaseModel):
    """Declared capabilities of a resolved model.

    ``supports_tools`` drives the structured-output strategy: models with native
    tool/function-calling use a forced-tool strategy; others fall back to
    JSON-mode prompting (ADR-0003, ``ARCHITECTURE.md §7``).
    """

    model_config = ConfigDict(extra="forbid")

    supports_tools: bool
    supports_json: bool
    max_context: int


class LLMProvider(ABC):
    """Uniform interface over chat models and embedders.

    Concrete subclasses set :attr:`model` and :attr:`capabilities` in ``__init__``.
    """

    #: The resolved model identifier this provider instance is bound to.
    model: str
    #: Declared capabilities of :attr:`model`.
    capabilities: Capabilities

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        **params: Any,
    ) -> ChatResponse:
        """Single-turn (non-streaming) completion."""

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage], **params: Any) -> Iterator[Chunk]:
        """Streaming completion yielding incremental deltas."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch of texts into vectors."""

    def structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        *,
        max_repair: int = 2,
        **params: Any,
    ) -> T:
        """Return a schema-validated object, repairing on failure (Task 0.6).

        Delegates to :func:`app.providers.structured.structured_call` so the
        repair logic lives in one place. Imported locally to avoid an import
        cycle (``structured`` imports the types defined in this module).
        """
        from app.providers.structured import structured_call

        return structured_call(self, messages, schema, max_repair=max_repair, **params)
