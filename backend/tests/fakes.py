"""Test doubles for the provider layer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from app.providers.base import (
    Capabilities,
    ChatMessage,
    ChatResponse,
    Chunk,
    LLMProvider,
    ToolSpec,
    Vector,
)


@dataclass
class RecordedCall:
    messages: list[ChatMessage]
    tools: Sequence[ToolSpec] | None
    params: dict[str, Any]


class FakeProvider(LLMProvider):
    """A scripted provider that records calls and returns queued responses."""

    def __init__(
        self,
        *,
        model: str = "fake-model",
        capabilities: Capabilities | None = None,
        responses: Sequence[ChatResponse] | None = None,
        embed_dim: int = 3,
    ) -> None:
        self.model = model
        self.capabilities = capabilities or Capabilities(
            supports_tools=False, supports_json=False, max_context=8192
        )
        self._responses: list[ChatResponse] = list(responses or [])
        self._embed_dim = embed_dim
        self.calls: list[RecordedCall] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        **params: Any,
    ) -> ChatResponse:
        self.calls.append(RecordedCall(messages=list(messages), tools=tools, params=dict(params)))
        if self._responses:
            return self._responses.pop(0)
        return ChatResponse(content="{}")

    def stream(self, messages: Sequence[ChatMessage], **params: Any) -> Iterator[Chunk]:
        yield Chunk(delta="", done=True)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [[0.0] * self._embed_dim for _ in texts]
