"""Ollama provider adapter (Task 0.4).

Binds a resolved role to a local Ollama model. Sets ``num_ctx``/``temperature``
per role and ``keep_alive`` to avoid model-reload thrash (ADR-0004), declares
``supports_tools``/``supports_json`` from config, and applies a small transport
retry with backoff (``ARCHITECTURE.md §14``).

The ``ollama`` client is imported lazily so the package can be imported (and the
factory registry populated) without a running server.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from app.core.errors import ProviderError
from app.providers.base import (
    Capabilities,
    ChatMessage,
    ChatResponse,
    Chunk,
    LLMProvider,
    ToolCall,
    ToolSpec,
    Usage,
    Vector,
)

if TYPE_CHECKING:
    from app.core.config import OllamaSettings, RoleModelConfig, Settings

_R = TypeVar("_R")

_TRANSPORT_RETRIES = 3
_BACKOFF_BASE_S = 0.5


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server."""

    def __init__(self, role_config: RoleModelConfig, ollama_settings: OllamaSettings) -> None:
        self.model = role_config.model
        self._rc = role_config
        self._settings = ollama_settings
        self.capabilities = Capabilities(
            supports_tools=role_config.supports_tools,
            supports_json=role_config.supports_json,
            max_context=role_config.max_context,
        )
        from ollama import Client

        self._client = Client(
            host=ollama_settings.base_url,
            timeout=ollama_settings.request_timeout_s,
        )

    # -- helpers ------------------------------------------------------------
    def _options(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "temperature": params.get("temperature", self._rc.temperature),
            "num_ctx": self._rc.num_ctx or self._settings.default_num_ctx,
        }

    @staticmethod
    def _format_arg(params: dict[str, Any]) -> Any:
        if "json_schema" in params:
            return params["json_schema"]
        if params.get("response_format") == "json":
            return "json"
        return None

    def _with_retry(self, call: Callable[[], _R]) -> _R:
        last_exc: Exception | None = None
        for attempt in range(_TRANSPORT_RETRIES):
            try:
                return call()
            except Exception as exc:
                last_exc = exc
                if attempt < _TRANSPORT_RETRIES - 1:
                    time.sleep(_BACKOFF_BASE_S * (2**attempt))
        raise ProviderError(
            f"Ollama request failed after {_TRANSPORT_RETRIES} attempt(s)"
        ) from last_exc

    # -- LLMProvider --------------------------------------------------------
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        **params: Any,
    ) -> ChatResponse:
        payload = [_to_ollama_message(m) for m in messages]
        ollama_tools = [_to_ollama_tool(t) for t in tools] if tools else None
        response = self._with_retry(
            lambda: self._client.chat(
                model=self.model,
                messages=payload,
                tools=ollama_tools,
                options=self._options(params),
                keep_alive=self._settings.keep_alive,
                format=self._format_arg(params),
            )
        )
        return _from_ollama_response(response)

    def stream(self, messages: Sequence[ChatMessage], **params: Any) -> Iterator[Chunk]:
        payload = [_to_ollama_message(m) for m in messages]
        stream = self._client.chat(
            model=self.model,
            messages=payload,
            options=self._options(params),
            keep_alive=self._settings.keep_alive,
            stream=True,
        )
        for part in stream:
            message = _get(part, "message", {}) or {}
            yield Chunk(
                delta=_get(message, "content", "") or "",
                done=bool(_get(part, "done", False)),
            )

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        response = self._with_retry(lambda: self._client.embed(model=self.model, input=list(texts)))
        embeddings = _get(response, "embeddings", None)
        if embeddings is None:
            raise ProviderError("Ollama embed response contained no 'embeddings'")
        return [list(vec) for vec in embeddings]


# ---------------------------------------------------------------------------
# Mapping helpers (module-level, provider-agnostic in/out shapes)
# ---------------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any) -> Any:
    """Read ``key`` from either a mapping or an attribute-style object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_ollama_message(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in message.tool_calls
        ]
    return payload


def _to_ollama_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _from_ollama_response(response: Any) -> ChatResponse:
    message = _get(response, "message", {}) or {}
    content = _get(message, "content", "") or ""

    tool_calls: list[ToolCall] = []
    for idx, raw_call in enumerate(_get(message, "tool_calls", None) or []):
        function = _get(raw_call, "function", {}) or {}
        arguments = _get(function, "arguments", {}) or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        tool_calls.append(
            ToolCall(
                id=str(_get(raw_call, "id", idx)),
                name=str(_get(function, "name", "")),
                arguments=dict(arguments),
            )
        )

    usage = Usage(
        input_tokens=_get(response, "prompt_eval_count", None),
        output_tokens=_get(response, "eval_count", None),
    )
    return ChatResponse(content=content, tool_calls=tool_calls, usage=usage)


def build_ollama_provider(role_config: RoleModelConfig, settings: Settings) -> LLMProvider:
    """Factory function registered under the ``ollama`` provider name."""
    return OllamaProvider(role_config, settings.ollama)
