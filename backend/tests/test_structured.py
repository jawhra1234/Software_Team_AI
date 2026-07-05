"""Task 0.6 — structured_call: repair-retry, tool vs JSON-mode strategy, failure."""

from __future__ import annotations

import pytest
from app.core.errors import StructuredOutputError
from app.providers.base import Capabilities, ChatMessage, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name, structured_call
from pydantic import BaseModel

from tests.fakes import FakeProvider

MESSAGES = [ChatMessage(role="user", content="give me a person")]


class Person(BaseModel):
    name: str
    age: int


def test_repair_retry_recovers_from_malformed_then_valid() -> None:
    provider = FakeProvider(
        capabilities=Capabilities(supports_tools=False, supports_json=False, max_context=8192),
        responses=[
            ChatResponse(content="not json at all"),
            ChatResponse(content='{"name": "Ada", "age": 36}'),
        ],
    )
    result = structured_call(provider, MESSAGES, Person, max_repair=2)
    assert result == Person(name="Ada", age=36)
    assert len(provider.calls) == 2  # one failure + one repair


def test_json_mode_fallback_when_tools_unsupported() -> None:
    # Capabilities fallback: supports_tools=False routes through JSON-mode prompting.
    provider = FakeProvider(
        capabilities=Capabilities(supports_tools=False, supports_json=False, max_context=8192),
        responses=[ChatResponse(content='{"name": "Grace", "age": 45}')],
    )
    structured_call(provider, MESSAGES, Person)
    call = provider.calls[0]
    assert call.tools is None  # no tools passed
    assert any("JSON Schema" in m.content for m in call.messages)  # instruction injected


def test_json_schema_passed_when_native_json_supported() -> None:
    provider = FakeProvider(
        capabilities=Capabilities(supports_tools=False, supports_json=True, max_context=8192),
        responses=[ChatResponse(content='{"name": "Lin", "age": 30}')],
    )
    structured_call(provider, MESSAGES, Person)
    assert "json_schema" in provider.calls[0].params


def test_tool_strategy_when_supported() -> None:
    tool_name = emit_tool_name(Person)
    provider = FakeProvider(
        capabilities=Capabilities(supports_tools=True, supports_json=True, max_context=8192),
        responses=[
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="1", name=tool_name, arguments={"name": "Kay", "age": 22})],
            )
        ],
    )
    result = structured_call(provider, MESSAGES, Person)
    assert result == Person(name="Kay", age=22)
    assert provider.calls[0].tools is not None
    assert provider.calls[0].tools[0].name == tool_name


def test_tool_envelope_emitted_as_content_is_unwrapped() -> None:
    # Some models (e.g. qwen2.5-coder) emit the forced tool call as JSON *content*
    # instead of a structured tool_call. structured_call must unwrap the envelope.
    tool_name = emit_tool_name(Person)
    provider = FakeProvider(
        capabilities=Capabilities(supports_tools=True, supports_json=True, max_context=8192),
        responses=[
            ChatResponse(
                content=f'{{"name": "{tool_name}", "arguments": {{"name": "Mae", "age": 41}}}}',
            )
        ],
    )
    result = structured_call(provider, MESSAGES, Person)
    assert result == Person(name="Mae", age=41)


def test_raises_after_exhausting_repairs() -> None:
    provider = FakeProvider(
        capabilities=Capabilities(supports_tools=False, supports_json=False, max_context=8192),
        responses=[ChatResponse(content="nope") for _ in range(3)],
    )
    with pytest.raises(StructuredOutputError) as exc_info:
        structured_call(provider, MESSAGES, Person, max_repair=2)
    assert exc_info.value.attempts == 3
    assert len(provider.calls) == 3
