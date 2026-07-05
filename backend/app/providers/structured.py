"""Schema-validated structured output with repair-retry (Task 0.6).

``structured_call`` is the single utility that tames provider unreliability
(``ARCHITECTURE.md §7``). Strategy selection follows ADR-0003:

* ``supports_tools`` → force a single "emit" tool whose parameters are the
  target schema, and read the validated arguments back.
* otherwise → JSON-mode prompting: instruct the model to emit a bare JSON
  object (optionally constrained via the provider's native JSON support).

Either way the payload is validated against the Pydantic schema; on failure a
repair message carrying the validation error is appended and the call retried
up to ``max_repair`` times before raising :class:`StructuredOutputError`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.errors import StructuredOutputError
from app.providers.base import ChatMessage, ToolSpec

if TYPE_CHECKING:
    from app.providers.base import ChatResponse, LLMProvider

T = TypeVar("T", bound=BaseModel)


def emit_tool_name(schema: type[BaseModel]) -> str:
    """Deterministic name of the forced 'emit' tool for ``schema``."""
    return f"emit_{schema.__name__.lower()}"


def _schema_tool(schema: type[BaseModel]) -> ToolSpec:
    return ToolSpec(
        name=emit_tool_name(schema),
        description=(
            f"Return the result as a single {schema.__name__} object. "
            "Call this tool exactly once with the fully-populated arguments."
        ),
        parameters=schema.model_json_schema(),
    )


def _json_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object that validates against the JSON Schema "
        "below. Output only the JSON object — no prose, no markdown fences.\n\n"
        f"JSON Schema:\n{json.dumps(schema.model_json_schema())}"
    )


def _ensure_json_instruction(
    messages: Sequence[ChatMessage], schema: type[BaseModel]
) -> list[ChatMessage]:
    instruction = ChatMessage(role="system", content=_json_instruction(schema))
    return [instruction, *messages]


def _repair_message(error: Exception) -> str:
    return (
        "Your previous response was not valid. Error:\n"
        f"{error}\n\n"
        "Respond again with ONLY a corrected JSON object that satisfies the schema."
    )


def _coerce_json(text: str) -> Any:
    """Parse a JSON object from raw model text, tolerating fences/surrounding prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop an opening fence line (``` or ```json) and any trailing fence.
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _payload_from_tools(response: ChatResponse, tool_name: str) -> dict[str, Any] | str:
    for call in response.tool_calls:
        if call.name == tool_name:
            return call.arguments
    return response.content


def _unwrap_emit(data: Any, tool_name: str) -> Any:
    """Unwrap a tool-call envelope some models emit as content rather than a call.

    Models occasionally return ``{"name": "emit_x", "arguments": {...}}`` (or the
    same nested under ``"function"``) as plain text instead of a structured tool
    call. When the envelope names our emit tool, return the inner arguments so it
    can be validated against the target schema. Clean argument dicts (which lack a
    matching ``name`` key) are returned unchanged.
    """
    if not isinstance(data, dict):
        return data
    envelope = data
    if isinstance(envelope.get("function"), dict):
        envelope = envelope["function"]
    if envelope.get("name") == tool_name:
        for key in ("arguments", "parameters"):
            inner = envelope.get(key)
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    continue
            if isinstance(inner, dict):
                return inner
    return data


def structured_call(
    provider: LLMProvider,
    messages: Sequence[ChatMessage],
    schema: type[T],
    *,
    max_repair: int = 2,
    **params: Any,
) -> T:
    """Obtain a validated ``schema`` instance from ``provider``.

    Raises :class:`StructuredOutputError` after ``max_repair`` failed repairs.
    """
    use_tools = provider.capabilities.supports_tools
    convo: list[ChatMessage] = list(messages)
    call_params = dict(params)

    if use_tools:
        tool_spec = _schema_tool(schema)
    else:
        convo = _ensure_json_instruction(convo, schema)
        if provider.capabilities.supports_json:
            # Constrain decoding to the schema when the provider supports it.
            call_params.setdefault("json_schema", schema.model_json_schema())

    last_error: Exception | None = None
    for _attempt in range(max_repair + 1):
        if use_tools:
            response = provider.chat(convo, tools=[tool_spec], **call_params)
            payload = _payload_from_tools(response, tool_spec.name)
        else:
            response = provider.chat(convo, **call_params)
            payload = response.content

        try:
            data = payload if isinstance(payload, dict) else _coerce_json(payload)
            if use_tools:
                data = _unwrap_emit(data, tool_spec.name)
            return schema.model_validate(data)
        except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
            last_error = exc
            echoed = response.content or json.dumps(payload, default=str)
            convo = [
                *convo,
                ChatMessage(role="assistant", content=echoed),
                ChatMessage(role="user", content=_repair_message(exc)),
            ]

    raise StructuredOutputError(schema.__name__, max_repair + 1, last_error)
