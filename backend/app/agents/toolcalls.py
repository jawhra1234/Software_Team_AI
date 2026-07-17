"""Shared tool-call extraction (used by the coder and planner loops).

Tolerant: prefers native ``tool_calls`` and falls back to parsing a JSON
tool-call envelope from message content (some local models emit calls as text)
— the same robustness class as ``structured_call``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from app.providers.base import ToolCall
from app.tools.base import ToolRegistry


def extract_tool_calls(
    content: str, native: Sequence[ToolCall], registry: ToolRegistry
) -> list[ToolCall]:
    """Prefer native tool_calls; else parse a JSON envelope from content."""
    if native:
        return list(native)
    parsed = _parse_json(content)
    if parsed is None:
        return []
    envelopes = parsed if isinstance(parsed, list) else [parsed]
    calls: list[ToolCall] = []
    for idx, item in enumerate(envelopes):
        if not isinstance(item, dict):
            continue
        func = item.get("function")
        node: dict[str, Any] = func if isinstance(func, dict) else item
        name = node.get("name")
        args = node.get("arguments", node.get("parameters", {}))
        if isinstance(name, str) and name in registry and isinstance(args, dict):
            calls.append(ToolCall(id=str(idx), name=name, arguments=args))
    return calls


def _parse_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
