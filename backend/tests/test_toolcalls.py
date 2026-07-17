"""Shared tool-call extraction (used by coder.py and planner.py)."""

from __future__ import annotations

from app.agents.toolcalls import extract_tool_calls
from app.providers.base import ToolCall
from app.tools.registry import build_default_registry


def test_native_tool_calls_preferred() -> None:
    registry = build_default_registry()
    native = extract_tool_calls("", [ToolCall(id="1", name="list_dir", arguments={})], registry)
    assert native[0].name == "list_dir"


def test_envelope_parsed_from_content_when_native_empty() -> None:
    registry = build_default_registry()
    parsed = extract_tool_calls(
        '{"name": "read_file", "arguments": {"path": "a.py"}}', [], registry
    )
    assert parsed and parsed[0].name == "read_file"


def test_unknown_tool_names_ignored() -> None:
    registry = build_default_registry()
    assert extract_tool_calls('{"name": "nope", "arguments": {}}', [], registry) == []


def test_fenced_json_envelope_parsed() -> None:
    registry = build_default_registry()
    content = '```json\n{"name": "list_dir", "arguments": {"path": "."}}\n```'
    parsed = extract_tool_calls(content, [], registry)
    assert parsed and parsed[0].name == "list_dir"


def test_non_json_content_yields_no_calls() -> None:
    registry = build_default_registry()
    assert extract_tool_calls("just some prose", [], registry) == []
