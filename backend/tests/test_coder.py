"""Task 1.8 — coder ReAct loop, hermetic (scripted FakeProvider)."""

from __future__ import annotations

from pathlib import Path

from app.agents.coder import Coder, CoderTask, _extract_tool_calls, _workspace_signature
from app.core.config import CoderSettings, Settings
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry

from tests.fakes import FakeProvider

_TOOL_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path, run_id="t")


def _call(name: str, **arguments: object) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="1", name=name, arguments=dict(arguments))]
    )


def test_coder_completes_task(tmp_path: Path) -> None:
    provider = FakeProvider(
        capabilities=_TOOL_CAPS,
        responses=[
            _call("write_file", path="calc.py", content="def add(a, b):\n    return a + b\n"),
            _call("finish_task", summary="added add()"),
        ],
    )
    outcome = Coder(provider, build_default_registry(), Settings(_env_file=None)).run_task(
        CoderTask(description="add function"), _ctx(tmp_path)
    )
    assert outcome.status == "completed"
    assert (tmp_path / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_coder_budget_exceeded(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None, coder=CoderSettings(max_steps_per_task=2, no_progress_limit=9)
    )
    provider = FakeProvider(
        capabilities=_TOOL_CAPS,
        responses=[_call("list_dir", path="."), _call("list_dir", path=".")],
    )
    outcome = Coder(provider, build_default_registry(), settings).run_task(
        CoderTask(description="loop forever"), _ctx(tmp_path)
    )
    assert outcome.status == "budget_exceeded"
    assert outcome.steps == 2


def test_coder_no_progress_when_not_calling_tools(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, coder=CoderSettings(no_progress_limit=2))
    provider = FakeProvider(
        capabilities=_TOOL_CAPS,
        responses=[ChatResponse(content="I am thinking...") for _ in range(6)],
    )
    outcome = Coder(provider, build_default_registry(), settings).run_task(
        CoderTask(description="do nothing"), _ctx(tmp_path)
    )
    assert outcome.status == "no_progress"


def test_extract_tool_calls_from_content_envelope(tmp_path: Path) -> None:
    registry = build_default_registry()
    # Native tool_calls preferred.
    native = _extract_tool_calls("", [ToolCall(id="1", name="list_dir", arguments={})], registry)
    assert native[0].name == "list_dir"
    # Envelope parsed from content when native is empty.
    parsed = _extract_tool_calls(
        '{"name": "read_file", "arguments": {"path": "a.py"}}', [], registry
    )
    assert parsed and parsed[0].name == "read_file"
    # Unknown tool names are ignored.
    assert _extract_tool_calls('{"name": "nope", "arguments": {}}', [], registry) == []


def test_workspace_signature_changes_with_content(tmp_path: Path) -> None:
    sig0 = _workspace_signature(tmp_path)
    (tmp_path / "f.py").write_text("x=1\n", encoding="utf-8")
    sig1 = _workspace_signature(tmp_path)
    assert sig0 != sig1
