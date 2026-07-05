"""Task 1.1 — tool protocol, result helpers, registry, LangChain adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, to_langchain_tool
from pydantic import BaseModel


class EchoArgs(BaseModel):
    text: str


class EchoTool(Tool[EchoArgs]):
    name = "echo"
    description = "Echo the given text."
    args_schema = EchoArgs

    def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.success(output=args.text, echoed=True)


def _ctx() -> ToolContext:
    return ToolContext(workspace_path=Path("."), run_id="t")


def test_result_helpers() -> None:
    ok = ToolResult.success("hi", n=1)
    assert ok.ok and ok.output == "hi" and ok.meta["n"] == 1
    bad = ToolResult.failure("boom", output="partial")
    assert not bad.ok and bad.error == "boom" and bad.output == "partial"


def test_run_and_spec() -> None:
    tool = EchoTool()
    result = tool.run(EchoArgs(text="yo"), _ctx())
    assert result.output == "yo"
    spec = tool.spec()
    assert spec.name == "echo"
    assert "text" in spec.parameters["properties"]


def test_registry_register_get_specs() -> None:
    reg = ToolRegistry()
    reg.register(EchoTool())
    assert "echo" in reg
    assert reg.names() == ["echo"]
    assert reg.get("echo").name == "echo"
    assert reg.specs()[0].name == "echo"


def test_registry_rejects_duplicate_and_unknown() -> None:
    reg = ToolRegistry()
    reg.register(EchoTool())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(EchoTool())
    with pytest.raises(KeyError):
        reg.get("nope")


def test_to_langchain_tool_adapter() -> None:
    lc_tool = to_langchain_tool(EchoTool(), _ctx())
    assert lc_tool.name == "echo"
    # StructuredTool is invokable and routes through our tool.
    assert lc_tool.invoke({"text": "adapted"}) == "adapted"
