"""Task 1.3 — authorization pipeline: validate, path-jail, command policy, truncate."""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.tools.authorization import AuthorizationPolicy, execute_tool, truncate_output
from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from app.tools.security import PathJailError, resolve_within
from pydantic import BaseModel


# --- dummy tools -----------------------------------------------------------
class PathArgs(BaseModel):
    path: str


class ReadDummy(Tool[PathArgs]):
    name = "read_dummy"
    description = "reads a path"
    args_schema = PathArgs

    def run(self, args: PathArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.success(output=f"read {args.path}")


class CommandArgs(BaseModel):
    command: str


class RunDummy(Tool[CommandArgs]):
    name = "run_dummy"
    description = "runs a command"
    args_schema = CommandArgs
    requires_approval = True

    def run(self, args: CommandArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.success(output=f"ran {args.command}")


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ReadDummy())
    reg.register(RunDummy())
    return reg


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path, run_id="t")


def _policy() -> AuthorizationPolicy:
    return AuthorizationPolicy.from_settings(Settings(_env_file=None))


# --- path-jail (unit) ------------------------------------------------------
def test_resolve_within_accepts_and_rejects(tmp_path: Path) -> None:
    assert resolve_within(tmp_path, "sub/file.py").is_relative_to(tmp_path.resolve())
    import pytest

    with pytest.raises(PathJailError):
        resolve_within(tmp_path, "../../etc/passwd")


# --- pipeline --------------------------------------------------------------
def test_unknown_tool_returns_failure(tmp_path: Path) -> None:
    result = execute_tool(_registry(), "nope", {}, _ctx(tmp_path), _policy())
    assert not result.ok


def test_schema_validation_failure(tmp_path: Path) -> None:
    result = execute_tool(_registry(), "read_dummy", {"wrong": 1}, _ctx(tmp_path), _policy())
    assert not result.ok
    assert "invalid arguments" in (result.error or "")


def test_path_jail_blocks_traversal(tmp_path: Path) -> None:
    result = execute_tool(
        _registry(), "read_dummy", {"path": "../../secrets"}, _ctx(tmp_path), _policy()
    )
    assert not result.ok
    assert "outside the workspace" in (result.error or "")


def test_command_allowlist(tmp_path: Path) -> None:
    reg, ctx, policy = _registry(), _ctx(tmp_path), _policy()
    ok = execute_tool(reg, "run_dummy", {"command": "python -m pytest"}, ctx, policy)
    assert ok.ok
    blocked = execute_tool(reg, "run_dummy", {"command": "make build"}, ctx, policy)
    assert not blocked.ok
    assert "allowlist" in (blocked.error or "")


def test_command_denylist(tmp_path: Path) -> None:
    result = execute_tool(
        _registry(), "run_dummy", {"command": "python x.py; rm -rf /"}, _ctx(tmp_path), _policy()
    )
    assert not result.ok
    assert "denied pattern" in (result.error or "")


def test_approval_hook_denies_in_semi(tmp_path: Path) -> None:
    policy = AuthorizationPolicy.from_settings(Settings(_env_file=None), autonomy="semi")
    result = execute_tool(
        _registry(),
        "run_dummy",
        {"command": "python x.py"},
        _ctx(tmp_path),
        policy,
        approve=lambda _name, _args: False,
    )
    assert not result.ok
    assert "approval denied" in (result.error or "")


def test_approval_hook_also_gates_in_manual(tmp_path: Path) -> None:
    # ADR-0009: manual is the strictest level, so it must gate commands too.
    policy = AuthorizationPolicy.from_settings(Settings(_env_file=None), autonomy="manual")
    result = execute_tool(
        _registry(),
        "run_dummy",
        {"command": "python x.py"},
        _ctx(tmp_path),
        policy,
        approve=lambda _name, _args: False,
    )
    assert not result.ok
    assert "approval denied" in (result.error or "")


def test_approval_hook_not_gated_in_auto(tmp_path: Path) -> None:
    policy = AuthorizationPolicy.from_settings(Settings(_env_file=None), autonomy="auto")
    result = execute_tool(
        _registry(),
        "run_dummy",
        {"command": "python x.py"},
        _ctx(tmp_path),
        policy,
        approve=lambda _name, _args: False,  # would deny, but auto never asks
    )
    assert result.ok


def test_truncate_output() -> None:
    text = "x" * 100
    out = truncate_output(text, 20)
    assert "truncated" in out
    assert len(out) < len(text) + 40
    assert truncate_output("short", 100) == "short"
