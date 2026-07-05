"""Code search tool (Task 1.5).

Text/regex search over the workspace — the keyword-search arm of ADR-0008. Uses
ripgrep when available (fast, .gitignore-aware) and falls back to a pure-Python
walker otherwise, so the tool works on any host. The tree-sitter symbol index is
deferred to Phase 3.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolContext, ToolResult
from app.tools.security import resolve_within

_RG_NO_MATCH = 1
_SEARCH_TIMEOUT_S = 20.0
_IGNORE_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


class SearchCodeArgs(BaseModel):
    query: str = Field(description="Text (or regex if is_regex) to search for.")
    path: str = Field(default=".", description="Workspace-relative directory to search.")
    is_regex: bool = Field(default=False, description="Treat query as a regex.")
    case_insensitive: bool = Field(default=False, description="Case-insensitive match.")
    max_results: int = Field(default=50, description="Maximum number of matching lines to return.")


class SearchCode(Tool[SearchCodeArgs]):
    name = "search_code"
    description = "Search workspace files for a string or regex (ripgrep when available)."
    args_schema = SearchCodeArgs

    def run(self, args: SearchCodeArgs, ctx: ToolContext) -> ToolResult:
        target = resolve_within(ctx.workspace_path, args.path)
        base = ctx.workspace_path.resolve()
        rg = shutil.which("rg") or shutil.which("rg.exe")
        matches = (
            self._ripgrep(rg, args, target)
            if rg is not None
            else _python_search(base, target, args)
        )
        if matches is None:
            return ToolResult.failure("search failed")
        total = len(matches)
        clipped = matches[: args.max_results]
        note = "" if total <= args.max_results else f"\n...[{total - args.max_results} more]"
        output = ("\n".join(clipped) + note) if clipped else "(no matches)"
        return ToolResult.success(output=output, matches=total)

    def _ripgrep(self, rg: str, args: SearchCodeArgs, target: Path) -> list[str] | None:
        cmd = [rg, "--line-number", "--no-heading", "--color", "never"]
        if not args.is_regex:
            cmd.append("--fixed-strings")
        if args.case_insensitive:
            cmd.append("--ignore-case")
        cmd.extend([args.query, str(target)])
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_SEARCH_TIMEOUT_S, check=False
            )
        except subprocess.TimeoutExpired:
            return None
        if completed.returncode == _RG_NO_MATCH:
            return []
        if completed.returncode != 0:
            return None
        base = str(target.resolve())
        return [
            line.replace(base + "\\", "").replace(base + "/", "")
            for line in completed.stdout.splitlines()
        ]


def _python_search(root: Path, target: Path, args: SearchCodeArgs) -> list[str]:
    flags = re.IGNORECASE if args.case_insensitive else 0
    pattern = re.compile(args.query if args.is_regex else re.escape(args.query), flags)
    results: list[str] = []
    for path in sorted(target.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in _IGNORE_DIRS for part in rel_parts):
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                results.append(f"{rel}:{lineno}:{line}")
    return results
