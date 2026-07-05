"""Filesystem tools (Task 1.4).

``read_file``, ``write_file``, ``edit_file`` (exact search-replace) and
``list_dir``. All paths are resolved through the workspace path-jail; these
tools act directly on workspace files on the host (the sandbox is only for
command execution).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolContext, ToolResult
from app.tools.security import resolve_within

_MAX_LIST_ENTRIES = 500


class ReadFileArgs(BaseModel):
    path: str = Field(description="Workspace-relative path to read.")
    start_line: int | None = Field(default=None, description="1-indexed first line (inclusive).")
    end_line: int | None = Field(default=None, description="1-indexed last line (inclusive).")


class ReadFile(Tool[ReadFileArgs]):
    name = "read_file"
    description = "Read a UTF-8 text file, optionally a 1-indexed inclusive line range."
    args_schema = ReadFileArgs

    def run(self, args: ReadFileArgs, ctx: ToolContext) -> ToolResult:
        target = resolve_within(ctx.workspace_path, args.path)
        if not target.is_file():
            return ToolResult.failure(f"not a file: {args.path}")
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(f"file is not UTF-8 text: {args.path}")
        if args.start_line is not None or args.end_line is not None:
            lines = text.splitlines()
            start = (args.start_line or 1) - 1
            end = args.end_line if args.end_line is not None else len(lines)
            text = "\n".join(lines[start:end])
        return ToolResult.success(output=text, path=args.path)


class WriteFileArgs(BaseModel):
    path: str = Field(description="Workspace-relative path to write (created if absent).")
    content: str = Field(description="Full file contents.")


class WriteFile(Tool[WriteFileArgs]):
    name = "write_file"
    description = "Create or overwrite a text file with the given contents."
    args_schema = WriteFileArgs

    def run(self, args: WriteFileArgs, ctx: ToolContext) -> ToolResult:
        target = resolve_within(ctx.workspace_path, args.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        return ToolResult.success(
            output=f"wrote {len(args.content)} chars to {args.path}", path=args.path
        )


class EditFileArgs(BaseModel):
    path: str = Field(description="Workspace-relative path to edit.")
    old_string: str = Field(description="Exact text to replace (must exist).")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace every occurrence.")


class EditFile(Tool[EditFileArgs]):
    name = "edit_file"
    description = (
        "Replace an exact substring in a file. Fails if the target text is absent, "
        "or matches more than once unless replace_all is set."
    )
    args_schema = EditFileArgs

    def run(self, args: EditFileArgs, ctx: ToolContext) -> ToolResult:
        target = resolve_within(ctx.workspace_path, args.path)
        if not target.is_file():
            return ToolResult.failure(f"not a file: {args.path}")
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(f"file is not UTF-8 text: {args.path}")

        count = text.count(args.old_string)
        if count == 0:
            return ToolResult.failure(f"old_string not found in {args.path}")
        if count > 1 and not args.replace_all:
            return ToolResult.failure(
                f"old_string matches {count} times in {args.path}; "
                "pass replace_all=true or provide a more specific string"
            )
        updated = text.replace(args.old_string, args.new_string)
        target.write_text(updated, encoding="utf-8")
        return ToolResult.success(
            output=f"edited {args.path} ({count} replacement(s))", path=args.path
        )


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Workspace-relative directory (default root).")


class ListDir(Tool[ListDirArgs]):
    name = "list_dir"
    description = "List the entries of a directory (directories marked with a trailing '/')."
    args_schema = ListDirArgs

    def run(self, args: ListDirArgs, ctx: ToolContext) -> ToolResult:
        target = resolve_within(ctx.workspace_path, args.path)
        if not target.is_dir():
            return ToolResult.failure(f"not a directory: {args.path}")
        entries = sorted(
            f"{child.name}/" if child.is_dir() else child.name for child in target.iterdir()
        )
        truncated = entries[:_MAX_LIST_ENTRIES]
        output = "\n".join(truncated) if truncated else "(empty)"
        return ToolResult.success(output=output, count=len(entries))
