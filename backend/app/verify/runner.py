"""Deterministic verify runner (Task 1.9, ADR-0005).

No LLM. Auto-detects a project's checks from files present and runs them in the
sandbox, returning a structured :class:`VerifyResult`. A timeout counts as a
failure (guards infinite loops). Output is truncated head+tail to protect the
context budget.

``VerifyResult``/``CheckResult`` are the canonical graph-state shapes defined in
``app.graph.state`` (``ARCHITECTURE.md §5``); this module re-exports them so
Phase 1 call sites are unaffected by the Phase 2 reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import CoderSettings
from app.graph.state import CheckResult, VerifyResult
from app.tools.authorization import truncate_output
from app.tools.sandbox import Sandbox

__all__ = ["Check", "CheckResult", "VerifyResult", "VerifyRunner", "detect_checks"]

_IGNORE_DIRS = frozenset({".git", "__pycache__", ".venv", "venv", "node_modules"})


@dataclass(frozen=True)
class Check:
    name: str
    cmd: str


def detect_checks(workspace_path: Path) -> list[Check]:
    """Infer the applicable checks from the files in the workspace."""
    checks: list[Check] = []
    py_files = [
        p
        for p in workspace_path.rglob("*.py")
        if not any(part in _IGNORE_DIRS for part in p.relative_to(workspace_path).parts)
    ]
    if py_files:
        checks.append(Check("compile", "python -m compileall -q ."))
        has_tests = (
            (workspace_path / "tests").is_dir()
            or (workspace_path / "conftest.py").exists()
            or any(p.name.startswith("test_") or p.name.endswith("_test.py") for p in py_files)
        )
        if has_tests:
            checks.append(Check("pytest", "python -m pytest -q"))
    if (workspace_path / "package.json").exists():
        checks.append(Check("npm-test", "npm test --silent"))
    return checks


class VerifyRunner:
    """Runs detected checks in the sandbox and aggregates results."""

    def __init__(self, sandbox: Sandbox, settings: CoderSettings) -> None:
        self._sandbox = sandbox
        self._tail = settings.output_tail_chars
        self._timeout = settings.check_timeout_s

    def run(self, workspace_path: Path) -> VerifyResult:
        checks = detect_checks(workspace_path)
        if not checks:
            return VerifyResult(passed=True, checks=[], summary="no applicable checks detected")

        results: list[CheckResult] = []
        for check in checks:
            outcome = self._sandbox.run(
                check.cmd, workspace_path=workspace_path, timeout_s=self._timeout
            )
            results.append(
                CheckResult(
                    name=check.name,
                    cmd=check.cmd,
                    passed=outcome.ok,
                    exit_code=outcome.exit_code,
                    stdout_tail=truncate_output(outcome.stdout, self._tail),
                    stderr_tail=truncate_output(outcome.stderr, self._tail),
                )
            )

        passed = all(r.passed for r in results)
        failed = [r.name for r in results if not r.passed]
        summary = "all checks passed" if passed else f"failed: {', '.join(failed)}"
        return VerifyResult(passed=passed, checks=results, summary=summary)
