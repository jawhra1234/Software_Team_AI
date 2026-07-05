"""Tasks 1.9 & 1.10 — verify runner (subprocess sandbox) and budget/loop guards."""

from __future__ import annotations

import sys
from pathlib import Path

from app.agents.budget import BudgetTracker
from app.core.config import CoderSettings, SandboxSettings
from app.tools.sandbox import SubprocessSandbox
from app.verify.runner import VerifyRunner, detect_checks


# ---------------------------------------------------------------------------
# 1.9 — verify
# ---------------------------------------------------------------------------
def _runner() -> VerifyRunner:
    return VerifyRunner(SubprocessSandbox(SandboxSettings(backend="subprocess")), CoderSettings())


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_detect_checks(tmp_path: Path) -> None:
    assert detect_checks(tmp_path) == []
    _write(tmp_path / "mod.py", "x = 1\n")
    names = {c.name for c in detect_checks(tmp_path)}
    assert "compile" in names and "pytest" not in names
    _write(tmp_path / "test_mod.py", "def test_x():\n    assert True\n")
    names = {c.name for c in detect_checks(tmp_path)}
    assert {"compile", "pytest"} <= names


def test_verify_passes_on_good_project(tmp_path: Path) -> None:
    _write(tmp_path / "calc.py", "def add(a, b):\n    return a + b\n")
    _write(
        tmp_path / "test_calc.py",
        "from calc import add\n\ndef test_add():\n    assert add(2,3)==5\n",
    )
    result = _runner().run(tmp_path)
    assert result.passed, result.summary
    assert {c.name for c in result.checks} == {"compile", "pytest"}


def test_verify_fails_on_failing_test(tmp_path: Path) -> None:
    _write(tmp_path / "calc.py", "def add(a, b):\n    return a - b\n")  # bug
    _write(
        tmp_path / "test_calc.py",
        "from calc import add\n\ndef test_add():\n    assert add(2,3)==5\n",
    )
    result = _runner().run(tmp_path)
    assert not result.passed
    pytest_check = next(c for c in result.checks if c.name == "pytest")
    assert not pytest_check.passed
    assert pytest_check.exit_code != 0


def test_verify_no_checks_is_pass(tmp_path: Path) -> None:
    _write(tmp_path / "readme.txt", "hello")
    result = _runner().run(tmp_path)
    assert result.passed and "no applicable checks" in result.summary


def test_verify_timeout_counts_as_failure(tmp_path: Path) -> None:
    # A test that sleeps beyond the (shortened) check timeout must be reported failed.
    runner = VerifyRunner(
        SubprocessSandbox(SandboxSettings(backend="subprocess")),
        CoderSettings(check_timeout_s=1.0),
    )
    _write(tmp_path / "conftest.py", "")
    _write(
        tmp_path / "test_slow.py",
        f"import time\n\ndef test_slow():\n    assert {sys.version_info.major}\n    time.sleep(10)\n",
    )
    result = runner.run(tmp_path)
    assert not result.passed
    pytest_check = next(c for c in result.checks if c.name == "pytest")
    assert not pytest_check.passed


# ---------------------------------------------------------------------------
# 1.10 — budget / loop guards
# ---------------------------------------------------------------------------
def test_step_budget() -> None:
    b = BudgetTracker(max_steps=2, max_wall_clock_s=100, no_progress_limit=3)
    b.start()
    assert b.exceeded_reason() is None
    b.tick_step()
    assert b.exceeded_reason() is None
    b.tick_step()
    assert "step budget" in (b.exceeded_reason() or "")


def test_token_budget() -> None:
    b = BudgetTracker(max_steps=100, max_wall_clock_s=100, no_progress_limit=3, max_tokens=10)
    b.start()
    b.add_tokens(5)
    assert b.exceeded_reason() is None
    b.add_tokens(6)
    assert "token budget" in (b.exceeded_reason() or "")


def test_no_progress_detection() -> None:
    b = BudgetTracker(max_steps=100, max_wall_clock_s=100, no_progress_limit=2)
    b.start()
    b.record_progress("sig-a")
    assert b.no_progress_reason() is None
    b.record_progress("sig-a")
    b.record_progress("sig-a")
    assert "no progress" in (b.no_progress_reason() or "")
    b.record_progress("sig-b")  # progress resets the counter
    assert b.no_progress_reason() is None
