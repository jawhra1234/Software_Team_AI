"""Task 1.2 — sandbox executors: subprocess (hermetic) + docker (integration).

Security assertions (ADR-0007): network egress blocked (docker), command
timeout enforced (both). Path-jail and command allow/deny-list live in the
authorization layer (Task 1.3) and are tested there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from app.core.config import SandboxSettings
from app.tools.sandbox import DockerSandbox, SubprocessSandbox, get_sandbox

from tests.conftest import docker_available


# ---------------------------------------------------------------------------
# Subprocess backend — hermetic
# ---------------------------------------------------------------------------
def test_get_sandbox_selects_backend() -> None:
    assert get_sandbox(SandboxSettings(backend="subprocess")).backend_name == "subprocess"
    assert get_sandbox(SandboxSettings(backend="docker")).backend_name == "docker"


def test_subprocess_runs_and_captures(tmp_path: Path) -> None:
    sb = SubprocessSandbox(SandboxSettings(backend="subprocess"))
    result = sb.run(f'"{sys.executable}" -c "print(2+2)"', workspace_path=tmp_path)
    assert result.ok
    assert result.stdout.strip() == "4"


def test_subprocess_reports_nonzero_exit(tmp_path: Path) -> None:
    sb = SubprocessSandbox(SandboxSettings(backend="subprocess"))
    result = sb.run(f'"{sys.executable}" -c "import sys; sys.exit(3)"', workspace_path=tmp_path)
    assert not result.ok
    assert result.exit_code == 3


def test_subprocess_timeout(tmp_path: Path) -> None:
    sb = SubprocessSandbox(SandboxSettings(backend="subprocess", default_timeout_s=1.0))
    result = sb.run(
        f'"{sys.executable}" -c "import time; time.sleep(5)"',
        workspace_path=tmp_path,
        timeout_s=1.0,
    )
    assert result.timed_out
    assert not result.ok


# ---------------------------------------------------------------------------
# Docker backend — integration (requires the daemon + built sandbox image)
# ---------------------------------------------------------------------------
pytest_docker = pytest.mark.skipif(not docker_available(), reason="docker daemon not available")


@pytest.mark.integration
@pytest_docker
def test_docker_runs_command(tmp_path: Path) -> None:
    sb = DockerSandbox(SandboxSettings(backend="docker"))
    result = sb.run("python -c 'print(6*7)'", workspace_path=tmp_path)
    assert result.ok, result.stderr
    assert result.stdout.strip() == "42"


@pytest.mark.integration
@pytest_docker
def test_docker_network_egress_blocked(tmp_path: Path) -> None:
    sb = DockerSandbox(SandboxSettings(backend="docker", network="none"))
    result = sb.run(
        "python -c \"import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)\"",
        workspace_path=tmp_path,
    )
    assert not result.ok  # no network → the request must fail


@pytest.mark.integration
@pytest_docker
def test_docker_timeout(tmp_path: Path) -> None:
    sb = DockerSandbox(SandboxSettings(backend="docker", default_timeout_s=2.0))
    result = sb.run("sleep 30", workspace_path=tmp_path, timeout_s=2.0)
    assert result.timed_out
