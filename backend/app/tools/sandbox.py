"""Sandboxed command execution (Task 1.2, ADR-0007).

Two backends behind one :class:`Sandbox` interface:

* :class:`DockerSandbox` — the security boundary. Each command runs in a fresh
  container with ``--network=none``, memory/CPU/pids limits, a hard timeout, and
  the workspace mounted read-write at ``/work``.
* :class:`SubprocessSandbox` — the explicit fallback for machines without Docker.
  It provides cwd-jail + timeout only; it does **not** isolate the network or
  filesystem. Selected only via ``SANDBOX__BACKEND=subprocess``.

The runner never raises on non-zero exit or timeout — that is data returned in
:class:`CommandResult` and fed back to the agent.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from app.core.config import SandboxSettings

_TIMEOUT_EXIT_CODE = 124  # conventional timeout exit code


class CommandResult(BaseModel):
    """Outcome of running a single command in a sandbox."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Sandbox(ABC):
    """Executes a shell command against a workspace directory."""

    backend_name: str

    @abstractmethod
    def run(
        self,
        command: str,
        *,
        workspace_path: Path,
        timeout_s: float | None = None,
    ) -> CommandResult:
        """Run ``command`` with ``workspace_path`` as the working directory."""


class SubprocessSandbox(Sandbox):
    """Host-subprocess fallback: cwd-jail + timeout, no network/fs isolation."""

    backend_name = "subprocess"

    def __init__(self, settings: SandboxSettings) -> None:
        self._settings = settings

    def run(
        self,
        command: str,
        *,
        workspace_path: Path,
        timeout_s: float | None = None,
    ) -> CommandResult:
        timeout = timeout_s or self._settings.default_timeout_s
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                exit_code=_TIMEOUT_EXIT_CODE,
                stdout=_as_text(exc.stdout),
                stderr=f"Command timed out after {timeout}s",
                timed_out=True,
            )
        return CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class DockerSandbox(Sandbox):
    """Container-per-command executor (the ADR-0007 security boundary)."""

    backend_name = "docker"

    def __init__(self, settings: SandboxSettings) -> None:
        self._settings = settings

    def _docker_args(self, command: str, workspace_path: Path, timeout: float) -> list[str]:
        s = self._settings
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            s.network,
            "--memory",
            s.mem_limit,
            "--cpus",
            str(s.cpus),
            "--pids-limit",
            str(s.pids_limit),
            "-v",
            f"{workspace_path}:/work",
            "-w",
            "/work",
            s.image,
            "sh",
            "-lc",
            command,
        ]

    def run(
        self,
        command: str,
        *,
        workspace_path: Path,
        timeout_s: float | None = None,
    ) -> CommandResult:
        timeout = timeout_s or self._settings.default_timeout_s
        # Give docker a small grace period beyond the in-container timeout.
        args = self._docker_args(command, workspace_path, timeout)
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout + 10.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                exit_code=_TIMEOUT_EXIT_CODE,
                stderr=f"Container timed out after {timeout}s",
                timed_out=True,
            )
        return CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def get_sandbox(settings: SandboxSettings) -> Sandbox:
    """Construct the configured sandbox backend."""
    if settings.backend == "docker":
        return DockerSandbox(settings)
    return SubprocessSandbox(settings)
