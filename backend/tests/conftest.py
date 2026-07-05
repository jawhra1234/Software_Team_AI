"""Shared test fixtures and helpers."""

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request

import pytest
from app.core.config import Settings


def ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """True if a local Ollama server responds (used to gate integration tests)."""
    try:
        with urllib.request.urlopen(base_url, timeout=1.0):
            return True
    except (urllib.error.URLError, OSError):
        return False


def docker_available() -> bool:
    """True if a Docker daemon is reachable (used to gate sandbox integration tests)."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def settings() -> Settings:
    """Default settings built without reading a developer's local .env file."""
    return Settings(_env_file=None)
