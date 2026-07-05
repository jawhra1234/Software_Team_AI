"""Phase-1 end-to-end acceptance (Task 1.8 / DoD).

Live: the coder builds a running, test-passing tiny project entirely inside the
sandbox. Requires Ollama (qwen2.5-coder) + Docker (sandbox image). Marked
integration and skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.agents.coder import Coder, CoderTask
from app.core.config import CoderSettings, Settings
from app.providers.factory import get_provider
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry
from app.tools.sandbox import get_sandbox
from app.verify.runner import VerifyRunner
from app.workspace.lifecycle import WorkspaceManager

from tests.conftest import docker_available, ollama_available

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not (ollama_available() and docker_available()),
    reason="requires live Ollama and Docker",
)
def test_e2e_coder_builds_passing_project(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  
        coder=CoderSettings(max_steps_per_task=15, max_wall_clock_s=600.0),
    )
    provider = get_provider("coder", settings)
    registry = build_default_registry()
    sandbox = get_sandbox(settings.sandbox)  # docker (ADR-0007)

    mgr = WorkspaceManager(tmp_path / "ws")
    ws = mgr.create("e2e")
    mgr.start_run(ws, "r1")
    ctx = ToolContext(workspace_path=ws.path, run_id="r1", sandbox=sandbox, workspace=ws)

    task = CoderTask(
        description=(
            "Create a Python module calc.py with a function add(a, b) that returns a + b, "
            "and a pytest test file test_calc.py that asserts add(2, 3) == 5."
        ),
        acceptance_criteria=["calc.py defines add(a, b)", "running pytest passes"],
        target_paths=["calc.py", "test_calc.py"],
    )
    outcome = Coder(provider, registry, settings).run_task(task, ctx)

    result = VerifyRunner(sandbox, settings.coder).run(ws.path)
    files = sorted(p.name for p in ws.path.iterdir())
    assert result.passed, f"outcome={outcome}, verify={result.summary}, files={files}"
