"""Tasks 2.4/2.6 — verify, finalize nodes (no LLM, hermetic).

The review node (Task 2.5's stub, replaced by the real reviewer in Phase 4) is
LLM-backed as of Phase 4 and has its own hermetic suite driven by a
``FakeProvider``: see ``test_graph_nodes_review.py``.
"""

from __future__ import annotations

from pathlib import Path

from app.core.clock import now_iso
from app.core.config import CoderSettings, GraphSettings, Settings
from app.graph.nodes.finalize import make_finalize_node
from app.graph.nodes.verify import make_verify_node
from app.graph.state import new_run_state
from app.tools.git import Git
from app.tools.sandbox import SubprocessSandbox


def _base_state(workspace_path: Path, **overrides: object) -> dict[str, object]:
    state = new_run_state(
        run_id="r1",
        project_id="p1",
        user_request="do it",
        workspace_path=str(workspace_path),
        autonomy_level="auto",
        max_tokens=None,
        max_steps=50,
        max_wall_clock_s=3600,
        started_at=now_iso(),
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state  # type: ignore[return-value]


def _git_repo(tmp_path: Path) -> Git:
    git = Git(tmp_path)
    git.init()
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    return git


# ---------------------------------------------------------------------------
# verify node (2.4)
# ---------------------------------------------------------------------------
def test_verify_node_passes_and_clears_hitl(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
    )
    sandbox = SubprocessSandbox(Settings(_env_file=None).sandbox.model_copy(update={"backend": "subprocess"}))
    node = make_verify_node(sandbox, GraphSettings(), CoderSettings())
    patch = node(_base_state(tmp_path))
    assert patch["verify_result"].passed
    assert patch["hitl_request"] is None
    assert "retries" not in patch


def test_verify_node_fails_and_increments_retries(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
    )
    sandbox = SubprocessSandbox(Settings(_env_file=None).sandbox.model_copy(update={"backend": "subprocess"}))
    node = make_verify_node(sandbox, GraphSettings(max_verify_retries=3), CoderSettings())
    patch = node(_base_state(tmp_path, retries={"verify": 0}))
    assert not patch["verify_result"].passed
    assert patch["retries"] == {"verify": 1}
    assert patch["hitl_request"] is None  # 1 < 3, no escalation yet


def test_verify_node_escalates_when_retries_exhausted(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
    )
    sandbox = SubprocessSandbox(Settings(_env_file=None).sandbox.model_copy(update={"backend": "subprocess"}))
    node = make_verify_node(sandbox, GraphSettings(max_verify_retries=2), CoderSettings())
    patch = node(_base_state(tmp_path, retries={"verify": 1}))
    assert patch["retries"] == {"verify": 1}
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "escalation"


# ---------------------------------------------------------------------------
# finalize node (2.6)
# ---------------------------------------------------------------------------
def test_finalize_defaults_to_succeeded(tmp_path: Path) -> None:
    git = _git_repo(tmp_path)
    base = git.commit("base")
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    git.commit("change")
    node = make_finalize_node()
    patch = node(_base_state(tmp_path, base_commit=base))
    assert patch["status"] == "succeeded"
    assert "x = 2" in patch["diff_summary"]
    assert patch["hitl_request"] is None


def test_finalize_preserves_prior_abort_status(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    node = make_finalize_node()
    for prior in ("cancelled", "failed"):
        patch = node(_base_state(tmp_path, status=prior))
        assert patch["status"] == prior


def test_finalize_no_changes_summary(tmp_path: Path) -> None:
    git = _git_repo(tmp_path)
    base = git.commit("base")
    node = make_finalize_node()
    patch = node(_base_state(tmp_path, base_commit=base))
    assert patch["diff_summary"] == "(no changes)"
