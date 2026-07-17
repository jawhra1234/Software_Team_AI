"""Tasks 2.4/2.5/2.6 — verify, review-stub, finalize nodes (no LLM, hermetic)."""

from __future__ import annotations

from pathlib import Path

from app.core.clock import now_iso
from app.core.config import CoderSettings, GraphSettings, Settings
from app.graph.nodes.finalize import make_finalize_node
from app.graph.nodes.review_stub import make_review_node
from app.graph.nodes.verify import make_verify_node
from app.graph.state import FileRef, new_run_state
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
# review-stub node (2.5)
# ---------------------------------------------------------------------------
def test_review_stub_approves_when_files_changed(tmp_path: Path) -> None:
    node = make_review_node(GraphSettings())
    state = _base_state(tmp_path, changed_files=[FileRef(path="a.py", status="modified")])
    patch = node(state)
    assert patch["review"].verdict == "approved"
    assert patch["hitl_request"] is None


def test_review_stub_requests_changes_when_nothing_changed(tmp_path: Path) -> None:
    node = make_review_node(GraphSettings(max_review_cycles=3))
    patch = node(_base_state(tmp_path, changed_files=[], retries={"review": 0}))
    assert patch["review"].verdict == "changes_requested"
    assert patch["retries"] == {"review": 1}
    assert patch["hitl_request"] is None


def test_review_stub_escalates_when_cycles_exhausted(tmp_path: Path) -> None:
    node = make_review_node(GraphSettings(max_review_cycles=1))
    patch = node(_base_state(tmp_path, changed_files=[], retries={"review": 0}))
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].payload["origin_node"] == "coder"


def test_review_stub_final_accept_gate_in_manual_autonomy(tmp_path: Path) -> None:
    node = make_review_node(GraphSettings())
    state = _base_state(
        tmp_path, changed_files=[FileRef(path="a.py", status="modified")], autonomy_level="manual"
    )
    patch = node(state)
    assert patch["review"].verdict == "approved"
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "final_accept"


def test_review_stub_no_final_accept_gate_outside_manual(tmp_path: Path) -> None:
    node = make_review_node(GraphSettings())
    for autonomy in ("auto", "semi"):
        state = _base_state(
            tmp_path,
            changed_files=[FileRef(path="a.py", status="modified")],
            autonomy_level=autonomy,
        )
        patch = node(state)
        assert patch["hitl_request"] is None


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
