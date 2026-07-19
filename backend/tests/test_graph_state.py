"""Task 2.1 — state reducers and the "no raw content in state" invariant."""

from __future__ import annotations

from app.graph.reducers import merge_by_path, merge_counts
from app.graph.state import CheckResult, FileRef, Plan, Task, VerifyResult


def test_merge_by_path_latest_write_wins() -> None:
    existing = [FileRef(path="a.py", status="added"), FileRef(path="b.py", status="modified")]
    updates = [FileRef(path="a.py", status="modified")]
    merged = merge_by_path(existing, updates)
    by_path = {f.path: f.status for f in merged}
    assert by_path == {"a.py": "modified", "b.py": "modified"}


def test_merge_by_path_first_call_from_empty() -> None:
    merged = merge_by_path([], [FileRef(path="a.py", status="added")])
    assert len(merged) == 1 and merged[0].path == "a.py"


def test_merge_counts_sums_per_key() -> None:
    assert merge_counts({}, {"verify": 1}) == {"verify": 1}
    assert merge_counts({"verify": 1}, {"verify": 1}) == {"verify": 2}
    assert merge_counts({"verify": 1}, {"review": 1}) == {"verify": 1, "review": 1}


def test_merge_counts_does_not_mutate_existing() -> None:
    existing = {"verify": 1}
    merge_counts(existing, {"verify": 1})
    assert existing == {"verify": 1}  # caller's dict untouched


# ---------------------------------------------------------------------------
# Invariant: no field holds raw file contents or un-truncated output.
# ---------------------------------------------------------------------------
def test_file_ref_has_no_content_field() -> None:
    assert set(FileRef.model_fields) == {"path", "status", "blob_sha"}


def test_check_result_only_carries_tails() -> None:
    fields = set(CheckResult.model_fields)
    assert "stdout_tail" in fields and "stderr_tail" in fields
    assert "stdout" not in fields and "stderr" not in fields


def test_verify_result_has_no_raw_output_field() -> None:
    assert set(VerifyResult.model_fields) == {"passed", "checks", "summary"}


# ---------------------------------------------------------------------------
# StrList coercion — tolerate a model emitting list-of-objects for list[str].
# ---------------------------------------------------------------------------
def test_task_coerces_object_acceptance_criteria_to_strings() -> None:
    # Exactly the shape qwen2.5-coder emitted that failed validation before.
    task = Task(
        id="task-1",
        title="add calc",
        description="d",
        kind="create",
        acceptance_criteria=[
            {"type": "file_exists", "path": "calc.py"},  # type: ignore[list-item]
            "calc.py defines add(a, b)",
        ],
    )
    assert all(isinstance(c, str) for c in task.acceptance_criteria)
    assert '"file_exists"' in task.acceptance_criteria[0]
    assert task.acceptance_criteria[1] == "calc.py defines add(a, b)"


def test_task_coerces_offenum_kind_to_modify() -> None:
    # Exactly the failure qwen2.5-coder produced live: kind set to a tool name.
    task = Task(id="t", title="t", description="d", kind="retrieve")  # type: ignore[arg-type]
    assert task.kind == "modify"


def test_task_kind_is_case_insensitive_and_valid_passes() -> None:
    assert Task(id="t", title="t", description="d", kind="CREATE").kind == "create"  # type: ignore[arg-type]
    assert Task(id="t", title="t", description="d", kind="fix").kind == "fix"


def test_plan_coerces_object_requirements_to_strings() -> None:
    plan = Plan(
        summary="s",
        functional_requirements=[{"must": "add two numbers"}],  # type: ignore[list-item]
    )
    assert isinstance(plan.functional_requirements[0], str)


def test_str_list_passes_through_plain_strings() -> None:
    task = Task(
        id="t", title="t", description="d", kind="create",
        acceptance_criteria=["a", "b"],
    )
    assert task.acceptance_criteria == ["a", "b"]
