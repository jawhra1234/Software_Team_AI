"""Task 3.13 — planner memory context assembler (hermetic)."""

from __future__ import annotations

from app.core.config import PlannerSettings
from app.graph.planning_context import build_planner_context
from app.memory.episodic import RunRecord
from app.memory.long_term import MemoryItem


class _FakeLongTerm:
    def __init__(self, items: list[MemoryItem]) -> None:
        self._items = items
        self.last_k: int | None = None

    def search(self, project_id: str, query: str, k: int = 5) -> list[MemoryItem]:
        self.last_k = k
        return self._items[:k]


class _FakeEpisodic:
    def __init__(self, runs: list[RunRecord]) -> None:
        self._runs = runs
        self.last_k: int | None = None

    def relevant(
        self, project_id: str, query: str, k: int = 3, *, candidate_window: int = 50
    ) -> list[RunRecord]:
        self.last_k = k
        return self._runs[:k]


class _BoomLongTerm:
    def search(self, project_id: str, query: str, k: int = 5) -> list[MemoryItem]:
        raise RuntimeError("db down")


def _item(text: str) -> MemoryItem:
    return MemoryItem(kind="decision", text=text)


def _run(status: str, summary: str) -> RunRecord:
    return RunRecord(run_id="r", project_id="p", status=status, summary=summary,
                     tasks_total=2, tasks_done=1)


_S = PlannerSettings()


def _build(long_term: object, episodic: object, settings: PlannerSettings = _S) -> str:
    return build_planner_context(
        "add a checkout total", "p", long_term=long_term, episodic=episodic, settings=settings  # type: ignore[arg-type]
    )


def test_both_sections_present() -> None:
    block = _build(
        _FakeLongTerm([_item("Use pnpm for packages.")]),
        _FakeEpisodic([_run("failed", "verify failed on calc.py")]),
    )
    assert "=== Project Conventions" in block and "Use pnpm" in block
    assert "=== Previous Attempts" in block and "verify failed on calc.py" in block
    assert "[failed]" in block  # status surfaced


def test_long_term_only() -> None:
    block = _build(_FakeLongTerm([_item("convention x")]), None)
    assert "Project Conventions" in block
    assert "Previous Attempts" not in block


def test_episodic_only() -> None:
    block = _build(None, _FakeEpisodic([_run("succeeded", "did the thing")]))
    assert "Previous Attempts" in block
    assert "Project Conventions" not in block


def test_all_none_is_empty() -> None:
    assert _build(None, None) == ""


def test_empty_results_is_empty() -> None:
    assert _build(_FakeLongTerm([]), _FakeEpisodic([])) == ""


def test_read_failure_degrades_to_empty() -> None:
    # A memory backend error must not raise — the section is simply omitted.
    block = _build(_BoomLongTerm(), _FakeEpisodic([_run("failed", "boom context")]))
    assert "Project Conventions" not in block
    assert "Previous Attempts" in block  # the healthy source still renders


def test_per_item_truncation() -> None:
    block = _build(_FakeLongTerm([_item("x" * 500)]), None)
    assert "…" in block  # long item truncated
    assert "x" * 500 not in block


def test_section_char_cap_drops_overflow() -> None:
    settings = PlannerSettings(memory_max_section_chars=80)
    items = [_item(f"convention number {i} " + "y" * 40) for i in range(10)]
    block = _build(_FakeLongTerm(items), None, settings)
    # Cap keeps at least one bullet but far fewer than 10.
    assert block.count("- convention number") < 10
    assert len(block) < 200


def test_k_comes_from_settings() -> None:
    settings = PlannerSettings(memory_long_term_k=2, memory_episodic_k=1)
    lt = _FakeLongTerm([_item("a"), _item("b"), _item("c")])
    ep = _FakeEpisodic([_run("failed", "one"), _run("failed", "two")])
    _build(lt, ep, settings)
    assert lt.last_k == 2 and ep.last_k == 1
