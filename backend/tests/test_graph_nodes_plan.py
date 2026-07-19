"""Task 2.2 graph wiring — plan node: approval gating, clarification interrupt, failure escalation."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from app.agents.planner import Planner
from app.core.clock import now_iso
from app.core.config import PlannerSettings, Settings
from app.graph.nodes.plan import make_plan_node
from app.graph.state import AgentState, Plan, new_run_state
from app.memory.episodic import EpisodicMemory, RunRecord
from app.memory.ingest import ingest_adrs
from app.memory.long_term import LongTermMemory, MemoryItem
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.rag.embeddings import ChunkEmbedder
from app.tools.registry import build_planner_registry
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from tests.fakes import FakeProvider
from tests.test_memory import _DSN, _EP_TABLE, _LT_TABLE, _SKIP, DirEmbedProvider

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)


def _emit(payload: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="9", name=emit_tool_name(Plan), arguments=payload)]
    )


_GOOD_PLAN = {
    "summary": "Add a calculator.",
    "tasks": [
        {"id": "task-1", "title": "add calc.py", "description": "d", "kind": "create"}
    ],
}
_BLOCKING_PLAN = {
    "summary": "Need clarity.",
    "open_questions": ["Should this support floats or integers only?"],
    "tasks": [],
}


def _state(tmp_path: Path, autonomy: str = "auto") -> AgentState:
    return new_run_state(
        run_id="r1", project_id="p1", user_request="build a calculator",
        workspace_path=str(tmp_path), autonomy_level=autonomy,  # type: ignore[arg-type]
        max_tokens=None, max_steps=50, max_wall_clock_s=3600,
        started_at=now_iso(),
    )


def _planner(responses: list[ChatResponse], grounding_steps: int = 1) -> Planner:
    settings = Settings(_env_file=None, planner=PlannerSettings(grounding_steps=grounding_steps))
    provider = FakeProvider(capabilities=_CAPS, responses=responses)
    return Planner(provider, build_planner_registry(), settings)


def test_plan_node_auto_autonomy_skips_approval_gate(tmp_path: Path) -> None:
    node = make_plan_node(_planner([ChatResponse(content="no grounding"), _emit(_GOOD_PLAN)]))
    patch = node(_state(tmp_path, autonomy="auto"))
    assert patch["plan"].summary == "Add a calculator."
    assert patch["hitl_request"] is None


def test_plan_node_semi_autonomy_requests_approval(tmp_path: Path) -> None:
    node = make_plan_node(_planner([ChatResponse(content="no grounding"), _emit(_GOOD_PLAN)]))
    patch = node(_state(tmp_path, autonomy="semi"))
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "plan_approval"


def test_plan_node_empty_task_list_escalates(tmp_path: Path) -> None:
    empty_plan = {"summary": "nothing to do", "tasks": []}
    node = make_plan_node(_planner([ChatResponse(content="no grounding"), _emit(empty_plan)]))
    patch = node(_state(tmp_path, autonomy="auto"))
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "escalation"
    assert patch["errors"][0].kind == "planning_failed"


def _memory_planner(
    responses: list[ChatResponse],
) -> tuple[Planner, FakeProvider]:
    settings = Settings(_env_file=None, planner=PlannerSettings(grounding_steps=1))
    provider = FakeProvider(capabilities=_CAPS, responses=responses)
    return Planner(provider, build_planner_registry(), settings), provider


class _FakeLongTerm:
    def search(self, project_id: str, query: str, k: int = 5) -> list[MemoryItem]:
        return [MemoryItem(kind="decision", text="Use pnpm for packages.")]


class _FakeEpisodic:
    def relevant(
        self, project_id: str, query: str, k: int = 3, *, candidate_window: int = 50
    ) -> list[RunRecord]:
        return [RunRecord(run_id="r0", project_id="p1", status="failed",
                          summary="verify failed on calc.py")]


def test_plan_node_injects_memory_into_planner_context(tmp_path: Path) -> None:
    planner, provider = _memory_planner([ChatResponse(content="ok"), _emit(_GOOD_PLAN)])
    node = make_plan_node(
        planner, None, _FakeLongTerm(), _FakeEpisodic(),  # type: ignore[arg-type]
        settings=PlannerSettings(),
    )
    node(_state(tmp_path, autonomy="auto"))

    seen = "\n".join(m.content or "" for call in provider.calls for m in call.messages)
    assert "Project Conventions" in seen and "Use pnpm" in seen
    assert "Previous Attempts" in seen and "verify failed on calc.py" in seen


def test_plan_node_without_memory_injects_nothing(tmp_path: Path) -> None:
    # No long_term/episodic wired -> behaves exactly as before (no memory message).
    planner, provider = _memory_planner([ChatResponse(content="ok"), _emit(_GOOD_PLAN)])
    node = make_plan_node(planner)
    node(_state(tmp_path, autonomy="auto"))

    # The system prompt itself mentions the section names, so assert on the
    # injected-message marker + the concrete memory payload instead.
    seen = "\n".join(m.content or "" for call in provider.calls for m in call.messages)
    assert "Retrieved memory for grounding" not in seen
    assert "Use pnpm" not in seen


def _state_for(tmp_path: Path, project_id: str, request: str) -> AgentState:
    return new_run_state(
        run_id="r1", project_id=project_id, user_request=request,
        workspace_path=str(tmp_path), autonomy_level="auto",
        max_tokens=None, max_steps=50, max_wall_clock_s=3600, started_at=now_iso(),
    )


@pytest.mark.integration
@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_plan_node_includes_real_memory_end_to_end(tmp_path: Path) -> None:
    """Seed real pgvector (via ADR ingestion) + real episodic Postgres, then prove
    the planner's recorded context contains both — and degrades cleanly when empty."""
    project = f"proj-{uuid.uuid4().hex[:8]}"
    lt = LongTermMemory(_DSN, ChunkEmbedder(provider=DirEmbedProvider()), dim=2, table=_LT_TABLE)
    lt.ensure_schema()
    ep = EpisodicMemory(_DSN, table=_EP_TABLE)
    ep.ensure_schema()

    adr_dir = tmp_path / "adr"
    adr_dir.mkdir()
    (adr_dir / "0001-py.md").write_text(
        "# ADR: Python toolchain\n\n## Decision\nUse python 3.11 and ruff.\n", encoding="utf-8"
    )
    try:
        ingest_adrs(lt, project, adr_dir)
        ep.record(RunRecord(run_id="old", project_id=project, status="failed",
                            summary="python verify failed on calc"))

        planner, provider = _memory_planner([ChatResponse(content="ok"), _emit(_GOOD_PLAN)])
        node = make_plan_node(planner, None, lt, ep, settings=PlannerSettings())
        node(_state_for(tmp_path, project, "add a python calculator"))

        seen = "\n".join(m.content or "" for call in provider.calls for m in call.messages)
        assert "Retrieved memory for grounding" in seen
        assert "python" in seen.lower()  # the ingested ADR decision
        assert "python verify failed on calc" in seen  # the relevant past failure

        # Degrade: a fresh project with empty memory injects no block.
        empty_project = f"proj-{uuid.uuid4().hex[:8]}"
        planner2, provider2 = _memory_planner([ChatResponse(content="ok"), _emit(_GOOD_PLAN)])
        node2 = make_plan_node(planner2, None, lt, ep, settings=PlannerSettings())
        node2(_state_for(tmp_path, empty_project, "unrelated request"))
        seen2 = "\n".join(m.content or "" for call in provider2.calls for m in call.messages)
        assert "Retrieved memory for grounding" not in seen2
    finally:
        lt.clear_project(project)
        with ep._connect() as conn:
            conn.execute(f"DELETE FROM {ep._table} WHERE project_id = %s", (project,))


def test_plan_node_structured_failure_escalates(tmp_path: Path) -> None:
    # Every response is unparseable prose -> structured_call exhausts repairs and raises.
    node = make_plan_node(
        _planner([ChatResponse(content="not json") for _ in range(5)], grounding_steps=0)
    )
    patch = node(_state(tmp_path, autonomy="auto"))
    assert patch["hitl_request"].kind == "escalation"
    assert "Planning failed" in patch["hitl_request"].context


def test_plan_node_clarification_interrupt_then_resume(tmp_path: Path) -> None:
    # LangGraph replays a node from the top on resume: the *draft* create_plan
    # call (the one producing open_questions) runs once before the original
    # pause and once again during replay, before interrupt() returns its
    # cached answer — so the FakeProvider must serve the same blocking draft
    # twice. Only the call *after* the interrupt line is genuinely new. This
    # mirrors the replay characteristic documented in graph/nodes/coder.py for
    # command_approval.
    planner = _planner(
        [
            ChatResponse(content="no grounding"),
            _emit(_BLOCKING_PLAN),  # draft, original execution -> hits interrupt()
            ChatResponse(content="no grounding"),
            _emit(_BLOCKING_PLAN),  # draft, replayed on resume -> interrupt() returns cached answer
            ChatResponse(content="no grounding"),
            _emit(_GOOD_PLAN),  # final plan, incorporating the human's answer
        ],
        grounding_steps=1,
    )
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("plan", make_plan_node(planner))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", END)
    compiled = graph.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t1"}}
    state = _state(tmp_path, autonomy="semi")
    paused = compiled.invoke(state, config=config)  # type: ignore[call-overload]
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["kind"] == "clarification"

    resumed = compiled.invoke(  # type: ignore[call-overload]
        Command(resume={"answers": ["integers only"]}), config=config
    )
    assert resumed["plan"].summary == "Add a calculator."
    assert resumed["clarification_answers"] == ["integers only"]
    assert resumed["plan"].open_questions == []
