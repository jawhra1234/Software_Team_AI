"""Task 2.11 — Postgres checkpointer durability (integration; needs live Postgres).

Mirrors the SQLite checkpoint-recovery test in test_graph_e2e.py but against
the production-realism backend (ADR-0010), proving "runs are durable under
both SQLite and Postgres checkpointers" (Phase 2 DoD).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from app.core.clock import now_iso
from app.core.config import PlannerSettings, ReviewerSettings, Settings
from app.graph.build_graph import build_graph
from app.graph.checkpointer import build_checkpointer
from app.graph.state import Plan, Review, new_run_state
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.tools.git import Git
from app.tools.sandbox import SubprocessSandbox
from langgraph.types import Command

from tests.fakes import FakeProvider

pytestmark = pytest.mark.integration

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)
_GOOD_PLAN = {
    "summary": "Add a calculator.",
    "tasks": [{"id": "task-1", "title": "add calc.py", "description": "d", "kind": "create"}],
}


def _postgres_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


def _emit_plan(payload: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="p", name=emit_tool_name(Plan), arguments=payload)]
    )


def _approving_reviewer() -> FakeProvider:
    payload = {"verdict": "approved", "issues": [], "summary": "looks correct"}
    return FakeProvider(
        capabilities=_CAPS,
        responses=[
            ChatResponse(
                content="",
                tool_calls=[ToolCall(id="rv", name=emit_tool_name(Review), arguments=payload)],
            )
        ],
    )


def _sandbox() -> SubprocessSandbox:
    return SubprocessSandbox(Settings(_env_file=None).sandbox.model_copy(update={"backend": "subprocess"}))


@pytest.mark.skipif(
    not _postgres_reachable(Settings(_env_file=None).postgres.dsn),
    reason="local Postgres not reachable",
)
def test_postgres_checkpoint_recovery_across_simulated_restart(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    git = Git(workspace)
    git.init()
    base = git.commit("chore: init")

    settings = Settings(
        _env_file=None,
        planner=PlannerSettings(grounding_steps=0),
        reviewer=ReviewerSettings(grounding_steps=0),
        checkpointer={"backend": "postgres"},  # type: ignore[arg-type]
    )
    config = {"configurable": {"thread_id": f"pg-recover-{now_iso()}"}}

    initial_state = new_run_state(
        run_id="r1", project_id="p1", user_request="build a calculator",
        workspace_path=str(workspace), autonomy_level="semi",
        max_tokens=None, max_steps=100, max_wall_clock_s=3600, started_at=now_iso(),
    )
    initial_state["base_commit"] = base
    initial_state["work_branch"] = git.current_branch()

    with build_checkpointer(settings) as cp1:
        graph1 = build_graph(
            settings,
            sandbox=_sandbox(),
            checkpointer=cp1,
            planner_provider=FakeProvider(capabilities=_CAPS, responses=[_emit_plan(_GOOD_PLAN)]),
            coder_provider=FakeProvider(capabilities=_CAPS, responses=[]),
        )
        paused = graph1.invoke(initial_state, config=config)  # type: ignore[call-overload]
        assert "__interrupt__" in paused
        assert paused["__interrupt__"][0].value["kind"] == "plan_approval"
    # Connection closed here — simulates a process restart.

    with build_checkpointer(settings) as cp2:
        graph2 = build_graph(
            settings,
            sandbox=_sandbox(),
            checkpointer=cp2,
            planner_provider=FakeProvider(capabilities=_CAPS, responses=[]),
            coder_provider=FakeProvider(
                capabilities=_CAPS,
                responses=[
                    ChatResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="1", name="write_file",
                                arguments={"path": "calc.py", "content": "def add(a,b):\n    return a+b\n"},
                            )
                        ],
                    ),
                    ChatResponse(
                        content="",
                        tool_calls=[ToolCall(id="2", name="finish_task", arguments={"summary": "done"})],
                    ),
                ],
            ),
            reviewer_provider=_approving_reviewer(),
        )
        state = graph2.get_state(config)  # type: ignore[arg-type]
        assert state.values["plan"].summary == "Add a calculator."

        result = graph2.invoke(  # type: ignore[call-overload]
            Command(resume={"decision": "approve"}), config=config
        )
        assert result["status"] == "succeeded"
