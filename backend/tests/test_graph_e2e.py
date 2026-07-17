"""Phase 2 DoD — full compiled graph: happy path, HITL, fix loop, autonomy, budget.

FakeProviders are injected via ``build_graph``'s ``planner_provider``/
``coder_provider`` overrides, so every test exercises the *real* assembly
function, not a re-implementation of its wiring.

Resume-replay note: only the currently-paused node re-executes on resume
(verified empirically) — nodes that already completed earlier in the run are
not re-run. So a ``human_gate`` interrupt (no LLM call before it) costs zero
extra provider responses on resume; only interrupts *inside* ``plan``/``coder``
need their preceding LLM call scripted twice (see test_graph_nodes_plan.py /
test_graph_nodes_coder.py for that case).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.clock import now_iso
from app.core.config import GraphSettings, PlannerSettings, Settings
from app.graph.build_graph import build_graph
from app.graph.checkpointer import build_checkpointer
from app.graph.state import Budget, new_run_state
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.tools.git import Git
from app.tools.sandbox import SubprocessSandbox
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import FakeProvider

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)

_GOOD_PLAN = {
    "summary": "Add a calculator.",
    "tasks": [
        {
            "id": "task-1", "title": "add calc.py", "description": "d", "kind": "create",
            "acceptance_criteria": ["calc.py defines add(a, b)", "tests pass"],
        }
    ],
}


def _emit_plan(payload: dict[str, Any]) -> ChatResponse:
    from app.graph.state import Plan

    return ChatResponse(
        content="", tool_calls=[ToolCall(id="p", name=emit_tool_name(Plan), arguments=payload)]
    )


def _multi_call(*calls: tuple[str, dict[str, Any]]) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[ToolCall(id=str(i), name=n, arguments=a) for i, (n, a) in enumerate(calls)],
    )


_PASSING_CODE = "def add(a, b):\n    return a + b\n"
_BUGGY_CODE = "def add(a, b):\n    return a - b\n"
_TEST_CODE = "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"

_WRITE_PASSING = _multi_call(
    ("write_file", {"path": "calc.py", "content": _PASSING_CODE}),
    ("write_file", {"path": "test_calc.py", "content": _TEST_CODE}),
)
_WRITE_BUGGY = _multi_call(
    ("write_file", {"path": "calc.py", "content": _BUGGY_CODE}),
    ("write_file", {"path": "test_calc.py", "content": _TEST_CODE}),
)
_FINISH = ChatResponse(content="", tool_calls=[ToolCall(id="f", name="finish_task", arguments={"summary": "done"})])
_FIX_CODE = _multi_call(("write_file", {"path": "calc.py", "content": _PASSING_CODE}))


def _planner_provider(*plans: dict[str, Any]) -> FakeProvider:
    return FakeProvider(capabilities=_CAPS, responses=[_emit_plan(p) for p in plans])


def _settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        planner=PlannerSettings(grounding_steps=0),
        **overrides,
    )


def _init_state(tmp_path: Path, autonomy: str, **overrides: Any) -> dict[str, Any]:
    git = Git(tmp_path)
    git.init()
    base = git.commit("chore: init")
    state = dict(
        new_run_state(
            run_id="r1", project_id="p1", user_request="build a calculator",
            workspace_path=str(tmp_path), autonomy_level=autonomy,  # type: ignore[arg-type]
            max_tokens=None, max_steps=100, max_wall_clock_s=3600,
            started_at=now_iso(),
        )
    )
    state["base_commit"] = base
    state["work_branch"] = git.current_branch()
    state.update(overrides)
    return state


def _sandbox() -> SubprocessSandbox:
    return SubprocessSandbox(Settings(_env_file=None).sandbox.model_copy(update={"backend": "subprocess"}))


# ---------------------------------------------------------------------------
# Recursion limit is actually baked into the compiled graph
# ---------------------------------------------------------------------------
def test_recursion_limit_is_applied() -> None:
    settings = _settings(graph=GraphSettings(recursion_limit=137))
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        planner_provider=_planner_provider(_GOOD_PLAN),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[]),
    )
    # The limit must be above the run-step budget so the budget circuit breaker
    # (which escalates) fires before a raw GraphRecursionError.
    assert (graph.config or {}).get("recursion_limit") == 137
    assert settings.graph.recursion_limit > settings.graph.max_run_steps


# ---------------------------------------------------------------------------
# Happy path — auto autonomy, zero interrupts
# ---------------------------------------------------------------------------
def test_happy_path_auto_autonomy_no_interrupts(tmp_path: Path) -> None:
    settings = _settings()
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        planner_provider=_planner_provider(_GOOD_PLAN),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[_WRITE_PASSING, _FINISH]),
    )
    config = {"configurable": {"thread_id": "happy"}}
    result = graph.invoke(_init_state(tmp_path, "auto"), config=config)  # type: ignore[call-overload]

    assert "__interrupt__" not in result
    assert result["status"] == "succeeded"
    assert result["verify_result"].passed
    assert result["review"].verdict == "approved"
    assert result["plan"].tasks[0].status == "done"


# ---------------------------------------------------------------------------
# plan_approval interrupt + resume (semi)
# ---------------------------------------------------------------------------
def test_plan_approval_interrupt_then_approve(tmp_path: Path) -> None:
    settings = _settings()
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        planner_provider=_planner_provider(_GOOD_PLAN),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[_WRITE_PASSING, _FINISH]),
    )
    config = {"configurable": {"thread_id": "approve"}}
    paused = graph.invoke(_init_state(tmp_path, "semi"), config=config)  # type: ignore[call-overload]
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["kind"] == "plan_approval"

    result = graph.invoke(  # type: ignore[call-overload]
        Command(resume={"decision": "approve"}), config=config
    )
    assert result["status"] == "succeeded"


def test_plan_approval_abort_cancels_run(tmp_path: Path) -> None:
    settings = _settings()
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        planner_provider=_planner_provider(_GOOD_PLAN),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[]),
    )
    config = {"configurable": {"thread_id": "abort"}}
    graph.invoke(_init_state(tmp_path, "semi"), config=config)  # type: ignore[call-overload]
    result = graph.invoke(Command(resume={"decision": "abort"}), config=config)  # type: ignore[call-overload]
    assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Plan-revise loop
# ---------------------------------------------------------------------------
def test_plan_revise_loop_replans_then_approves(tmp_path: Path) -> None:
    settings = _settings()
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        # 2 plan-node visits (initial + after "revise") -> 2 queued plan emissions.
        planner_provider=_planner_provider(_GOOD_PLAN, _GOOD_PLAN),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[_WRITE_PASSING, _FINISH]),
    )
    config = {"configurable": {"thread_id": "revise"}}
    graph.invoke(_init_state(tmp_path, "semi"), config=config)  # type: ignore[call-overload]

    revised = graph.invoke(  # type: ignore[call-overload]
        Command(resume={"decision": "revise", "note": "add more detail"}), config=config
    )
    assert "__interrupt__" in revised  # back at plan_approval for the re-plan
    assert revised["__interrupt__"][0].value["kind"] == "plan_approval"

    result = graph.invoke(Command(resume={"decision": "approve"}), config=config)  # type: ignore[call-overload]
    assert result["status"] == "succeeded"


# ---------------------------------------------------------------------------
# Verify-fail fix loop
# ---------------------------------------------------------------------------
def test_verify_fail_triggers_fix_then_passes(tmp_path: Path) -> None:
    settings = _settings(graph=GraphSettings(max_verify_retries=3))
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        planner_provider=_planner_provider(_GOOD_PLAN),
        coder_provider=FakeProvider(
            capabilities=_CAPS,
            responses=[_WRITE_BUGGY, _FINISH, _FIX_CODE, _FINISH],
        ),
    )
    config = {"configurable": {"thread_id": "fix"}}
    result = graph.invoke(_init_state(tmp_path, "auto"), config=config)  # type: ignore[call-overload]

    assert result["status"] == "succeeded"
    assert result["verify_result"].passed
    assert result["retries"]["verify"] == 1  # failed once, then fixed


def test_verify_fail_exhausted_escalates(tmp_path: Path) -> None:
    settings = _settings(graph=GraphSettings(max_verify_retries=1))
    # Coder never actually fixes the bug (keeps "finishing" without changing anything).
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        planner_provider=_planner_provider(_GOOD_PLAN),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[_WRITE_BUGGY, _FINISH]),
    )
    config = {"configurable": {"thread_id": "exhaust"}}
    paused = graph.invoke(_init_state(tmp_path, "auto"), config=config)  # type: ignore[call-overload]
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["kind"] == "escalation"

    result = graph.invoke(Command(resume={"decision": "abort"}), config=config)  # type: ignore[call-overload]
    assert result["status"] == "failed"


# ---------------------------------------------------------------------------
# Autonomy matrix
# ---------------------------------------------------------------------------
def test_autonomy_manual_adds_final_accept_gate(tmp_path: Path) -> None:
    settings = _settings()
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        planner_provider=_planner_provider(_GOOD_PLAN),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[_WRITE_PASSING, _FINISH]),
    )
    config = {"configurable": {"thread_id": "manual"}}
    plan_gate = graph.invoke(_init_state(tmp_path, "manual"), config=config)  # type: ignore[call-overload]
    assert plan_gate["__interrupt__"][0].value["kind"] == "plan_approval"

    final_gate = graph.invoke(  # type: ignore[call-overload]
        Command(resume={"decision": "approve"}), config=config
    )
    assert "__interrupt__" in final_gate
    assert final_gate["__interrupt__"][0].value["kind"] == "final_accept"

    result = graph.invoke(Command(resume={"decision": "accept"}), config=config)  # type: ignore[call-overload]
    assert result["status"] == "succeeded"


def test_autonomy_auto_has_zero_gates_on_happy_path(tmp_path: Path) -> None:
    # Re-assert the auto case explicitly as part of the matrix (see also the
    # dedicated happy-path test above).
    settings = _settings()
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        planner_provider=_planner_provider(_GOOD_PLAN),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[_WRITE_PASSING, _FINISH]),
    )
    result = graph.invoke(  # type: ignore[call-overload]
        _init_state(tmp_path, "auto"), config={"configurable": {"thread_id": "auto-matrix"}}
    )
    assert "__interrupt__" not in result
    assert result["status"] == "succeeded"


# ---------------------------------------------------------------------------
# Run-wide budget circuit breaker
# ---------------------------------------------------------------------------
def test_run_budget_exhausted_escalates_without_crashing(tmp_path: Path) -> None:
    settings = _settings()
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        planner_provider=_planner_provider(_GOOD_PLAN),
        # Coder should never actually be called — budget exhausts right after `plan`.
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[]),
    )
    state = _init_state(tmp_path, "auto")
    state["budget"] = Budget(
        max_tokens=None, max_steps=1, max_wall_clock_s=3600.0,
        steps_used=0, started_at=now_iso(),
    )
    config = {"configurable": {"thread_id": "budget"}}
    paused = graph.invoke(state, config=config)  # type: ignore[call-overload]

    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["kind"] == "escalation"
    assert "budget" in paused["__interrupt__"][0].value["context"]

    result = graph.invoke(Command(resume={"decision": "abort"}), config=config)  # type: ignore[call-overload]
    assert result["status"] == "failed"


# ---------------------------------------------------------------------------
# Checkpoint recovery — durable SQLite, simulated process restart
# ---------------------------------------------------------------------------
def test_checkpoint_recovery_across_simulated_restart(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    settings = _settings()
    settings.checkpointer.sqlite_path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "recover"}}

    with build_checkpointer(settings) as cp1:
        graph1 = build_graph(
            settings,
            sandbox=_sandbox(),
            checkpointer=cp1,
            planner_provider=_planner_provider(_GOOD_PLAN),
            coder_provider=FakeProvider(capabilities=_CAPS, responses=[]),
        )
        paused = graph1.invoke(  # type: ignore[call-overload]
            _init_state(workspace, "semi"), config=config
        )
        assert "__interrupt__" in paused
        assert paused["__interrupt__"][0].value["kind"] == "plan_approval"
    # `with` block exited: connection closed, simulating a process crash.

    with build_checkpointer(settings) as cp2:
        graph2 = build_graph(
            settings,
            sandbox=_sandbox(),
            checkpointer=cp2,
            planner_provider=_planner_provider(_GOOD_PLAN),  # unused: plan already completed
            coder_provider=FakeProvider(capabilities=_CAPS, responses=[_WRITE_PASSING, _FINISH]),
        )
        state = graph2.get_state(config)  # type: ignore[arg-type]
        assert state.values["plan"].summary == "Add a calculator."  # survived the "restart"

        result = graph2.invoke(  # type: ignore[call-overload]
            Command(resume={"decision": "approve"}), config=config
        )
        assert result["status"] == "succeeded"
