"""Tasks 2.3/2.8 — coder node: task selection, fix mode, escalation, command approval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from app.core.clock import now_iso
from app.core.config import CoderSettings, Settings
from app.graph.nodes.coder import build_fix_task, make_coder_node, select_next_task
from app.graph.state import AgentState, Plan, Task, VerifyResult, new_run_state
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.tools.git import Git
from app.tools.sandbox import SubprocessSandbox
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from tests.fakes import FakeProvider

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)


def _tool_call(name: str, **arguments: object) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="1", name=name, arguments=dict(arguments))]
    )


def _sandbox() -> SubprocessSandbox:
    return SubprocessSandbox(Settings(_env_file=None).sandbox.model_copy(update={"backend": "subprocess"}))


def _state(tmp_path: Path, plan: Plan, **overrides: object) -> AgentState:
    git = Git(tmp_path)
    git.init()
    base = git.commit("chore: init")
    state = new_run_state(
        run_id="r1", project_id="p1", user_request="x", workspace_path=str(tmp_path),
        autonomy_level="auto", max_tokens=None, max_steps=50, max_wall_clock_s=3600,
        started_at=now_iso(),
    )
    state["plan"] = plan
    state["base_commit"] = base
    state["work_branch"] = git.current_branch()
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _one_task_plan() -> Plan:
    return Plan(
        summary="s",
        tasks=[
            Task(
                id="task-1", title="add calc.py", description="d", kind="create",
                acceptance_criteria=["calc.py defines add"],
            )
        ],
    )


# ---------------------------------------------------------------------------
# select_next_task / build_fix_task (pure helpers)
# ---------------------------------------------------------------------------
def test_select_next_task_respects_dependencies() -> None:
    t1 = Task(id="t1", title="a", description="d", kind="create", status="pending")
    t2 = Task(id="t2", title="b", description="d", kind="create", status="pending", depends_on=["t1"])
    assert select_next_task([t1, t2]).id == "t1"  # type: ignore[union-attr]
    t1.status = "done"
    assert select_next_task([t1, t2]).id == "t2"  # type: ignore[union-attr]
    t2.status = "done"
    assert select_next_task([t1, t2]) is None


def test_build_fix_task_from_failed_verify() -> None:
    result = VerifyResult(passed=False, summary="pytest failed", checks=[])
    task = build_fix_task(cast("AgentState", {"verify_result": result}))
    assert task is not None
    assert "pytest failed" in task.description


def test_build_fix_task_none_when_nothing_to_fix() -> None:
    assert build_fix_task(cast("AgentState", {})) is None


# ---------------------------------------------------------------------------
# coder node — task mode (no graph needed; no interrupt on this path)
# ---------------------------------------------------------------------------
def test_coder_node_completes_task_and_advances(tmp_path: Path) -> None:
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            _tool_call("write_file", path="calc.py", content="def add(a,b):\n    return a+b\n"),
            _tool_call("finish_task", summary="done"),
        ],
    )
    from app.tools.registry import build_default_registry

    node = make_coder_node(provider, build_default_registry(), Settings(_env_file=None), _sandbox())
    patch = node(_state(tmp_path, _one_task_plan()))

    assert patch["current_task_id"] is None  # only task, now done
    assert patch["plan"].tasks[0].status == "done"
    assert patch["hitl_request"] is None
    assert any(f.path == "calc.py" for f in patch["changed_files"])


def test_coder_node_escalates_on_budget_exceeded(tmp_path: Path) -> None:
    from app.tools.registry import build_default_registry

    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[_tool_call("list_dir", path=".") for _ in range(5)],
    )
    settings = Settings(_env_file=None, coder=CoderSettings(max_steps_per_task=1, no_progress_limit=9))
    node = make_coder_node(provider, build_default_registry(), settings, _sandbox())
    patch = node(_state(tmp_path, _one_task_plan()))

    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "escalation"
    assert patch["plan"].tasks[0].status == "failed"


def test_coder_node_fix_mode_runs_ad_hoc_task(tmp_path: Path) -> None:
    from app.tools.registry import build_default_registry

    done_plan = _one_task_plan()
    done_plan.tasks[0].status = "done"
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            _tool_call("write_file", path="fix.py", content="x = 1\n"),
            _tool_call("finish_task", summary="fixed"),
        ],
    )
    node = make_coder_node(provider, build_default_registry(), Settings(_env_file=None), _sandbox())
    state = _state(
        tmp_path, done_plan, current_task_id=None,
        verify_result=VerifyResult(passed=False, summary="tests failed"),
        retries={"verify": 1},
    )
    patch = node(state)
    assert patch["hitl_request"] is None
    assert any(f.path == "fix.py" for f in patch["changed_files"])


def test_coder_node_fix_mode_no_changes_escalates(tmp_path: Path) -> None:
    from app.tools.registry import build_default_registry

    done_plan = _one_task_plan()
    done_plan.tasks[0].status = "done"
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[_tool_call("finish_task", summary="nothing to do, already fine")],
    )
    node = make_coder_node(provider, build_default_registry(), Settings(_env_file=None), _sandbox())
    state = _state(
        tmp_path, done_plan, current_task_id=None,
        verify_result=VerifyResult(passed=False, summary="tests failed"),
        retries={"verify": 1},
    )
    patch = node(state)
    assert patch["hitl_request"] is not None
    assert patch["hitl_request"].kind == "escalation"
    assert "no changes" in patch["hitl_request"].context


# ---------------------------------------------------------------------------
# command_approval interrupt (Task 2.8) — needs a real compiled graph
# ---------------------------------------------------------------------------
def test_coder_node_command_approval_interrupt(tmp_path: Path) -> None:
    from app.tools.registry import build_default_registry

    # See graph/nodes/coder.py's module docstring: resuming replays the node
    # from the top, so the *first* LLM call (leading to the gated run_command)
    # must be served twice — once for the original pause, once for replay,
    # before interrupt() returns its cached decision.
    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            _tool_call("run_command", command="python -c \"print(1)\""),
            _tool_call("run_command", command="python -c \"print(1)\""),
            _tool_call("finish_task", summary="ran it"),
        ],
    )
    settings = Settings(_env_file=None)
    node_fn = make_coder_node(provider, build_default_registry(), settings, _sandbox())

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("coder", node_fn)
    graph.add_edge(START, "coder")
    graph.add_edge("coder", END)
    compiled = graph.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t1"}}
    state: dict[str, Any] = dict(_state(tmp_path, _one_task_plan(), autonomy_level="semi"))
    paused = compiled.invoke(state, config=config)  # type: ignore[call-overload]
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["kind"] == "command_approval"

    resumed = compiled.invoke(Command(resume="approve"), config=config)  # type: ignore[call-overload]
    assert resumed["current_task_id"] is None
    assert resumed["plan"].tasks[0].status == "done"


def test_coder_node_command_denied_fails_task(tmp_path: Path) -> None:
    from app.tools.registry import build_default_registry

    provider = FakeProvider(
        capabilities=_CAPS,
        responses=[
            _tool_call("run_command", command="python -c \"print(1)\""),
            _tool_call("run_command", command="python -c \"print(1)\""),
            _tool_call("finish_task", summary="gave up"),
        ],
    )
    settings = Settings(_env_file=None)
    node_fn = make_coder_node(provider, build_default_registry(), settings, _sandbox())

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("coder", node_fn)
    graph.add_edge(START, "coder")
    graph.add_edge("coder", END)
    compiled = graph.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t2"}}
    state = dict(_state(tmp_path, _one_task_plan(), autonomy_level="semi"))
    compiled.invoke(state, config=config)  # type: ignore[call-overload]
    resumed = compiled.invoke(Command(resume="deny"), config=config)  # type: ignore[call-overload]
    # Denied command -> tool returns an error observation -> coder still finishes via finish_task.
    assert resumed["current_task_id"] is None
