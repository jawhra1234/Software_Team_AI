"""Task 3.12 — retrieved_context is populated end-to-end through the real graph.

Integration: real Postgres-backed index + retriever, real `plan`/`coder` node
wiring, scripted LLM that calls `retrieve`. Proves the wiring — not retrieval
quality (that's Task 3.13's precision@k harness).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from app.core.clock import now_iso
from app.core.config import PlannerSettings, ReviewerSettings, Settings
from app.graph.build_graph import build_graph
from app.graph.state import Plan, Review, new_run_state
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.rag.embeddings import ChunkEmbedder
from app.rag.indexer import Indexer
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.tools.git import Git
from app.tools.sandbox import SubprocessSandbox
from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import FakeProvider

pytestmark = pytest.mark.integration

_CAPS = Capabilities(supports_tools=True, supports_json=True, max_context=8192)
_DSN = Settings(_env_file=None).postgres.dsn


def _postgres_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


_SKIP = not _postgres_reachable(_DSN)


def _tool_call(name: str, **arguments: object) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="1", name=name, arguments=dict(arguments))]
    )


def _emit_plan(payload: dict[str, object]) -> ChatResponse:
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


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_coder_retrieve_call_populates_retrieved_context(tmp_path: Path) -> None:
    # A pre-existing repo with a symbol the coder will retrieve.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    git = Git(repo)
    git.init()
    base = git.commit("init")

    project_id = f"proj-{uuid.uuid4().hex[:8]}"
    embed_provider = FakeProvider(capabilities=_CAPS, embed_dim=3)
    store = VectorStore(_DSN, dim=3, table="rag_chunks_test")
    embedder = ChunkEmbedder(provider=embed_provider)
    Indexer(store, embedder).index_project(project_id, repo)
    retriever = Retriever(store, embedder)

    settings = Settings(
        _env_file=None,
        planner=PlannerSettings(grounding_steps=0),
        reviewer=ReviewerSettings(grounding_steps=0),
    )
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        retriever=retriever,
        planner_provider=FakeProvider(
            capabilities=_CAPS,
            responses=[_emit_plan({
                "summary": "Use add()",
                "tasks": [{"id": "task-1", "title": "use add", "description": "d", "kind": "modify"}],
            })],
        ),
        coder_provider=FakeProvider(
            capabilities=_CAPS,
            responses=[
                _tool_call("retrieve", query="add"),
                _tool_call(
                    "write_file", path="main.py",
                    content="from calc import add\n\nprint(add(2, 3))\n",
                ),
                _tool_call("finish_task", summary="used add() via retrieval"),
            ],
        ),
        reviewer_provider=_approving_reviewer(),
    )

    state = dict(new_run_state(
        run_id="r1", project_id=project_id, user_request="use the add function",
        workspace_path=str(repo), autonomy_level="auto",
        max_tokens=None, max_steps=100, max_wall_clock_s=3600, started_at=now_iso(),
    ))
    state["base_commit"] = base
    state["work_branch"] = git.current_branch()
    config = {"configurable": {"thread_id": "t1"}}

    # `retrieved_context` is ephemeral/overwritten-per-step (Task 3.12): every node
    # that can retrieve (plan/coder/review) replaces it with its own step's chunks,
    # so a later node's empty capture can clobber an earlier one's in the *final*
    # state. Capture the coder step's own patch via the stream instead of relying
    # on graph.invoke()'s terminal-state-only view, to test what this file is
    # actually about — the coder's retrieve() call populating the field.
    coder_retrieved_context = None
    for chunk in graph.stream(state, config=config, stream_mode="updates"):  # type: ignore[call-overload]
        if "coder" in chunk:
            coder_retrieved_context = chunk["coder"].get("retrieved_context")

    assert graph.get_state(config).values["status"] == "succeeded"  # type: ignore[arg-type]
    assert coder_retrieved_context  # the coder's retrieve() call populated this
    assert any(c.symbol == "add" for c in coder_retrieved_context)


@pytest.mark.skipif(_SKIP, reason="local Postgres not reachable")
def test_plan_without_retrieve_call_has_empty_retrieved_context(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = Git(repo)
    git.init()
    base = git.commit("init")

    settings = Settings(
        _env_file=None,
        planner=PlannerSettings(grounding_steps=0),
        reviewer=ReviewerSettings(grounding_steps=0),
    )
    graph = build_graph(
        settings,
        sandbox=_sandbox(),
        checkpointer=InMemorySaver(),
        retriever=None,  # explicitly no index -> retrieve degrades, no crash
        planner_provider=FakeProvider(
            capabilities=_CAPS,
            responses=[_emit_plan({
                "summary": "s",
                "tasks": [{"id": "task-1", "title": "t", "description": "d", "kind": "create"}],
            })],
        ),
        coder_provider=FakeProvider(capabilities=_CAPS, responses=[]),
    )
    state = dict(new_run_state(
        run_id="r2", project_id="proj-none", user_request="x",
        workspace_path=str(repo), autonomy_level="semi",
        max_tokens=None, max_steps=100, max_wall_clock_s=3600, started_at=now_iso(),
    ))
    state["base_commit"] = base
    state["work_branch"] = git.current_branch()

    paused = graph.invoke(state, config={"configurable": {"thread_id": "t2"}})  # type: ignore[call-overload]
    assert "__interrupt__" in paused  # plan_approval gate (semi)
    assert paused.get("retrieved_context", []) == []
