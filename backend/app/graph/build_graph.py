"""Graph assembly (Task 2.9, ARCHITECTURE.md §4).

Wires the 6 nodes and conditional edges into the compiled LangGraph state
machine::

    plan -> human_gate[plan_approval] --approve--> coder <-> coder (more tasks)
                |revise-> plan                        |all done
                |abort -> finalize                    v
                                                     verify --pass--> review --approved--> finalize
                                                       |fail(retry)      |changes(retry)
                                                       +--> coder <------+
    budget/loop/exhausted -> human_gate[escalation] --(retry|accept|abort)--> coder|plan|finalize

Recursion limit is set well above the sum of bounded per-node retries
(``GraphSettings.max_verify_retries`` + ``max_review_cycles``, each cycling
through 2-3 nodes) so escalation — not a raw ``GraphRecursionError`` — is
always the terminal path for a run that keeps failing.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.planner import Planner
from app.agents.reviewer import Reviewer
from app.core.config import Settings
from app.graph import routing
from app.graph.events import EventSink, NullEventSink
from app.graph.instrument import instrument_node
from app.graph.nodes.coder import make_coder_node
from app.graph.nodes.finalize import make_finalize_node
from app.graph.nodes.human_gate import make_human_gate_node
from app.graph.nodes.plan import make_plan_node
from app.graph.nodes.review import make_review_node
from app.graph.nodes.verify import make_verify_node
from app.graph.state import AgentState
from app.memory.episodic import EpisodicMemory
from app.memory.long_term import LongTermMemory
from app.providers.base import LLMProvider
from app.providers.factory import get_provider
from app.rag.retriever import Retriever
from app.tools.registry import build_default_registry, build_planner_registry
from app.tools.sandbox import Sandbox, get_sandbox

_CompiledGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def build_graph(
    settings: Settings,
    *,
    sandbox: Sandbox | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    sink: EventSink | None = None,
    planner_provider: LLMProvider | None = None,
    coder_provider: LLMProvider | None = None,
    reviewer_provider: LLMProvider | None = None,
    retriever: Retriever | None = None,
    episodic: EpisodicMemory | None = None,
    long_term: LongTermMemory | None = None,
) -> _CompiledGraph:
    """Assemble and compile the orchestration graph.

    ``planner_provider``/``coder_provider``/``reviewer_provider`` override what
    would otherwise be resolved from ``settings`` — the same dependency-injection
    seam as ``sandbox``/``checkpointer``, used by tests to inject scripted providers.

    ``retriever``/``episodic``/``long_term`` are opt-in and default to ``None``
    (RAG + both memory reads silently degrade to no-ops in that case — see
    ``tools/retrieve.py``, ``graph/nodes/finalize.py`` and
    ``graph/planning_context.py``). They are **not** auto-built from settings
    here, because that would make ``finalize`` depend on Postgres reachability
    for every caller, including hermetic tests — exactly what ADR-0010's
    "SQLite is the clone-and-run default" is meant to avoid. Callers who want
    live RAG/memory build the stack explicitly: ``build_rag_stack(settings)``.
    """
    sink = sink or NullEventSink()
    sandbox = sandbox or get_sandbox(settings.sandbox)

    planner = Planner(
        planner_provider or get_provider("planner", settings), build_planner_registry(), settings
    )
    coder_provider = coder_provider or get_provider("coder", settings)
    coder_registry = build_default_registry()
    # The reviewer's grounding is read-only (ADR-0006), so it reuses the planner's
    # tool set: retrieve/read_file/list_dir/search_code — never write_file/edit_file.
    reviewer = Reviewer(
        reviewer_provider or get_provider("reviewer", settings), build_planner_registry(), settings
    )

    # mypy's overload resolution for `add_node` doesn't structurally match our
    # plain `Callable[[AgentState], dict[str, Any]]` node functions against
    # LangGraph's richer `_Node` protocol (which also accepts config/writer/store
    # variants) — a known typing-strictness friction with TypedDict state, not a
    # runtime issue (verified empirically; see the Phase 2 completion report).
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node(  # type: ignore[call-overload]
        "plan",
        instrument_node(
            "plan",
            make_plan_node(planner, retriever, long_term, episodic, settings=settings.planner),
            sink,
        ),
    )
    graph.add_node(  # type: ignore[call-overload]
        "human_gate",
        instrument_node("human_gate", make_human_gate_node(), sink, enforce_budget=False),
    )
    graph.add_node(  # type: ignore[call-overload]
        "coder",
        instrument_node(
            "coder",
            make_coder_node(coder_provider, coder_registry, settings, sandbox, retriever),
            sink,
        ),
    )
    graph.add_node(  # type: ignore[call-overload]
        "verify",
        instrument_node("verify", make_verify_node(sandbox, settings.graph, settings.coder), sink),
    )
    graph.add_node(  # type: ignore[call-overload]
        "review",
        instrument_node("review", make_review_node(reviewer, settings.graph, retriever), sink),
    )
    graph.add_node(  # type: ignore[call-overload]
        "finalize",
        instrument_node("finalize", make_finalize_node(episodic), sink, enforce_budget=False),
    )

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", routing.route_after_plan, {"human_gate": "human_gate", "coder": "coder"}
    )
    graph.add_conditional_edges(
        "human_gate",
        routing.route_after_gate,
        {"plan": "plan", "coder": "coder", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "coder",
        routing.route_after_coder,
        {"coder": "coder", "verify": "verify", "human_gate": "human_gate"},
    )
    graph.add_conditional_edges(
        "verify",
        routing.route_after_verify,
        {"review": "review", "coder": "coder", "human_gate": "human_gate"},
    )
    graph.add_conditional_edges(
        "review",
        routing.route_after_review,
        {"finalize": "finalize", "coder": "coder", "human_gate": "human_gate"},
    )
    graph.add_edge("finalize", END)

    compiled = graph.compile(checkpointer=checkpointer)
    # Bake the recursion limit into the compiled graph so it travels with it
    # regardless of how the caller invokes. It is set above the sum of bounded
    # per-node retries and the run-step budget so the budget circuit breaker /
    # escalation (not a raw GraphRecursionError) is always the terminal path.
    return compiled.with_config({"recursion_limit": settings.graph.recursion_limit})
