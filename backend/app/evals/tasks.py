"""The eval task suite (Phase 5).

Five fixed, self-contained tasks, each reusing a fixture already validated live
in Phases 3-4. Each task function sets up its fixture, drives the real graph
(or, for retrieval, the retriever directly), and returns a scored
:class:`RunReport`. Nothing here takes arbitrary user input — the tasks are
frozen so their numbers are comparable across runs.

Two tasks are deliberately special:
* ``defect_injection`` **scripts the planner + coder** so a real duplication
  defect reliably exists; only the *reviewer* is live, since the reviewer is
  what's under test.
* ``retrieval`` is **not a graph run** at all — it's a deterministic
  precision@k measurement over a real index (the one gate-worthy metric).

Provider policy: live tasks leave providers unset so ``build_graph`` resolves
the real models; a caller may pass ``reviewer_provider`` (etc.) to inject a
scripted provider for hermetic testing.
"""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.core.clock import now_iso
from app.core.config import Settings
from app.evals.report import (
    CROSS_RUN_MEMORY,
    DEFECT_INJECTION,
    HAPPY_PATH,
    RAG_REQUIRED,
    RETRIEVAL,
    RunReport,
)
from app.evals.runner import run_graph
from app.graph.build_graph import build_graph
from app.graph.planning_context import build_planner_context
from app.graph.state import Plan, new_run_state
from app.providers.base import Capabilities, ChatResponse, Chunk, LLMProvider, ToolCall, Vector
from app.providers.structured import emit_tool_name
from app.rag.evaluation import RetrievalCase, index_and_evaluate
from app.rag.factory import RagStack
from app.tools.git import Git
from app.tools.sandbox import Sandbox


@dataclass
class TaskContext:
    """Everything a task needs to run against the real (or a scripted) stack."""

    settings: Settings
    sandbox: Sandbox
    rag: RagStack
    #: Repo to index for the retrieval task (defaults to this backend package's root).
    retrieval_repo: Path
    #: Provider overrides — ``None`` means live (``build_graph`` resolves the real model).
    #: Hermetic tests inject fakes; the ``defect_injection`` task pins its own scripted
    #: planner/coder regardless, since a reliable defect is the point.
    reviewer_provider: LLMProvider | None = None


# ---------------------------------------------------------------------------
# Fixtures (reused from the Phase 3-4 live-validation scripts, inlined so the
# eval suite is self-contained inside the app package).
# ---------------------------------------------------------------------------
_CALC_REQUEST = (
    "Create a file calc.py with a function add(a, b) that returns a + b, and a file "
    "test_calc.py with a test asserting add(2, 3) == 5. Run `python -m pytest -q` to "
    "confirm the test passes, then finish."
)

_PRICING_RULES = '''\
import math


def apply_levy(amount, rate_pct):
    """The company's standard order-total rule: surcharge, round UP to 5c, + 0.30 fee."""
    surcharged = amount + amount * rate_pct / 100.0
    rounded_up = math.ceil(surcharged * 20) / 20
    return round(rounded_up + 0.30, 2)
'''

_CHECKOUT_TEST = '''\
from checkout import checkout_total
from pricing_rules import apply_levy


def test_matches_company_rule():
    assert checkout_total([4.00, 6.00], 8) == apply_levy(10.00, 8)
'''

_RAG_REQUEST = (
    "Add a function checkout_total(cart, rate_pct) in a new file checkout.py. cart is a "
    "list of item prices. Return the tax-inclusive total for the whole cart using the "
    "company's standard order-total rule. A helper implementing that exact rule already "
    "exists in this project — find it with retrieve and reuse it rather than reimplement it."
)

_WEAK_TEST = '''\
from checkout import checkout_total


def test_returns_a_number_above_subtotal():
    total = checkout_total([10.0, 20.0], 8)
    assert isinstance(total, float)
    assert total > 30.0
'''

_BUGGY_CHECKOUT = (
    "def checkout_total(cart, rate_pct):\n"
    "    subtotal = sum(cart)\n"
    "    return round(subtotal * (1 + rate_pct / 100.0), 2)\n"
)
_FIXED_CHECKOUT = (
    "from pricing_rules import apply_levy\n\n"
    "def checkout_total(cart, rate_pct):\n"
    "    return apply_levy(sum(cart), rate_pct)\n"
)

_MEMORY_REQUEST_1 = _CALC_REQUEST
_MEMORY_REQUEST_2 = (
    "Add a function multiply(a, b) to calc.py returning a * b, mirroring how add was built."
)

#: Fixed (query -> expected symbol) cases over this backend, for precision@k. The symbols
#: are real, stable public functions in the indexed tree.
_RETRIEVAL_CASES = [
    RetrievalCase(query="reciprocal rank fusion of ranked lists", expected_symbol="rrf_scores"),
    RetrievalCase(query="assemble the RAG stack from settings", expected_symbol="build_rag_stack"),
    RetrievalCase(query="ingest ADR files into long-term memory", expected_symbol="ingest_adrs"),
    RetrievalCase(
        query="precision at k over a fixed query set", expected_symbol="evaluate_retrieval"
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _new_repo(files: dict[str, str]) -> tuple[Path, str, str]:
    root = Path(tempfile.mkdtemp(prefix="eval-"))
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    git = Git(root)
    git.init()
    base = git.commit("eval fixture")
    return root, base, git.current_branch()


def _initial_state(
    root: Path, project_id: str, request: str, base: str, branch: str
) -> dict[str, Any]:
    state = dict(new_run_state(
        run_id=f"eval-{project_id}", project_id=project_id, user_request=request,
        workspace_path=str(root), autonomy_level="auto",
        max_tokens=None, max_steps=40, max_wall_clock_s=1800, started_at=now_iso(),
    ))
    state["base_commit"] = base
    state["work_branch"] = branch
    return state


def _cleanup(ctx: TaskContext, project_id: str) -> None:
    try:
        ctx.rag.store.clear_project(project_id)
        ctx.rag.long_term.clear_project(project_id)
        with ctx.rag.episodic._connect() as conn:
            conn.execute(
                f"DELETE FROM {ctx.rag.episodic._table} WHERE project_id = %s", (project_id,)
            )
    except Exception:  # cleanup is best-effort; never mask the task's own result
        pass


def _emit(schema: type, payload: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="e", name=emit_tool_name(schema), arguments=payload)]
    )


class _ScriptedProvider(LLMProvider):
    """Minimal deterministic provider returning queued responses in order.

    Local to the eval package so app code never imports from ``tests`` — used
    only to pin the planner/coder for the ``defect_injection`` task, where a
    reliable, reproducible defect is the whole point.
    """

    def __init__(self, responses: list[ChatResponse]) -> None:
        self.model = "scripted"
        self.capabilities = Capabilities(supports_tools=True, supports_json=True, max_context=8192)
        self._responses = list(responses)

    def chat(
        self, messages: Any, *, tools: Any = None, **params: Any
    ) -> ChatResponse:
        return self._responses.pop(0) if self._responses else ChatResponse(content="{}")

    def stream(self, messages: Any, **params: Any) -> Any:
        yield Chunk(delta="", done=True)

    def embed(self, texts: Any) -> list[Vector]:
        return [[0.0, 0.0, 0.0] for _ in texts]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def happy_path_task(ctx: TaskContext) -> RunReport:
    """Trivial self-contained build. Should always succeed; a clean-code false-flag signal."""
    project = f"eval-happy-{uuid.uuid4().hex[:6]}"
    root, base, branch = _new_repo({})
    try:
        graph = build_graph(
            ctx.settings, sandbox=ctx.sandbox, checkpointer=InMemorySaver(),
            retriever=ctx.rag.retriever, episodic=ctx.rag.episodic, long_term=ctx.rag.long_term,
            reviewer_provider=ctx.reviewer_provider,
        )
        res = run_graph(graph, _initial_state(root, project, _CALC_REQUEST, base, branch),
                        {"configurable": {"thread_id": project}})
    finally:
        _cleanup(ctx, project)
    return _graph_report("happy_path", HAPPY_PATH, res)


def rag_required_task(ctx: TaskContext) -> RunReport:
    """Solvable only by discovering the hidden apply_levy helper via retrieve."""
    project = f"eval-rag-{uuid.uuid4().hex[:6]}"
    root, base, branch = _new_repo({
        "pricing_rules.py": _PRICING_RULES, "test_checkout.py": _CHECKOUT_TEST,
    })
    try:
        ctx.rag.indexer.index_project(project, root)
        ctx.rag.retriever.invalidate(project)
        graph = build_graph(
            ctx.settings, sandbox=ctx.sandbox, checkpointer=InMemorySaver(),
            retriever=ctx.rag.retriever, episodic=ctx.rag.episodic, long_term=ctx.rag.long_term,
            reviewer_provider=ctx.reviewer_provider,
        )
        res = run_graph(graph, _initial_state(root, project, _RAG_REQUEST, base, branch),
                        {"configurable": {"thread_id": project}})
        checkout = root / "checkout.py"
        reused = checkout.exists() and "apply_levy" in checkout.read_text(encoding="utf-8")
    finally:
        _cleanup(ctx, project)
    report = _graph_report("rag_required", RAG_REQUIRED, res)
    report.expected_symbol_reused = reused
    return report


def defect_injection_task(ctx: TaskContext) -> RunReport:
    """Reviewer under test: a scripted coder plants a duplication defect the weak test
    can't catch; measures whether the (live) reviewer flags it."""
    project = f"eval-defect-{uuid.uuid4().hex[:6]}"
    root, base, branch = _new_repo({
        "pricing_rules.py": _PRICING_RULES, "test_checkout.py": _WEAK_TEST,
    })
    plan_payload = {
        "summary": "Add checkout_total to checkout.py.",
        "tasks": [{"id": "task-1", "title": "Add checkout_total", "kind": "create",
                   "description": _RAG_REQUEST, "target_paths": ["checkout.py"]}],
    }
    planner = _ScriptedProvider([_emit(Plan, plan_payload)])
    coder = _ScriptedProvider([
        _tool("write_file", path="checkout.py", content=_BUGGY_CHECKOUT),
        _tool("finish_task", summary="added checkout_total"),
        _tool("write_file", path="checkout.py", content=_FIXED_CHECKOUT),
        _tool("finish_task", summary="reused apply_levy per review"),
    ])
    try:
        ctx.rag.indexer.index_project(project, root)
        ctx.rag.retriever.invalidate(project)
        # Planner grounding must not consume the single scripted plan emission.
        settings = ctx.settings.model_copy(deep=True)
        settings.planner.grounding_steps = 0
        graph = build_graph(
            settings, sandbox=ctx.sandbox, checkpointer=InMemorySaver(),
            retriever=ctx.rag.retriever, episodic=ctx.rag.episodic, long_term=ctx.rag.long_term,
            planner_provider=planner, coder_provider=coder,
            reviewer_provider=ctx.reviewer_provider,  # None -> live reviewer
        )
        res = run_graph(graph, _initial_state(root, project, _RAG_REQUEST, base, branch),
                        {"configurable": {"thread_id": project}})
    finally:
        _cleanup(ctx, project)
    return _graph_report("defect_injection", DEFECT_INJECTION, res)


def cross_run_memory_task(ctx: TaskContext) -> RunReport:
    """Two runs on one project: run 1 writes episodic memory at finalize; measures whether
    run 2's planner context carries run 1's record."""
    project = f"eval-mem-{uuid.uuid4().hex[:6]}"
    root, base, branch = _new_repo({})
    try:
        graph = build_graph(
            ctx.settings, sandbox=ctx.sandbox, checkpointer=InMemorySaver(),
            retriever=ctx.rag.retriever, episodic=ctx.rag.episodic, long_term=ctx.rag.long_term,
            reviewer_provider=ctx.reviewer_provider,
        )
        run_graph(graph, _initial_state(root, project, _MEMORY_REQUEST_1, base, branch),
                  {"configurable": {"thread_id": f"{project}-1"}})

        # Faithful to what the plan node does on run 2: build the injected memory context
        # and check it now carries a "Previous Attempts" entry from run 1.
        context = build_planner_context(
            _MEMORY_REQUEST_2, project,
            long_term=ctx.rag.long_term, episodic=ctx.rag.episodic, settings=ctx.settings.planner,
        )
        influenced = "Previous Attempts" in context

        base2 = Git(root).current_commit() or base
        res = run_graph(graph, _initial_state(root, project, _MEMORY_REQUEST_2, base2, branch),
                        {"configurable": {"thread_id": f"{project}-2"}})
    finally:
        _cleanup(ctx, project)
    report = _graph_report("cross_run_memory", CROSS_RUN_MEMORY, res)
    report.memory_influenced = influenced
    return report


def retrieval_task(ctx: TaskContext) -> RunReport:
    """Deterministic precision@k over the indexed backend — the one gate-worthy metric."""
    project = f"eval-retr-{uuid.uuid4().hex[:6]}"
    try:
        report = index_and_evaluate(
            ctx.rag.indexer, ctx.rag.retriever, project, ctx.retrieval_repo, _RETRIEVAL_CASES, k=5
        )
        precision = report.precision_at_k
        misses = [c.expected_symbol for c in report.misses()]
    finally:
        _cleanup(ctx, project)
    return RunReport(
        task_id="retrieval", category=RETRIEVAL, precision_at_k=precision,
        notes=f"misses={misses}" if misses else "all cases hit",
    )


def _tool(name: str, **arguments: object) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="t", name=name, arguments=dict(arguments))]
    )


def _graph_report(task_id: str, category: str, res: Any) -> RunReport:
    return RunReport(
        task_id=task_id, category=category,
        status=res.status, verify_passed=res.verify_passed, verify_retries=res.verify_retries,
        review_verdicts=res.review_verdicts, review_flagged_blocking=res.review_flagged_blocking,
        retrieved_symbols=res.retrieved_symbols, steps=res.steps, wall_clock_s=res.wall_clock_s,
    )


#: The suite, in run order (retrieval last — it re-indexes a large repo).
ALL_TASKS: list[Callable[[TaskContext], RunReport]] = [
    happy_path_task,
    rag_required_task,
    defect_injection_task,
    cross_run_memory_task,
    retrieval_task,
]
