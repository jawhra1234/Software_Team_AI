"""Live end-to-end validation of the full pipeline WITH memory wired in (Task 3.13).

Not a unit/integration test — a production-style validation harness. Drives the
real compiled graph with the real qwen2.5-coder:7b model twice on the same
project, to prove that memory written by run 1 actually influences planning in
run 2. Captures evidence at every stage: injected planner context, long-term +
episodic reads, repo `retrieve`, the plan, tool calls, verify, and the episodic
record finalize writes.

    python scripts/memory_e2e.py

Requires live Ollama (qwen2.5-coder + nomic-embed-text) + Postgres/pgvector.
Reuses the fixture + timeline helpers from rag_validate.py.
"""

from __future__ import annotations

# ruff: noqa: E402  — path bootstrap (sys.path + env) must precede app imports.
import os

# Bound the coder so two full runs finish in reasonable wall-clock on a 16 GB box.
os.environ.setdefault("CODER__MAX_WALL_CLOCK_S", "360")
os.environ.setdefault("CODER__MAX_STEPS_PER_TASK", "12")

import sys
import uuid
from pathlib import Path
from typing import Any

# Put both scripts/ (for rag_validate) and backend/ (for app.*) on the path up
# front, so the import order below is independent of any module side effect.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "backend"))

# rag_validate's import also sets num_ctx/subprocess-sandbox/verify-retry env.
from app.core.clock import now_iso
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.graph.build_graph import build_graph
from app.graph.planning_context import build_planner_context
from app.graph.state import new_run_state
from app.memory.episodic import EpisodicMemory, RunRecord
from app.memory.ingest import ingest_adrs
from app.memory.long_term import LongTermMemory
from app.rag.factory import build_rag_stack
from app.tools.git import Git
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rag_validate import BAR, _route_reason, _write_fixture

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Run 1 = the checkout task (find + reuse apply_levy). Run 2 = a *related* task
# that references the earlier work, so run 1's episodic summary is relevant.
_TASK_1 = (
    "Add checkout_total(cart, rate_pct) in a new file checkout.py. cart is a list "
    "of item prices; return the tax-inclusive order total using the company's "
    "standard order-total rule. A helper implementing that exact rule already "
    "exists in this project — find it with retrieve and reuse it."
)
_TASK_2 = (
    "Add checkout_receipt(cart, rate_pct) in a new file receipt.py that returns a "
    "one-line receipt string for the order total, using the SAME company "
    "order-total rule that checkout_total already uses. Reuse the existing helper."
)


class RecordingLongTerm:
    """Wraps LongTermMemory, printing every semantic search the planner makes."""

    def __init__(self, inner: LongTermMemory) -> None:
        self._inner = inner

    def search(self, project_id: str, query: str, k: int = 5) -> list[Any]:
        hits = self._inner.search(project_id, query, k=k)
        print(f"  [long_term.search] q={query[:55]!r} k={k} -> {len(hits)} hit(s)")
        for h in hits:
            print(f"      ({h.kind}, {h.score:+.3f}) {h.text.splitlines()[0][:72]!r}")
        return hits

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class RecordingEpisodic:
    """Wraps EpisodicMemory, printing relevance reads and the record finalize writes."""

    def __init__(self, inner: EpisodicMemory) -> None:
        self._inner = inner

    def relevant(
        self, project_id: str, query: str, k: int = 3, *, candidate_window: int = 50
    ) -> list[RunRecord]:
        runs = self._inner.relevant(project_id, query, k=k, candidate_window=candidate_window)
        print(f"  [episodic.relevant] q={query[:55]!r} k={k} -> {len(runs)} run(s)")
        for r in runs:
            print(f"      [{r.status}] {r.summary[:60]!r} ({r.tasks_done}/{r.tasks_total})")
        return runs

    def record(self, record: RunRecord) -> None:
        print(f"  [episodic.record] run={record.run_id} status={record.status} "
              f"summary={record.summary[:55]!r} tasks={record.tasks_done}/{record.tasks_total}")
        self._inner.record(record)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _run_once(
    *, label: str, request: str, run_id: str, project: str, root: Path,
    rag: Any, lt: RecordingLongTerm, ep: RecordingEpisodic, settings: Any,
) -> dict[str, Any]:
    print(f"\n{BAR}\n{label}\n{BAR}")

    # 1) Show the exact memory block the plan node will inject (same function it
    #    calls; the node rebuilds it identically during the run below).
    print("\n--- STAGE 1: PLANNER MEMORY CONTEXT (what will be injected) ---")
    ctx = build_planner_context(
        request, project, long_term=lt, episodic=ep, settings=settings.planner
    )
    print("\nINJECTED CONTEXT:\n" + ("\n".join("    " + ln for ln in ctx.splitlines())
                                     if ctx else "    (empty — no memory available yet)"))

    git = Git(root)
    base = git.current_commit()
    graph = build_graph(
        settings, checkpointer=InMemorySaver(),
        retriever=rag.retriever, episodic=ep, long_term=lt,
    )
    state = dict(new_run_state(
        run_id=run_id, project_id=project, user_request=request,
        workspace_path=str(root), autonomy_level="auto",
        max_tokens=None, max_steps=30, max_wall_clock_s=1200, started_at=now_iso(),
    ))
    state["base_commit"] = base
    state["work_branch"] = git.current_branch()
    config: Any = {"configurable": {"thread_id": run_id}}

    print("\n--- STAGE 2-5: NODE TIMELINE (plan -> retrieve -> coder -> verify -> finalize) ---")
    payload: Any = state
    for _ in range(60):
        interrupted = False
        for chunk in graph.stream(payload, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                print("  -- interrupt (escalation) -> auto-abort --")
                interrupted = True
                break
            for node, patch in chunk.items():
                patch = patch or {}
                nxt, reason = _route_reason(node, patch)
                rc = patch.get("retrieved_context")
                tag = f"   [retrieved {len(rc)}: {[c.symbol for c in rc]}]" if rc else ""
                print(f"  {node:<11} -> {nxt:<20} ({reason}){tag}")
                if node == "plan" and patch.get("plan"):
                    p = patch["plan"]
                    print(f"     PLAN v{p.version}: {p.summary!r} — {len(p.tasks)} task(s)")
                    for t in p.tasks:
                        print(f"        - {t.id} [{t.kind}]: {t.title} -> {t.target_paths}")
        if not interrupted:
            break
        payload = Command(resume="abort")

    final = graph.get_state(config).values
    print(f"\nFINAL status: {final.get('status')}")
    vr = final.get("verify_result")
    if vr is not None:
        print(f"verify: {'PASS' if vr.passed else 'FAIL'} — {vr.summary[:120]}")
    for f in sorted(root.glob("*.py")):
        if f.name == "pricing_rules.py":  # the pre-existing fixture helper, not agent output
            continue
        print(f"\n{f.name}:\n" + f.read_text(encoding="utf-8")[:600])
    return final


_SIMPLE_TASK = (
    "Create a new file calc.py containing a function add(a, b) that returns a + b. "
    "Also create test_calc.py with a test that asserts add(2, 3) == 5. Then run the "
    "tests with `python -m pytest -q` to confirm they pass, and finish."
)


def _seed_memory(rag: Any, project: str) -> None:
    rag.long_term.ensure_schema()
    rag.episodic.ensure_schema()
    n_adr = ingest_adrs(rag.long_term, project, _REPO_ROOT / "docs" / "adr")
    print(f"seeded long-term memory: {n_adr} ADR-derived decisions")
    rag.long_term.write(
        project,
        "Customer order totals use apply_levy: add the surcharge, round UP to the "
        "nearest 5 cents, then add a flat 0.30 processing fee.",
        kind="convention",
    )
    print("seeded long-term memory: +1 domain convention (apply_levy rule)")


def _cleanup(rag: Any, project: str) -> None:
    rag.store.clear_project(project)
    rag.long_term.clear_project(project)
    with rag.episodic._connect() as conn:
        conn.execute(f"DELETE FROM {rag.episodic._table} WHERE project_id = %s", (project,))


def cross_run_scenario(rag: Any, settings: Any) -> None:
    """Two related runs — proves run 1's episodic record influences run 2's planning."""
    import tempfile
    project = f"mem-e2e-{uuid.uuid4().hex[:6]}"
    root = Path(tempfile.mkdtemp(prefix="mem-e2e-"))
    _write_fixture(root)
    git = Git(root)
    git.init()
    git.commit("existing project with pricing_rules.apply_levy")

    print(f"{BAR}\nSETUP\n{BAR}")
    stats = rag.indexer.index_project(project, root)
    print(f"indexed fixture for repo-RAG: {stats.files_indexed} files, "
          f"{stats.chunks_indexed} chunks")
    _seed_memory(rag, project)

    lt, ep = RecordingLongTerm(rag.long_term), RecordingEpisodic(rag.episodic)
    try:
        _run_once(label="RUN 1 — first request (episodic empty)", request=_TASK_1,
                  run_id=f"{project}-run1", project=project, root=root,
                  rag=rag, lt=lt, ep=ep, settings=settings)
        print(f"\n{BAR}\nBETWEEN RUNS — episodic memory now holds run 1\n{BAR}")
        for r in rag.episodic.recent(project):
            print(f"  episodic.recent -> [{r.status}] {r.summary!r} "
                  f"({r.tasks_done}/{r.tasks_total})")
        _run_once(label="RUN 2 — related request (episodic warm + LT seeded)", request=_TASK_2,
                  run_id=f"{project}-run2", project=project, root=root,
                  rag=rag, lt=lt, ep=ep, settings=settings)
    finally:
        _cleanup(rag, project)


def simple_scenario(rag: Any, settings: Any) -> None:
    """One self-contained, easy task — the coder should CONVERGE, so the full happy
    path (coder -> verify PASS -> review -> finalize succeeded) is exercised with
    memory still wired in. A prior 'succeeded' episodic record is seeded so both
    memory sections render."""
    import tempfile
    project = f"mem-simple-{uuid.uuid4().hex[:6]}"
    root = Path(tempfile.mkdtemp(prefix="mem-simple-"))
    (root / "README.md").write_text("# demo project\n", encoding="utf-8")
    git = Git(root)
    git.init()
    git.commit("empty starter project")

    print(f"{BAR}\nSETUP (simple full-pipeline run)\n{BAR}")
    stats = rag.indexer.index_project(project, root)
    print(f"indexed repo for repo-RAG: {stats.files_indexed} files, {stats.chunks_indexed} chunks")
    _seed_memory(rag, project)
    # A realistic prior run so "Previous Attempts" also renders this single run.
    rag.episodic.record(RunRecord(
        run_id="prior-seed", project_id=project, status="succeeded",
        summary="Added a small utility module with a passing pytest test.",
        tasks_total=2, tasks_done=2))
    print("seeded episodic memory: +1 prior successful run")

    lt, ep = RecordingLongTerm(rag.long_term), RecordingEpisodic(rag.episodic)
    try:
        final = _run_once(
            label="SIMPLE — full pipeline on a trivial self-contained task",
            request=_SIMPLE_TASK, run_id=f"{project}-run1", project=project, root=root,
            rag=rag, lt=lt, ep=ep, settings=settings)
        status = final.get("status")
        verdict = "HAPPY PATH — coder converged" if status == "succeeded" else "did not converge"
        print(f"\n>>> PIPELINE RESULT: status={status} ({verdict})")
    finally:
        _cleanup(rag, project)


def main() -> int:
    settings = get_settings()  # lru_cached; mutation persists to graph builders
    settings.models.coder.temperature = 0.0
    configure_logging(settings)
    rag = build_rag_stack(settings)

    which = sys.argv[1] if len(sys.argv) > 1 else "cross"
    if which == "simple":
        simple_scenario(rag, settings)
    else:
        cross_run_scenario(rag, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
