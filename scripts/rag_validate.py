"""Live Phase-3 validation: real models, full graph, RAG off vs on.

Not a unit test — a manual validation harness (run explicitly, reviewed by a
human). Uses only existing app code. Three parts, selectable via argv:

    python scripts/rag_validate.py part1     # live retrieval quality (real backend/app)
    python scripts/rag_validate.py abtest    # full-pipeline RAG off vs on on a fixture
    python scripts/rag_validate.py memory     # episodic + long-term memory read-back
    python scripts/rag_validate.py all        # all of the above

Honesty note: RAG-OFF still has Phase-1 `search_code` (ripgrep) + `read_file`,
so the A/B is *illustrative*, not a clean binary — it shows how each arm grounds
(grep vs `retrieve`) and whether verify passes. The rigorous "RAG adds semantic
capability" evidence is Part 1 (paraphrase queries a keyword search would miss).
Requires live Ollama (qwen2.5-coder + nomic-embed-text) and Postgres.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

# Memory-friendly on a 16 GB CPU box (learned in Phase 2 smoke). Coder temperature
# is set to 0 in main() by mutating settings (a nested MODELS__CODER__ env override
# would drop the required `model` field — pydantic-settings doesn't deep-merge it).
os.environ.setdefault("OLLAMA__DEFAULT_NUM_CTX", "4096")
os.environ.setdefault("SANDBOX__BACKEND", "subprocess")
os.environ.setdefault("GRAPH__MAX_VERIFY_RETRIES", "1")  # fast-fail the doomed RAG-off arm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.clock import now_iso
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.graph.build_graph import build_graph
from app.graph.state import new_run_state
from app.rag.factory import build_rag_stack
from app.tools.git import Git
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

BAR = "=" * 70


# ---------------------------------------------------------------------------
# Fixture: a helper with a non-guessable rule, named to disadvantage keyword grep
# ---------------------------------------------------------------------------
_PRICING_RULES = '''\
import math


def apply_levy(amount, rate_pct):
    """Add the mandatory surcharge to a base amount for a customer order.

    The surcharge is ``rate_pct`` percent of the amount; the surcharged total is
    then rounded UP to the nearest five cents, and a flat 0.30 processing fee is
    added. This is the standard rule applied to every customer order total.
    """
    surcharged = amount + amount * rate_pct / 100.0
    rounded_up = math.ceil(surcharged * 20) / 20
    return round(rounded_up + 0.30, 2)
'''

# Pre-written test: passes ONLY if checkout_total delegates to apply_levy's exact
# rule. A naive reimplementation (subtotal*(1+rate/100)) yields different numbers.
_TEST_CHECKOUT = '''\
from checkout import checkout_total
from pricing_rules import apply_levy


def test_matches_company_rule():
    assert checkout_total([4.00, 6.00], 8) == apply_levy(10.00, 8)


def test_exact_values():
    # apply_levy(10, 8): 10.80 -> ceil(216)/20=10.80 -> +0.30 = 11.10
    assert checkout_total([4.00, 6.00], 8) == 11.10
    # apply_levy(3.33, 7): 3.5631 -> ceil(72)/20=3.60 -> +0.30 = 3.90
    assert checkout_total([3.33], 7) == 3.90
'''

_TASK = (
    "Add a function checkout_total(cart, rate_pct) in a new file checkout.py. "
    "cart is a list of item prices. It must return the tax-inclusive total for "
    "the whole cart using the company's standard tax rule for customer orders. "
    "There is already a helper in this project that implements that exact rule — "
    "find it and use it rather than reimplementing the math."
)


def _write_fixture(root: Path) -> None:
    (root / "pricing_rules.py").write_text(_PRICING_RULES, encoding="utf-8")
    (root / "test_checkout.py").write_text(_TEST_CHECKOUT, encoding="utf-8")


# ---------------------------------------------------------------------------
# Part 1 — live retrieval quality against the real backend/app
# ---------------------------------------------------------------------------
def part1_live_retrieval() -> None:
    print(f"\n{BAR}\nPART 1 — Live retrieval (real nomic-embed-text over backend/app)\n{BAR}")
    settings = get_settings()
    rag = build_rag_stack(settings)
    project = f"backend-app-{uuid.uuid4().hex[:6]}"
    repo = Path(__file__).resolve().parents[1] / "backend"

    print(f"Indexing {repo} (real embeddings, one-time)...")
    stats = rag.indexer.index_project(project, repo)
    print(f"  indexed: {stats.files_indexed} files, {stats.chunks_indexed} chunks")

    exact = ["reciprocal_rank_fusion", "build_rag_stack", "EpisodicMemory"]
    semantic = [
        "how are source files split into chunks",
        "where is the checkpointer backend chosen",
        "how does the coder recover from a failed command",
    ]
    try:
        for label, queries in (("EXACT-SYMBOL", exact), ("SEMANTIC", semantic)):
            print(f"\n--- {label} queries ---")
            for q in queries:
                hits = rag.retriever.retrieve(project, q, k=3)
                print(f"\n  query: {q!r}")
                for h in hits:
                    snippet = h.content.strip().splitlines()[0][:80] if h.content.strip() else ""
                    print(f"    {h.score:+.3f}  {h.path}::{h.symbol}   | {snippet}")
    finally:
        rag.store.clear_project(project)


# ---------------------------------------------------------------------------
# Routing-reason derivation (from each node's real returned patch)
# ---------------------------------------------------------------------------
def _route_reason(node: str, patch: dict[str, Any]) -> tuple[str, str]:
    hitl = patch.get("hitl_request")
    kind = getattr(hitl, "kind", None)
    if node == "plan":
        return ("human_gate", f"gate: {kind}") if hitl else ("coder", "auto: no approval gate")
    if node == "coder":
        if hitl:
            return "human_gate", f"escalation ({kind})"
        return ("coder", "more tasks pending") if patch.get("current_task_id") else (
            "verify", "all tasks done")
    if node == "verify":
        if hitl:
            return "human_gate", "verify retries exhausted -> escalation"
        vr = patch.get("verify_result")
        passed = getattr(vr, "passed", False)
        return ("review", "verify PASSED") if passed else ("coder", "verify FAILED -> fix mode")
    if node == "review":
        if hitl:
            return "human_gate", "escalation"
        verdict = getattr(patch.get("review"), "verdict", None)
        return ("finalize", "approved") if verdict == "approved" else ("coder", "changes_requested")
    if node == "human_gate":
        return "finalize/coder/plan", "per human decision (auto-abort here)"
    if node == "finalize":
        return "END", f"status={patch.get('status')}"
    return "?", "?"


def _run_arm(label: str, *, rag_on: bool) -> None:
    print(f"\n{BAR}\n{label}\n{BAR}")
    settings = get_settings()
    root = Path(tempfile.mkdtemp(prefix="ragval-"))
    _write_fixture(root)
    git = Git(root)
    git.init()
    base = git.commit("existing project")

    rag = build_rag_stack(settings)
    project = f"ragval-{uuid.uuid4().hex[:6]}"
    retriever = None
    if rag_on:
        stats = rag.indexer.index_project(project, root)
        print(f"indexed fixture: {stats.files_indexed} files, {stats.chunks_indexed} chunks")
        retriever = rag.retriever

    graph = build_graph(
        settings, checkpointer=InMemorySaver(), retriever=retriever, episodic=rag.episodic
    )
    state = dict(new_run_state(
        run_id=f"run-{label.lower().replace(' ', '-')}", project_id=project,
        user_request=_TASK, workspace_path=str(root), autonomy_level="auto",
        max_tokens=None, max_steps=40, max_wall_clock_s=1800, started_at=now_iso(),
    ))
    state["base_commit"] = base
    state["work_branch"] = git.current_branch()
    config = {"configurable": {"thread_id": project}}

    print("\nNODE TIMELINE (task requires discovering the existing helper):")
    payload: Any = state
    for _ in range(50):  # generous cap; graph's own budgets/recursion bound it
        interrupted = False
        for chunk in graph.stream(payload, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                print("  -- interrupt (escalation) -> auto-abort --")
                interrupted = True
                break
            for node, patch in chunk.items():
                nxt, reason = _route_reason(node, patch or {})
                rc = patch.get("retrieved_context") if patch else None
                tag = f"  [retrieved {len(rc)} chunk(s)]" if rc else ""
                print(f"  {node:<11} -> {nxt:<20} ({reason}){tag}")
        if not interrupted:
            break
        payload = Command(resume="abort")

    final = graph.get_state(config).values
    print(f"\nFINAL status: {final.get('status')}")
    vr = final.get("verify_result")
    if vr is not None:
        print(f"verify: {'PASS' if vr.passed else 'FAIL'} — {vr.summary}")
    rc = final.get("retrieved_context") or []
    print(f"retrieved_context: {len(rc)} chunk(s)" + (
        f" (symbols: {[c.symbol for c in rc]})" if rc else ""))
    checkout = root / "checkout.py"
    print("\nGENERATED checkout.py:\n" + (
        checkout.read_text(encoding="utf-8") if checkout.exists() else "  (not created)"))
    rag.store.clear_project(project)


def abtest() -> None:
    _run_arm("RUN A — RAG OFF", rag_on=False)
    _run_arm("RUN B — RAG ON", rag_on=True)


# ---------------------------------------------------------------------------
# Part 4 — memory read-back
# ---------------------------------------------------------------------------
def memory_check() -> None:
    print(f"\n{BAR}\nPART 4 — Memory (episodic + long-term read-back)\n{BAR}")
    settings = get_settings()
    rag = build_rag_stack(settings)
    project = f"mem-{uuid.uuid4().hex[:6]}"

    from app.memory.episodic import RunRecord

    rag.episodic.ensure_schema()
    rag.episodic.record(RunRecord(
        run_id="demo-run", project_id=project, status="succeeded",
        summary="built checkout_total", tasks_total=1, tasks_done=1))
    recent = rag.episodic.recent(project)
    print(f"episodic.recent -> {[(r.run_id, r.status, r.summary) for r in recent]}")

    rag.long_term.ensure_schema()
    rag.long_term.write(project, "This project rounds order totals UP to the nearest 5 cents.",
                        kind="convention")
    hits = rag.long_term.search(project, "how are totals rounded", k=1)
    print(f"long_term.search('how are totals rounded') -> "
          f"{[(h.kind, h.text) for h in hits]}")


def main() -> int:
    settings = get_settings()  # lru_cached — mutation persists across all callers
    settings.models.coder.temperature = 0.0  # deterministic A/B
    configure_logging(settings)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("part1", "all"):
        part1_live_retrieval()
    if which in ("abtest", "all"):
        abtest()
    if which in ("memory", "all"):
        memory_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
