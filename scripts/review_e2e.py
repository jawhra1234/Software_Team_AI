"""Live end-to-end validation of the review/self-correction loop (Task 4.11).

Not a unit/integration test — a production-style validation harness. Drives the
real compiled graph with a **live reviewer** (real qwen2.5-coder:7b) against a
seeded real defect: the coder's first attempt reimplements an existing helper's
logic instead of reusing it — an architecture/duplication issue the project's
*weak* pre-written test doesn't catch (verify passes on both the buggy and the
fixed version), so only the reviewer can catch it. The coder side is scripted
(deterministic) so the specific defect + fix are reproducible; the reviewer is
live throughout, so what it does with the defect is genuine model behavior, not
a scripted outcome.

Proves live: reviewer flags a real defect (blocker/major) -> coder receives the
reviewer's targeted finding (not a re-plan) -> verify re-runs -> reviewer
re-reviews -> approves -> finalize succeeds. If the live reviewer doesn't behave
that way this run, the harness reports exactly what happened instead — it does
not force the narrative.

    python scripts/review_e2e.py

Requires live Ollama (qwen2.5-coder) + Postgres/pgvector.
"""

from __future__ import annotations

# ruff: noqa: E402  — path bootstrap (sys.path) must precede app imports.
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "backend"))

# rag_validate's import also sets num_ctx/subprocess-sandbox/verify-retry env.
from app.core.clock import now_iso
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.graph.build_graph import build_graph
from app.graph.state import Plan, new_run_state
from app.providers.base import Capabilities, ChatResponse, ToolCall
from app.providers.structured import emit_tool_name
from app.rag.factory import build_rag_stack
from app.tools.git import Git
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rag_validate import BAR, _route_reason
from tests.fakes import FakeProvider

_PRICING_HELPER = '''\
def apply_levy(amount, rate_pct):
    """Add the mandatory surcharge to an order total (company standard rule).

    Surcharge is rate_pct percent of the amount, plus a flat 0.30 processing fee.
    """
    return round(amount + amount * rate_pct / 100.0 + 0.30, 2)
'''

# Deliberately WEAK: only checks the return is a plausible number greater than
# the subtotal — passes for BOTH the buggy reimplementation and the real fix,
# so verify alone cannot catch the architecture defect. Only the reviewer can.
_WEAK_TEST = '''\
from checkout import checkout_total


def test_returns_a_number_above_subtotal():
    total = checkout_total([10.0, 20.0], 8)
    assert isinstance(total, float)
    assert total > 30.0
'''

_TASK = (
    "Add a function checkout_total(cart, rate_pct) in a new file checkout.py. "
    "cart is a list of item prices. It must return the tax-inclusive total for "
    "the whole cart using the company's standard tax rule for customer orders."
)

# The coder's first attempt: reimplements the tax math independently instead of
# reusing pricing_rules.apply_levy — an architecture/duplication defect, not a
# test failure (the weak test still passes).
_BUGGY_CHECKOUT = (
    "def checkout_total(cart, rate_pct):\n"
    "    subtotal = sum(cart)\n"
    "    return round(subtotal * (1 + rate_pct / 100.0), 2)\n"
)
# The corrected version: delegates to the existing helper.
_FIXED_CHECKOUT = (
    "from pricing_rules import apply_levy\n\n"
    "def checkout_total(cart, rate_pct):\n"
    "    return apply_levy(sum(cart), rate_pct)\n"
)


def _plan_payload() -> dict[str, Any]:
    return {
        "summary": "Add checkout_total(cart, rate_pct) to checkout.py.",
        "tasks": [{
            "id": "task-1", "title": "Add checkout_total", "kind": "create",
            "description": _TASK, "target_paths": ["checkout.py"],
            "acceptance_criteria": ["checkout.py defines checkout_total(cart, rate_pct)"],
        }],
    }


def _write_file(path: str, content: str) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(id="1", name="write_file",
                                         arguments={"path": path, "content": content})]
    )


def _finish(summary: str) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[ToolCall(id="2", name="finish_task", arguments={"summary": summary})],
    )


def main() -> int:
    settings = get_settings()  # lru_cached; mutation persists to graph builders
    settings.models.coder.temperature = 0.0
    settings.models.reviewer.temperature = 0.0
    # The planner is scripted with exactly one queued response (the Plan emission);
    # a nonzero grounding_steps would consume it as a grounding tool call instead
    # (the same class of bug already fixed in test_graph_retrieval_wiring.py).
    settings.planner.grounding_steps = 0
    configure_logging(settings)

    root = Path(tempfile.mkdtemp(prefix="review-e2e-"))
    (root / "pricing_rules.py").write_text(_PRICING_HELPER, encoding="utf-8")
    (root / "test_checkout.py").write_text(_WEAK_TEST, encoding="utf-8")
    git = Git(root)
    git.init()
    base = git.commit("existing project with pricing_rules.apply_levy + a weak test")

    rag = build_rag_stack(settings)
    project = f"review-e2e-{uuid.uuid4().hex[:6]}"
    stats = rag.indexer.index_project(project, root)
    print(f"{BAR}\nSETUP\n{BAR}")
    print(f"indexed fixture: {stats.files_indexed} files, {stats.chunks_indexed} chunks")

    # Scripted planner (deterministic — planning isn't what this harness validates)
    # and scripted coder (deterministic defect + fix). The reviewer provider is
    # left unset -> build_graph resolves it to the REAL live Ollama reviewer.
    caps = Capabilities(supports_tools=True, supports_json=True, max_context=8192)
    planner_provider = FakeProvider(
        capabilities=caps, responses=[
            ChatResponse(content="", tool_calls=[
                ToolCall(id="p", name=emit_tool_name(Plan), arguments=_plan_payload())
            ])
        ],
    )
    coder_provider = FakeProvider(
        capabilities=caps, responses=[
            _write_file("checkout.py", _BUGGY_CHECKOUT),
            _finish("added checkout_total"),
            # Fix-mode attempt (only consumed if the reviewer requests changes):
            _write_file("checkout.py", _FIXED_CHECKOUT),
            _finish("delegated to apply_levy per review feedback"),
        ],
    )

    graph = build_graph(
        settings, checkpointer=InMemorySaver(),
        retriever=rag.retriever, episodic=rag.episodic,
        planner_provider=planner_provider, coder_provider=coder_provider,
    )
    state = dict(new_run_state(
        run_id=f"run-{project}", project_id=project, user_request=_TASK,
        workspace_path=str(root), autonomy_level="auto",
        max_tokens=None, max_steps=40, max_wall_clock_s=1800, started_at=now_iso(),
    ))
    state["base_commit"] = base
    state["work_branch"] = git.current_branch()
    config: Any = {"configurable": {"thread_id": project}}

    print("\nNODE TIMELINE (live reviewer; scripted planner+coder):")
    review_cycle = 0
    payload: Any = state
    try:
        for _ in range(30):
            interrupted = False
            for chunk in graph.stream(payload, config=config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    print("  -- interrupt (escalation) -> auto-abort --")
                    interrupted = True
                    break
                for node, patch in chunk.items():
                    patch = patch or {}
                    nxt, reason = _route_reason(node, patch)
                    print(f"  {node:<11} -> {nxt:<20} ({reason})")
                    if node == "review" and patch.get("review"):
                        review_cycle += 1
                        r = patch["review"]
                        print(f"     REVIEW #{review_cycle}: verdict={r.verdict!r} "
                              f"summary={r.summary[:100]!r}")
                        for issue in r.issues:
                            print(f"        [{issue.severity}] {issue.description[:110]}"
                                  + (f" ({issue.file})" if issue.file else ""))
            if not interrupted:
                break
            payload = Command(resume="abort")
    finally:
        rag.store.clear_project(project)

    final = graph.get_state(config).values
    print(f"\nFINAL status: {final.get('status')}")
    vr = final.get("verify_result")
    if vr is not None:
        print(f"verify: {'PASS' if vr.passed else 'FAIL'} — {vr.summary[:120]}")
    checkout = root / "checkout.py"
    print("\nFINAL checkout.py:\n" + (
        checkout.read_text(encoding="utf-8") if checkout.exists() else "  (not created)"))

    print(f"\n{BAR}\nCONCLUSION\n{BAR}")
    used_apply_levy = checkout.exists() and "apply_levy" in checkout.read_text(encoding="utf-8")
    if review_cycle >= 2 and final.get("status") == "succeeded" and used_apply_levy:
        print("Reviewer caught the duplication, coder applied the targeted fix, "
              "verify passed, reviewer approved. Full catch->fix->approve cycle demonstrated.")
    elif review_cycle == 1 and final.get("status") == "succeeded":
        print("Reviewer approved on the FIRST pass (did not flag the duplication this run). "
              "This is genuine live-model behavior, not a pipeline defect — the loop is "
              "proven elsewhere (hermetic tests force the finding); this run's reviewer "
              "judged the duplication as acceptable or didn't ground enough to notice.")
    else:
        print(f"Did not reach the expected 2-cycle catch->fix->approve pattern "
              f"(review_cycle={review_cycle}, status={final.get('status')}). See timeline above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
