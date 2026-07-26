"""Graph-run capture (Phase 5).

``run_graph`` streams one compiled-graph run to completion and reduces it to a
task-agnostic :class:`GraphRunResult` — the per-cycle review verdicts, the
symbols surfaced by ``retrieve``, verify pass/retries, node/step count, and
wall-clock. Escalation interrupts are auto-aborted (the eval never has a human
to answer), so every run reaches a terminal state. This mirrors the capture the
Phase 3-4 live scripts already do, factored into one reusable place.

Task-specific scoring (did the RAG task reuse the helper? did run 2 see run 1's
memory?) is layered on top by ``app/evals/tasks.py`` using ``final_state``; this
module stays task-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command

# Verdicts that mean the reviewer blocked on substance (a blocker/major, or a
# fundamental rejection). The review node already overrides the effective
# verdict from issue severities, so this is a faithful "did it block" signal.
_BLOCKING_VERDICTS = frozenset({"changes_requested", "rejected"})


@dataclass
class GraphRunResult:
    """Task-agnostic capture from one streamed graph run."""

    status: str
    verify_passed: bool | None
    verify_retries: int
    review_verdicts: list[str] = field(default_factory=list)
    retrieved_symbols: list[str] = field(default_factory=list)
    steps: int = 0
    wall_clock_s: float = 0.0
    final_state: dict[str, Any] = field(default_factory=dict)

    @property
    def review_flagged_blocking(self) -> bool:
        return any(v in _BLOCKING_VERDICTS for v in self.review_verdicts)


def run_graph(graph: Any, initial_state: dict[str, Any], config: dict[str, Any]) -> GraphRunResult:
    """Stream one run to a terminal state, auto-aborting any escalation interrupt."""
    review_verdicts: list[str] = []
    retrieved: list[str] = []
    seen_symbols: set[str] = set()

    started = time.perf_counter()
    payload: Any = initial_state
    for _ in range(60):  # generous outer cap; the graph's own budgets bound real work
        interrupted = False
        for chunk in graph.stream(payload, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupted = True
                break
            for node, patch in chunk.items():
                patch = patch or {}
                if node == "review" and patch.get("review") is not None:
                    review_verdicts.append(patch["review"].verdict)
                for chunk_obj in patch.get("retrieved_context") or []:
                    sym = getattr(chunk_obj, "symbol", None)
                    if sym and sym not in seen_symbols:
                        seen_symbols.add(sym)
                        retrieved.append(sym)
        if not interrupted:
            break
        payload = Command(resume="abort")
    wall_clock_s = time.perf_counter() - started

    final = graph.get_state(config).values
    verify_result = final.get("verify_result")
    return GraphRunResult(
        status=str(final.get("status", "unknown")),
        verify_passed=getattr(verify_result, "passed", None),
        verify_retries=int(final.get("retries", {}).get("verify", 0)),
        review_verdicts=review_verdicts,
        retrieved_symbols=retrieved,
        steps=len(final.get("node_history", [])),
        wall_clock_s=wall_clock_s,
        final_state=final,
    )
