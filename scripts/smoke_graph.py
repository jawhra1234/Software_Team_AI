"""Phase-2 smoke: drive the full compiled graph with the real local model, end to end.

Runs a real request through plan -> coder -> verify -> review -> finalize using the
actual configured model (Ollama, default qwen2.5-coder) and the default sandbox
(Docker), printing each node transition as it happens. Uses a real SQLite
checkpointer, exercising the same default configuration a real run would use.

    # from repo root, with the backend venv active:
    python scripts/smoke_graph.py

Requires a running Ollama with the configured model pulled, and (with the default
sandbox backend) Docker with the aiswe-sandbox:latest image built. Exits non-zero if
the run doesn't reach "succeeded".
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

# Windows consoles default to a legacy codepage that can't encode the arrows
# printed below; force UTF-8 so this runs the same everywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Make the backend package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from langgraph.types import Command  # noqa: E402

from app.core.clock import now_iso  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.graph.build_graph import build_graph  # noqa: E402
from app.graph.checkpointer import build_checkpointer  # noqa: E402
from app.graph.events import GraphEvent  # noqa: E402
from app.graph.state import new_run_state  # noqa: E402
from app.tools.git import Git  # noqa: E402

REQUEST = (
    "Create a Python module calc.py with a function add(a, b) that returns a + b, "
    "and a pytest test file test_calc.py that asserts add(2, 3) == 5."
)


class PrintingSink:
    """Prints each node transition live, with a short summary of what changed."""

    def emit(self, event: GraphEvent) -> None:
        if event.kind == "node_start":
            print(f"\n▶ {event.node}")
            return
        if event.kind != "node_end":
            return

        data = event.data
        if data.get("skipped"):
            print(f"  skipped — {data.get('reason')}")
            return

        if event.node == "plan" and data.get("plan"):
            plan = data["plan"]
            print(f'  plan: "{plan["summary"]}" ({len(plan["tasks"])} task(s))')
        elif event.node == "coder":
            print(f"  next task: {data.get('current_task_id')!r}")
        elif event.node == "verify" and data.get("verify_result"):
            vr = data["verify_result"]
            print(f"  verify: {'PASS' if vr['passed'] else 'FAIL'} — {vr['summary']}")
        elif event.node == "review" and data.get("review"):
            rv = data["review"]
            print(f"  review: {rv['verdict']} — {rv['summary']}")
        elif event.node == "finalize":
            print(f"  status: {data.get('status')}")

        if data.get("hitl_request"):
            print(f"  → human gate requested: {data['hitl_request']['kind']}")


def main() -> int:
    settings = get_settings()
    configure_logging(settings)
    log = get_logger("smoke_graph")

    workspace_path = Path(tempfile.mkdtemp(prefix="aiswe-smoke-"))
    git = Git(workspace_path)
    git.init()
    base_commit = git.commit("chore: init")

    settings.checkpointer.sqlite_path = str(workspace_path.parent / f"{workspace_path.name}.sqlite")

    print(f"model (coder):   {settings.models.coder.model}")
    print(f"model (planner): {settings.models.planner.model}")
    print(f"sandbox backend: {settings.sandbox.backend}")
    print(f"checkpointer:    {settings.checkpointer.backend} ({settings.checkpointer.sqlite_path})")
    print(f"workspace:       {workspace_path}")
    print(f"\nrequest: {REQUEST}")

    with build_checkpointer(settings) as checkpointer:
        graph = build_graph(settings, checkpointer=checkpointer, sink=PrintingSink())

        state = new_run_state(
            run_id="smoke-graph-1",
            project_id="smoke-graph",
            user_request=REQUEST,
            workspace_path=str(workspace_path),
            autonomy_level="auto",
            max_tokens=None,
            max_steps=30,
            max_wall_clock_s=900,
            started_at=now_iso(),
        )
        state["base_commit"] = base_commit
        state["work_branch"] = git.current_branch()

        config = {"configurable": {"thread_id": "smoke-graph-1"}}
        log.info("smoke_graph_start")
        result: dict[str, Any] = graph.invoke(state, config=config)  # type: ignore[assignment]

        if "__interrupt__" in result:
            print(f"\n(!) Unexpected escalation: {result['__interrupt__'][0].value}")
            print("Auto-aborting for a clean report...")
            result = graph.invoke(Command(resume={"decision": "abort"}), config=config)  # type: ignore[assignment]

    print("\n" + "=" * 60)
    print(f"Final status: {result.get('status')}")
    if result.get("plan"):
        for task in result["plan"].tasks:
            print(f"  task '{task.id}': {task.status}")
    if result.get("verify_result"):
        vr = result["verify_result"]
        print(f"Verify: {'PASS' if vr.passed else 'FAIL'} — {vr.summary}")
    if result.get("review"):
        print(f"Review: {result['review'].verdict} — {result['review'].summary}")
    print(f"\nDiff summary:\n{result.get('diff_summary', '(none)')}")
    print(f"\nWorkspace files: {sorted(p.name for p in workspace_path.iterdir())}")

    ok = result.get("status") == "succeeded"
    print(f"\n{'OK' if ok else 'FAILED'}: full graph run {'succeeded' if ok else 'did not succeed'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
