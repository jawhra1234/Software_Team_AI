"""Run manager for the Mission-Control API (Phase 6).

Drives the **existing** compiled graph (``build_graph``) — nothing about the
graph, nodes, or agents changes. Each run executes on its own worker thread; the
graph's ``EventSink`` seam (Task 2.14, built for exactly this) is adapted into an
append-only, replayable event log that the SSE endpoint tails. Human-in-the-loop
pauses surface as ``interrupt`` events and are answered by ``Command(resume=...)``
— the same mechanism the tests and CLI scripts use, now driven over HTTP.

Concurrency model: the graph is synchronous, so it runs in a thread; events land
in a lock-guarded list (so late SSE subscribers and reconnections replay from the
start); the worker blocks on a per-run resume queue while waiting for a human.
"""

from __future__ import annotations

import queue
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.core.clock import now_iso
from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.build_graph import build_graph
from app.graph.events import GraphEvent
from app.graph.state import new_run_state
from app.memory.episodic import EpisodicMemory
from app.memory.long_term import LongTermMemory
from app.providers.base import LLMProvider
from app.rag.retriever import Retriever
from app.tools.git import Git
from app.tools.sandbox import Sandbox, get_sandbox

log = get_logger("api.run_manager")

# The per-run workspace ignores build/test artifacts so the UI diff shows only the
# agent's real source changes, not __pycache__/*.pyc created by running the tests.
_WORKSPACE_GITIGNORE = "__pycache__/\n*.pyc\n.pytest_cache/\n"

RunStatus = str  # "queued" | "running" | "waiting_human" | "done" | "error"


@dataclass
class Run:
    """Live state of one Mission-Control run (in-memory; a process restart drops it)."""

    run_id: str
    request: str
    autonomy: str
    workspace_path: str
    events: list[dict[str, Any]] = field(default_factory=list)
    status: RunStatus = "running"
    pending_interrupt: dict[str, Any] | None = None
    final_state: dict[str, Any] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _resume_q: queue.Queue[Any] = field(default_factory=queue.Queue, repr=False)
    _finished: threading.Event = field(default_factory=threading.Event, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "request": self.request,
                "autonomy": self.autonomy,
                "status": self.status,
                "pending_interrupt": self.pending_interrupt,
                "event_count": len(self.events),
                "final_state": self.final_state,
            }

    def events_from(self, index: int) -> tuple[list[dict[str, Any]], bool]:
        """New events since ``index`` and whether the run has finished (for SSE tailing)."""
        with self._lock:
            return self.events[index:], self._finished.is_set()

    # -- internal (worker thread) ------------------------------------------
    def _append(self, event: dict[str, Any]) -> None:
        with self._lock:
            event.setdefault("seq", len(self.events))
            event.setdefault("ts", now_iso())
            self.events.append(event)


class _RunSink:
    """Adapts the graph's :class:`EventSink` (node_start/node_end) into run events."""

    def __init__(self, run: Run) -> None:
        self._run = run

    def emit(self, event: GraphEvent) -> None:
        self._run._append({"type": event.kind, "node": event.node, "data": event.data})


class RunManager:
    """Owns all runs and the worker threads driving them."""

    def __init__(
        self,
        settings: Settings,
        *,
        sandbox: Sandbox | None = None,
        retriever: Retriever | None = None,
        episodic: EpisodicMemory | None = None,
        long_term: LongTermMemory | None = None,
        provider_overrides: dict[str, LLMProvider] | None = None,
    ) -> None:
        self._settings = settings
        self._sandbox = sandbox or get_sandbox(settings.sandbox)
        #: Opt-in RAG/memory (Phase 3), wired so `retrieve` grounds instead of erroring.
        self._retriever = retriever
        self._episodic = episodic
        self._long_term = long_term
        #: Test seam: inject scripted planner/coder/reviewer providers (keys match build_graph).
        self._provider_overrides = provider_overrides or {}
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        # Serialize graph execution: one local Ollama serves a single model, so running
        # multiple graphs at once only thrashes it. Runs queue and execute one at a time
        # (the slot is held for a run's whole life, human pauses included).
        self._run_slot = threading.BoundedSemaphore(1)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r.snapshot() for r in self._runs.values()]

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def start(self, request: str, autonomy: str = "auto") -> Run:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        workspace = Path(tempfile.mkdtemp(prefix=f"mc-{run_id}-"))
        git = Git(workspace)
        git.init()
        (workspace / ".gitkeep").write_text("", encoding="utf-8")
        (workspace / ".gitignore").write_text(_WORKSPACE_GITIGNORE, encoding="utf-8")
        base = git.commit("mission-control: new run workspace")

        run = Run(run_id=run_id, request=request, autonomy=autonomy, workspace_path=str(workspace))
        with self._lock:
            self._runs[run_id] = run

        thread = threading.Thread(
            target=self._drive, args=(run, base, git.current_branch()), daemon=True
        )
        thread.start()
        return run

    def respond(self, run_id: str, decision: dict[str, Any] | str) -> bool:
        """Answer a pending HITL interrupt. Returns False if the run isn't awaiting one."""
        run = self.get(run_id)
        if run is None or run.status != "waiting_human":
            return False
        run._resume_q.put(decision)
        return True

    # -- worker ------------------------------------------------------------
    def _drive(self, run: Run, base_commit: str, work_branch: str) -> None:
        """Thread entry point: queue behind any active run, then drive the graph.

        The single concurrency slot (see ``__init__``) is what serializes runs so
        they can't oversubscribe the one local Ollama model. All failures are
        caught here so a run always reaches a terminal state and ``_finished`` is
        always set (SSE tailers never hang).
        """
        try:
            if not self._run_slot.acquire(blocking=False):
                run.status = "queued"
                run._append({"type": "queued"})
                self._run_slot.acquire()  # block until the active run frees the slot
            run.status = "running"
            try:
                self._run_graph(run, base_commit, work_branch)
            finally:
                self._run_slot.release()
        except Exception as exc:  # a driver/model failure ends the run cleanly, never hangs
            log.warning("run_failed", run_id=run.run_id, error=str(exc))
            run.status = "error"
            run._append({"type": "error", "error": str(exc)})
        finally:
            run._finished.set()

    def _run_graph(self, run: Run, base_commit: str, work_branch: str) -> None:
        """Build the graph (RAG/memory wired) and stream it, handling HITL pauses."""
        graph = build_graph(
            self._settings,
            sandbox=self._sandbox,
            checkpointer=InMemorySaver(),
            sink=_RunSink(run),
            retriever=self._retriever,
            episodic=self._episodic,
            long_term=self._long_term,
            planner_provider=self._provider_overrides.get("planner"),
            coder_provider=self._provider_overrides.get("coder"),
            reviewer_provider=self._provider_overrides.get("reviewer"),
        )
        state = dict(new_run_state(
            run_id=run.run_id, project_id=run.run_id, user_request=run.request,
            workspace_path=run.workspace_path, autonomy_level=run.autonomy,  # type: ignore[arg-type]
            max_tokens=None, max_steps=60, max_wall_clock_s=3600, started_at=now_iso(),
        ))
        state["base_commit"] = base_commit
        state["work_branch"] = work_branch
        config: dict[str, Any] = {"configurable": {"thread_id": run.run_id}}

        payload: Any = state
        for _ in range(80):  # generous; graph's own budgets bound real work
            interrupted = False
            for chunk in graph.stream(  # type: ignore[call-overload]
                payload, config=config, stream_mode="updates"
            ):
                if "__interrupt__" in chunk:
                    request_value = chunk["__interrupt__"][0].value
                    run.pending_interrupt = request_value
                    run.status = "waiting_human"
                    run._append({"type": "interrupt", "request": request_value})
                    interrupted = True
                    break
            if not interrupted:
                break
            decision = run._resume_q.get()  # blocks until POST /respond
            run.pending_interrupt = None
            run.status = "running"
            run._append({"type": "resumed", "decision": decision})
            payload = Command(resume=decision)

        final = graph.get_state(config).values  # type: ignore[arg-type]
        run.final_state = _snapshot_state(final)
        run.status = "done"
        run._append({"type": "done", "status": final.get("status"), "state": run.final_state})


def _model_dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def _snapshot_state(state: dict[str, Any]) -> dict[str, Any]:
    """Compact, JSON-safe projection of the final graph state for the UI."""
    plan = state.get("plan")
    review = state.get("review")
    verify = state.get("verify_result")
    return {
        "status": state.get("status"),
        "plan": _model_dump(plan) if plan is not None else None,
        "review": _model_dump(review) if review is not None else None,
        "verify_result": _model_dump(verify) if verify is not None else None,
        "diff_summary": state.get("diff_summary", ""),
        "changed_files": [_model_dump(f) for f in state.get("changed_files", [])],
        "node_history": state.get("node_history", []),
    }
