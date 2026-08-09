"""Mission-Control API (Phase 6) — FastAPI over the existing graph.

Thin HTTP/SSE surface around :class:`RunManager`. Endpoints:

* ``POST   /api/runs``               start a run  ``{request, autonomy}``
* ``GET    /api/runs``               list runs
* ``GET    /api/runs/{id}``          one run's snapshot (status, pending interrupt, final state)
* ``GET    /api/runs/{id}/events``   Server-Sent Events — the live, replayable event stream
* ``POST   /api/runs/{id}/respond``  answer a HITL interrupt ``{decision, edits?, note?}``
* ``GET    /api/health``             liveness

CORS is open (localhost dev). Nothing here touches the graph/agents; it only
starts runs and relays their events, so the core pipeline is undisturbed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.run_manager import RunManager
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

log = get_logger("api.server")


class StartRunRequest(BaseModel):
    request: str = Field(min_length=1, description="The coding task for the agent.")
    autonomy: str = Field(default="auto", pattern="^(auto|semi|manual)$")


class RespondRequest(BaseModel):
    decision: str = Field(min_length=1, description="e.g. approve | revise | abort | accept")
    edits: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


def create_app(manager: RunManager | None = None) -> FastAPI:
    """Build the app. Tests inject a ``RunManager`` wired with scripted providers."""
    settings = get_settings()
    app = FastAPI(title="AI SWE Mission Control", version="0.6.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )
    app.state.manager = manager or _default_manager(settings)

    def mgr() -> RunManager:
        return app.state.manager  # type: ignore[no-any-return]

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return mgr().list_runs()

    @app.post("/api/runs")
    def start_run(body: StartRunRequest) -> dict[str, Any]:
        run = mgr().start(body.request, body.autonomy)
        return run.snapshot()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = mgr().get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run.snapshot()

    @app.post("/api/runs/{run_id}/respond")
    def respond(run_id: str, body: RespondRequest) -> dict[str, Any]:
        run = mgr().get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        decision: dict[str, Any] = {
            "decision": body.decision, "edits": body.edits, "note": body.note,
        }
        if not mgr().respond(run_id, decision):
            raise HTTPException(status_code=409, detail="run is not awaiting a human response")
        return {"ok": True}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str) -> StreamingResponse:
        run = mgr().get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")

        async def stream() -> AsyncIterator[bytes]:
            yield b": connected\n\n"  # open + flush the stream immediately
            index = 0
            while True:
                new, finished = run.events_from(index)
                index += len(new)
                for event in new:
                    yield f"data: {json.dumps(event)}\n\n".encode()
                if finished and not new:
                    yield b"event: end\ndata: {}\n\n"
                    return
                await asyncio.sleep(0.15)

        # Headers that stop intermediaries (and Next's dev proxy) from buffering SSE.
        headers = {
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    return app


def _default_manager(settings: Settings) -> RunManager:
    """Wire the UI to run the graph in the SAME proven configuration the validation
    scripts and eval harness use — so UI runs are as reproducible/capable as those.

    - Pin the `coder` temperature to 0.0 — and ONLY the coder — to exactly match the
      proven eval config (`run_evals.py` pins coder + reviewer; reviewer already
      defaults to 0.0). The coder default is 0.2 (sampling), which is why the
      un-pinned UI sometimes wrote an empty test. The **planner is deliberately left
      at its 0.1 default**: pinning it to 0.0 makes decoding fully greedy, which
      locked in a most-likely-but-wrong plan (files under a `workspace/` subdir that
      pytest, run from the root, can't find). The proven config's slight planner
      sampling avoids that — so we match it rather than over-pin.
    - Wire the RAG stack best-effort so `retrieve` grounds instead of erroring on
      every call — falling back to no-RAG if Postgres is unreachable.
    - Local-dev friendly: subprocess sandbox (no Docker) + smaller context for 16 GB.
    """
    settings.sandbox.backend = "subprocess"
    settings.ollama.default_num_ctx = min(settings.ollama.default_num_ctx, 4096)
    settings.models.coder.temperature = 0.0

    retriever = episodic = long_term = None
    try:
        from app.rag.factory import build_rag_stack

        rag = build_rag_stack(settings)
        rag.long_term.ensure_schema()
        rag.episodic.ensure_schema()
        retriever, episodic, long_term = rag.retriever, rag.episodic, rag.long_term
        log.info("api_rag_wired")
    except Exception as exc:  # Postgres unreachable → run without RAG (retrieve degrades)
        log.warning("api_rag_unavailable", error=str(exc))

    return RunManager(settings, retriever=retriever, episodic=episodic, long_term=long_term)


# Module-level app for `uvicorn app.api.server:app`.
configure_logging(get_settings())
app = create_app()
