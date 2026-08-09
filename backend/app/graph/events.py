"""Graph event hooks for streaming (Task 2.14).

Nodes emit structured events through an :class:`EventSink`; the API layer
(later phases) adapts a sink to SSE/WebSocket. :class:`NullEventSink` is the
default so the graph works without a listener; :class:`ListEventSink` is used
in tests and for local debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

EventKind = str  # "node_start" | "node_end" | "tool_call" | "verify_result" | "interrupt"


@dataclass
class GraphEvent:
    kind: EventKind
    node: str
    data: dict[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    def emit(self, event: GraphEvent) -> None: ...


class NullEventSink:
    def emit(self, event: GraphEvent) -> None:
        return None


class ListEventSink:
    """Accumulates events in memory (tests, local debugging)."""

    def __init__(self) -> None:
        self.events: list[GraphEvent] = []

    def emit(self, event: GraphEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Tool-activity bridge (Phase 6). Lets the *inside* of a node — where individual
# tool calls happen, opaque to node-level instrumentation — surface "tool"
# events for a live UI. Purely additive: a ContextVar defaulting to ``None``, so
# with no sink bound (every non-API caller, and all tests) it is a no-op and the
# pipeline behaves exactly as before. ContextVars are per-thread, so concurrent
# API runs (each on its own worker thread) never cross streams.
# ---------------------------------------------------------------------------
import contextlib  # noqa: E402
import contextvars  # noqa: E402  (kept next to what it powers)

_active_sink: contextvars.ContextVar[tuple[EventSink, str] | None] = contextvars.ContextVar(
    "active_sink", default=None
)


def bind_sink(sink: EventSink, node: str) -> contextvars.Token[tuple[EventSink, str] | None]:
    """Bind ``sink`` (tagged with the current node name) for the duration of a node."""
    return _active_sink.set((sink, node))


def unbind_sink(token: contextvars.Token[tuple[EventSink, str] | None]) -> None:
    _active_sink.reset(token)


def emit_tool_event(tool: str, ok: bool) -> None:
    """Best-effort ``tool`` event to the bound sink, if any (never raises)."""
    binding = _active_sink.get()
    if binding is None:
        return
    sink, node = binding
    with contextlib.suppress(Exception):
        sink.emit(GraphEvent(kind="tool", node=node, data={"tool": tool, "ok": ok}))
