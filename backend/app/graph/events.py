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
