"""Planner memory context: long-term + episodic → labeled block (Task 3.13).

Assembles the read-only memory the planner receives *automatically*, before it
grounds. Two clearly-labeled sections so a small model can tell them apart:

    === Project Conventions ===   durable decisions/norms (long-term, semantic)
    === Previous Attempts ===     relevant past runs (episodic, lexical-ranked)

Repository code is deliberately **not** here — it stays on-demand via the
`retrieve` tool, preserving the agentic loop and saving tokens. Every read is
best-effort: a memory backend being down degrades that section to nothing, never
fails planning. A section renders only when its source returns something, so an
empty store yields an empty block and the planner behaves exactly as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import PlannerSettings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.memory.episodic import EpisodicMemory
    from app.memory.long_term import LongTermMemory

log = get_logger("graph.planning_context")


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _cap_section(header: str, bullets: list[str], max_chars: int) -> str | None:
    """Join bullets under a header, dropping trailing bullets past the char cap."""
    if not bullets:
        return None
    kept: list[str] = []
    used = 0
    for bullet in bullets:
        if used + len(bullet) + 1 > max_chars and kept:
            break
        kept.append(bullet)
        used += len(bullet) + 1
    return f"=== {header} ===\n" + "\n".join(kept)


def _long_term_section(
    long_term: LongTermMemory | None, project_id: str, request: str, s: PlannerSettings
) -> str | None:
    if long_term is None:
        return None
    try:
        items = long_term.search(project_id, request, k=s.memory_long_term_k)
    except Exception as exc:  # best-effort: a memory outage must not fail planning
        log.warning("long_term_read_failed", error=str(exc))
        return None
    bullets = [f"- {_truncate(item.text, 240)}" for item in items]
    return _cap_section(
        "Project Conventions (learned decisions & repo norms)", bullets, s.memory_max_section_chars
    )


def _episodic_section(
    episodic: EpisodicMemory | None, project_id: str, request: str, s: PlannerSettings
) -> str | None:
    if episodic is None:
        return None
    try:
        runs = episodic.relevant(
            project_id,
            request,
            k=s.memory_episodic_k,
            candidate_window=s.memory_episodic_candidate_window,
        )
    except Exception as exc:  # best-effort
        log.warning("episodic_read_failed", error=str(exc))
        return None
    bullets = [
        f"- [{r.status}] {_truncate(r.summary or '(no summary)', 200)} "
        f"({r.tasks_done}/{r.tasks_total} tasks)"
        for r in runs
    ]
    return _cap_section(
        "Previous Attempts (past runs on this project)", bullets, s.memory_max_section_chars
    )


def build_planner_context(
    request: str,
    project_id: str,
    *,
    long_term: LongTermMemory | None,
    episodic: EpisodicMemory | None,
    settings: PlannerSettings,
) -> str:
    """Return the injected memory block, or ``""`` when nothing is available."""
    sections = [
        section
        for section in (
            _long_term_section(long_term, project_id, request, settings),
            _episodic_section(episodic, project_id, request, settings),
        )
        if section
    ]
    return "\n\n".join(sections)
