"""Deterministic long-term-memory writer: ingest ADRs (Task 3.13).

Long-term memory (``memory_semantic``) needs a writer, otherwise
:meth:`LongTermMemory.search` returns nothing in real runs and the planner's
"Project Conventions" context is permanently empty. The project already keeps
its durable decisions as Architecture Decision Records under ``docs/adr/`` — so
this ingests each ADR's title + ``## Decision`` section as one ``decision``
memory, idempotently.

This is the *deterministic* writer. Automatic distillation of durable facts
from run history (an LLM summarising each run at ``finalize``) is deferred to a
later phase — it needs dedup and quality gating a local model can't be trusted
to provide.
"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger
from app.memory.long_term import LongTermMemory, MemoryKind

log = get_logger("memory.ingest")

#: Per-ADR text cap kept generous but bounded, so one long ADR can't dominate
#: the embedding / retrieved context.
_MAX_ADR_CHARS = 1500


def _extract_section(body: str, heading: str) -> str | None:
    """Return the text under a ``## <heading>`` markdown section, if present."""
    lines = body.splitlines()
    target = f"## {heading}".lower()
    out: list[str] = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("## "):
            if capturing:  # next section starts -> stop
                break
            capturing = stripped.lower() == target
            continue
        if capturing:
            out.append(line)
    text = "\n".join(out).strip()
    return text or None


def parse_adr(text: str) -> str | None:
    """Distil one ADR into a concise durable fact: title + its Decision.

    Falls back to the whole document (truncated) when there's no ``## Decision``
    section. Returns ``None`` for a file with no title line (not an ADR).
    """
    title = next(
        (ln.strip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")),
        None,
    )
    if not title:
        return None
    decision = _extract_section(text, "Decision")
    fact = f"{title}\n\nDecision: {decision}" if decision else f"{title}\n\n{text.strip()}"
    return fact[:_MAX_ADR_CHARS].strip()


def ingest_adrs(long_term: LongTermMemory, project_id: str, adr_dir: Path) -> int:
    """Write every ADR under ``adr_dir`` into long-term memory as ``decision``s.

    Idempotent: clears the project's existing ``decision`` memories first, so
    re-running after editing/adding ADRs replaces rather than duplicates them.
    ``README.md`` (the ADR index) is skipped. Returns the number written.
    """
    facts: list[tuple[MemoryKind, str]] = []
    for path in sorted(adr_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        fact = parse_adr(path.read_text(encoding="utf-8"))
        if fact:
            facts.append(("decision", fact))

    long_term.clear_project(project_id, kind="decision")
    written = long_term.write_many(project_id, facts)
    log.info("adrs_ingested", project_id=project_id, count=written, source=str(adr_dir))
    return written
