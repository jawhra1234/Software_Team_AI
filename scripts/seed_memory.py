"""Seed long-term memory from the repo's ADRs (Task 3.13).

The deterministic long-term-memory writer, runnable as a CLI. Ingests every ADR
under ``docs/adr/`` into ``memory_semantic`` as ``decision`` memories, so the
planner's "Project Conventions" context has real content. Idempotent — re-run
after editing ADRs. Requires Postgres + pgvector.

    python scripts/seed_memory.py <project_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.memory.ingest import ingest_adrs  # noqa: E402
from app.rag.factory import build_rag_stack  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    project_id = argv[0]

    configure_logging(get_settings())
    stack = build_rag_stack(get_settings())
    stack.long_term.ensure_schema()

    adr_dir = _ROOT / "docs" / "adr"
    count = ingest_adrs(stack.long_term, project_id, adr_dir)
    print(f"wrote {count} decision memories for project '{project_id}' from {adr_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
