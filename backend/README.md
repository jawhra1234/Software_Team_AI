# Backend — AI Software Engineering Workspace

Python service implementing the supervised coding-agent orchestrator. See the
architecture and phased build plans under [`../docs/`](../docs/):

- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — normative system spec
- [`docs/adr/`](../docs/adr/) — architecture decision records
- [`docs/build-plans/`](../docs/build-plans/) — phased implementation plans

## Layout (`ARCHITECTURE.md §2.6`)

```
app/
├─ api/        # FastAPI routers (later phases)
├─ graph/      # LangGraph: state, nodes, edges (Phase 2)
├─ agents/     # planner, coder, reviewer (Phase 2+)
├─ tools/      # fs, search, git, shell/sandbox (Phase 1)
├─ providers/  # LLM provider abstraction (Phase 0)
├─ memory/     # short/long-term memory (Phase 3)
├─ rag/        # chunking, index, retrieval (Phase 3)
├─ core/       # config, logging, tracing, errors (Phase 0)
└─ db/         # models, migrations (later phases)
```

## Development

```bash
# from backend/
uv venv
uv pip install -e ".[dev]"

uv run ruff check .
uv run mypy
uv run pytest
```

Package is PEP 561 typed (`app/py.typed`).
