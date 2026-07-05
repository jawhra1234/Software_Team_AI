# ADR-0010: Postgres + pgvector + LangGraph checkpointer (SQLite dev default)

**Status:** Accepted

## Context
The system needs relational storage (projects, runs, events, artifacts), vector storage (code chunks + memory), and durable graph checkpoints. Minimizing moving parts on a 16 GB dev machine matters, but production realism is a resume goal.

## Decision
Use **Postgres with pgvector** for both relational and vector data — one database for both. Use the **LangGraph Postgres checkpointer** for production realism, with the **SQLite checkpointer as the clone-and-run local default**. Final diffs/summaries are filesystem artifacts locally, object storage in cloud.

## Consequences
- One system for relational + vector reduces operational surface.
- SQLite default means the repo runs with zero external services for a quick start.
- Postgres checkpointer externalizes state → horizontal-scale seam.
- Demonstrates Postgres + pgvector competency.

## Alternatives rejected
- **Dedicated vector DB (Qdrant/Chroma) + separate relational DB:** more services to run locally for no early benefit; Chroma retained only as a fallback.
- **In-memory checkpointer:** no durability/resume.
