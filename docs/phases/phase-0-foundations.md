# Phase 0 — Foundations

[← Back to README](../../README.md) · [Docs index](../README.md) · [Build plan](../build-plans/PHASE-0.md)

> The substrate every later phase sits on: one swappable, schema-safe way to call an LLM, plus config, logging/tracing, and local infra.

## What was built

- **Config** (`app/core/config.py`) — `pydantic-settings` with a per-role model block
  (planner / coder / reviewer / embed), sandbox, and coder budgets. Fully env-driven, so
  swapping a model or provider is a config change, not a code change.
- **Provider abstraction** (`app/providers/`) — an `LLMProvider` interface + an Ollama
  adapter (`chat` / `stream` / `embed`, tool-calling, `num_ctx` / `keep_alive`), a
  factory/registry, and **`structured_call`**: schema-validated output with repair-retry
  that also unwraps the tool-call envelope some local models emit as plain text. This one
  utility is what makes an unreliable 7B produce valid structured artifacts everywhere else.
- **Logging & tracing** (`app/core/`) — `structlog` JSON logs with a run/trace id threaded
  through context vars; self-hosted Langfuse wiring that degrades to a graceful no-op when
  disabled.
- **Infra** (`infra/`) — `docker-compose.yml` for Postgres + pgvector and Langfuse (plus an
  opt-in Ollama profile); Postgres init scripts.
- **Scripts** (`scripts/`) — `bootstrap.*` (pull models + health checks) and `smoke_llm.py`
  (a validated structured LLM call end to end).

## How it was verified

`scripts/smoke_llm.py` makes a real structured LLM call against live Ollama and validates
the result against its Pydantic schema — proving the provider adapter + `structured_call`
work before anything is built on them.

## Key decisions

- [ADR-0003 — Provider abstraction](../adr/0003-provider-abstraction.md)
- [ADR-0004 — Single primary local model (`qwen2.5-coder:7b` + `nomic-embed-text`)](../adr/0004-ollama-model-choice.md)

---

Next: [Phase 1 — Core coder loop →](phase-1-coder-loop.md)
