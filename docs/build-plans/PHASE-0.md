# Phase 0 — Foundations

> **Goal:** Lay the substrate everything else sits on — config, provider abstraction, one validated LLM call, structured-output reliability, logging/tracing, and local infra — so that no later phase has to rework the base.
>
> **Why this phase comes first:** The provider abstraction and `structured_call` utility are touched by every node. Getting them right before the graph exists prevents a rewrite cascade later. This phase deliberately contains **no graph and no agents** — it de-risks the primitives.

## Objectives
1. Reproducible local environment (Postgres+pgvector, Ollama) via docker-compose.
2. A config-driven `LLMProvider` abstraction with an Ollama adapter and per-role config.
3. A `structured_call()` utility with Pydantic validation + repair-retry.
4. Structured logging + Langfuse tracing wired end to end.
5. Proof that a provider/model swap is a config change only.

## Scope
**In:** config loader, provider interface + Ollama adapter + embed, `structured_call`, logging/tracing, docker-compose, health checks, model pull scripts.
**Out (later phases):** graph, nodes, tools, workspace, RAG, UI, HITL. No `run_command`, no git integration yet.

## Prerequisites
- Ollama installed; models pulled: `qwen2.5-coder:7b-instruct` (Q4_K_M) and `nomic-embed-text`.
- Docker Desktop available for Postgres+pgvector.
- Python 3.11+ toolchain (`uv` or `poetry`).

## Work breakdown & deliverables
| # | Task | Deliverable |
|---|---|---|
| 0.1 | Repo skeleton per ARCHITECTURE §2.6 (backend/app/{core,providers}, tests, infra, docs) | Importable package, lint/format/typecheck configured (ruff + mypy) |
| 0.2 | `core/config.py` — env/file-driven settings; per-role model config block | `Settings` object; `.env.example` |
| 0.3 | `providers/base.py` — `LLMProvider` interface (`chat/structured/stream/embed/capabilities`) | Typed protocol + `ChatResponse`/`Chunk`/`Vector` types |
| 0.4 | `providers/ollama.py` — adapter (sets `num_ctx`, `keep_alive`, temp per role; declares `supports_tools`) | Working chat + embed against local Ollama |
| 0.5 | `providers/factory.py` — `get(role)` resolution from config | Role→model resolution; single-model-locally verified |
| 0.6 | `providers/structured.py` — `structured_call(messages, schema)` with validate + repair-retry ×2 | Returns validated Pydantic; raises after retries |
| 0.7 | `core/logging.py` + Langfuse init — run/trace id context var threaded into logs and LLM calls | JSON logs; a trace visible in self-hosted Langfuse |
| 0.8 | `infra/docker-compose.yml` — Postgres+pgvector, Langfuse (+ optional Ollama service) | `docker compose up` yields healthy services |
| 0.9 | Model pull + smoke script | `scripts/bootstrap.*` pulls models, checks health |

## Testing strategy
- **Golden LLM test:** a fixed prompt returns a schema-valid object via `structured_call` (mark as integration; skippable in CI without Ollama).
- **Provider-swap test:** with a stub/second provider registered, switching `coder` role via config changes the resolved model with **zero code change**. This is the phase's key assertion.
- **Structured repair test:** feed a deliberately malformed model output (mocked) → `structured_call` triggers the repair path and ultimately validates or raises cleanly.
- **Capabilities fallback test:** a provider with `supports_tools=False` routes through JSON-mode prompting.
- **Infra health test:** compose services reachable; pgvector extension present.

## Definition of Done
- `docker compose up` + `bootstrap` → healthy stack.
- A one-liner script performs a validated structured LLM call and the trace appears in Langfuse.
- Swapping the model for any role is a config edit only (test proves it).
- Lint, type-check, and Phase-0 tests are green.

## Risks & mitigations
- **Ollama tool/JSON flakiness** → owned by `structured_call` repair loop and `capabilities` fallback (built here, not deferred).
- **16 GB pressure** → set `num_ctx` conservatively; `keep_alive` to avoid reloads; embed model co-resides.
- **Langfuse self-host friction** → keep it optional behind a config flag; logging must work without it.
