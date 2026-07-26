# Phase 0 — Foundations

**Navigation:** [← Documentation Hub](../README.md) · Previous: — · [Next: Phase 1 — Core coder loop →](phase-1-coder-loop.md)

> *This is an **as-built** writeup — what actually shipped and how it was verified. For the original forward-looking specification, see the [Phase 0 build plan](../build-plans/PHASE-0.md).*

The substrate every later phase sits on: one swappable, schema-safe way to call an LLM, plus
config, logging/tracing, and local infra.

## What this phase adds

A provider-agnostic LLM layer (`chat` / `stream` / `embed`), a `structured_call` utility that
coerces unreliable local models into schema-valid output, env-driven config, structured logging
with optional Langfuse tracing, and the local infra (Postgres + pgvector, Ollama) to run it all.

## Why it was needed

Every node in every later phase calls an LLM. Settling the call interface, structured-output
handling, and config **before** the graph exists means Phase 2 debugs *orchestration* only, not
orchestration + a flaky LLM call at the same time. It's also the seam that makes "swap the
model/provider by config" true for the whole project.

## Architecture / how it works

- **`LLMProvider`** is an interface; concrete adapters (Ollama today; hosted providers later)
  live behind it and are chosen by config. A **factory/registry** resolves a provider *per
  role* (`planner` / `coder` / `reviewer` / `embed`), so each role can point at a different
  model without code changes ([ADR-0003](../adr/0003-provider-abstraction.md)).
- **`structured_call`** picks a strategy from the model's declared capabilities: a forced "emit"
  tool call for tool-capable models, else JSON-mode prompting. Either way the payload is
  validated against a Pydantic schema, and on failure a repair message carrying the validation
  error is appended and the call retried. It also unwraps the tool-call envelope some local
  models emit as plain text. This one utility is why an unreliable 7B produces valid `Plan` /
  `Review` artifacts everywhere else.

## Implementation

- `app/core/config.py` — `pydantic-settings`; nested env via the `__` delimiter.
- `app/providers/base.py` — the `LLMProvider` interface + value objects.
- `app/providers/ollama.py` — the Ollama adapter (with a small transport retry/backoff).
- `app/providers/factory.py` — provider registry + per-role resolution.
- `app/providers/structured.py` — `structured_call` (schema validation + repair-retry).
- `app/core/logging.py`, `app/core/tracing.py` — structlog JSON logs; graceful-no-op Langfuse.
- `infra/` — `docker-compose.yml` (Postgres + pgvector, Langfuse, opt-in Ollama), init scripts.
- `scripts/bootstrap.*`, `scripts/smoke_llm.py`.

## Configuration

```bash
MODELS__CODER__MODEL=qwen2.5-coder:7b-instruct   # per-role model; also PLANNER / REVIEWER / EMBED
MODELS__REVIEWER__PROVIDER=ollama                # per-role provider override (else Settings.provider)
OLLAMA__BASE_URL=http://localhost:11434
OLLAMA__REQUEST_TIMEOUT_S=600                     # generous for local CPU
LANGFUSE__ENABLED=false                           # tracing off by default (no-op)
```

## Testing and validation

- **Hermetic:** provider adapter behavior and `structured_call`'s repair-retry / envelope
  unwrapping are unit-tested with a fake provider (`tests/test_structured.py`, `test_smoke.py`).

## Live validation

`scripts/smoke_llm.py` makes a real structured call against live Ollama and validates the result
against its schema — proving the provider + `structured_call` work before anything builds on them.

## What worked

`structured_call` is the quiet workhorse of the whole project: its repair-retry + envelope
unwrapping is what lets a mediocre local model reliably emit valid structured artifacts in every
later phase.

## Known limitations / honest findings

- Nothing phase-specific — this layer is solid. The only external dependency is a working Ollama
  install; a broken/partial Ollama (missing inference binary) blocks *all* live calls machine-wide
  (an environment issue, surfaced during Phase 5 validation, not a code defect).

## Key engineering decisions

- [ADR-0003 — Provider abstraction](../adr/0003-provider-abstraction.md)
- [ADR-0004 — Single primary local model (`qwen2.5-coder:7b` + `nomic-embed-text`)](../adr/0004-ollama-model-choice.md)

## Current status

✅ **Complete and verified.**

---

**Navigation:** [← Documentation Hub](../README.md) · Previous: — · [Next: Phase 1 — Core coder loop →](phase-1-coder-loop.md)
