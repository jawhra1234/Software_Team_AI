# ADR-0003: Config-driven provider abstraction with per-role model config

**Status:** Accepted

## Context
Development uses Ollama; the system must later swap to OpenRouter/Gemini/Groq/OpenAI without touching application architecture. Ollama's structured-output and tool-calling support is model-dependent and flakier than hosted APIs.

## Decision
Define an `LLMProvider` interface (`chat`, `structured`, `stream`, `embed`, `capabilities`) with a factory keyed by **role** (`planner`, `coder`, `reviewer`, `embed`). Locally all roles resolve to one Ollama model; in cloud, roles can resolve to different models via config only. Domain code never imports provider SDKs directly and contains no `if provider == ...` branches. A `structured()` helper validates against a Pydantic schema and re-prompts with the validation error (×2) on failure.

## Consequences
- Provider/model swap = env/config change.
- Per-role upgrades in cloud (e.g. frontier planner) without graph changes.
- The `structured()` repair loop absorbs Ollama JSON/tool-calling flakiness — a major real-world time saver.
- `capabilities` flags let the graph fall back to JSON-mode prompting when native tool-calling is absent.

## Alternatives rejected
- **Raw LangChain models throughout:** leaks framework types into domain code; couples swap to code changes.
- **One hardcoded provider:** violates the explicit portability requirement.
