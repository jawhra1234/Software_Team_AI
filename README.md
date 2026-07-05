# AI Software Engineering Workspace

A **supervised coding-agent workspace**: a LangGraph state machine that drives a small
set of capability-bounded agents through a `plan → code → verify → review` loop over a
**git-backed workspace**, grounded in the real repository, with sandboxed execution,
human-in-the-loop gates, and full observability.

It is **not** a chatbot, and **not** a one-prompt app generator. Agents are defined by
tool boundaries, context isolation, and verification — not by human job titles. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the decision records in
[`docs/adr/`](docs/adr/) for the full design and rationale.

> **Status:** Phase 0 (Foundations) and Phase 1 (Core coder loop) are complete and
> verified. Phases 2–7 are specced in [`docs/build-plans/`](docs/build-plans/) and not
> yet implemented.

---

## Why this design

The target architecture is an orchestrator over **3 real LLM roles** (planner, coder,
reviewer) + a **deterministic verify** node — not a role-play "team" of seven agents.
Three deliberately separated sources of truth keep it honest and cheap on a 16 GB laptop:

| Concern | Lives in |
|---|---|
| Control flow + small structured artifacts | LangGraph state (checkpointed) |
| The actual code | Git-backed workspace on disk |
| Retrievable knowledge (code chunks, memory) | Vector store + Postgres |

The full graph is `plan · human_gate · coder · verify · review · finalize` (arriving in
Phase 2). Everything runs locally on **one primary Ollama model** (`qwen2.5-coder:7b`),
with a config-only path to swap in OpenRouter / Gemini / Groq / OpenAI later.

---

## What's implemented (Phases 0–1)

### Phase 0 — Foundations
- **Config** (`app/core/config.py`): `pydantic-settings` with a per-role model block
  (planner/coder/reviewer/embed), sandbox, and coder budgets. Env-driven, provider-swap
  by config alone.
- **Provider abstraction** (`app/providers/`): `LLMProvider` interface + Ollama adapter
  (`chat`/`stream`/`embed`, tool-calling, `num_ctx`/`keep_alive`), a factory/registry,
  and `structured_call` — schema-validated output with repair-retry that also unwraps
  tool-call envelopes some local models emit as text.
- **Logging & tracing** (`app/core/`): `structlog` JSON logs with a run/trace id threaded
  through context vars; self-hosted Langfuse wiring (graceful no-op when disabled).
- **Infra** (`infra/`): `docker-compose.yml` for Postgres + pgvector and Langfuse (+ an
  opt-in Ollama profile); Postgres init scripts.
- **Scripts** (`scripts/`): `bootstrap.*` (pull models + health checks) and `smoke_llm.py`
  (a validated structured LLM call).

### Phase 1 — Core coder loop
- **Tool layer** (`app/tools/`): a typed `Tool` protocol, `ToolResult`/`ToolContext`,
  registry, and a `to_langchain_tool` adapter. A central **authorization pipeline**
  (`execute_tool`): schema-validate → path-jail → command allow/deny → approval hook →
  execute → truncate → trace.
- **Tools**: `read_file` / `write_file` / `edit_file` / `list_dir`, `search_code`
  (ripgrep with a pure-Python fallback), `run_command`, git (`status`/`diff`/`add`/`commit`),
  and `finish_task`.
- **Sandbox** (`app/tools/sandbox.py`, `infra/sandbox/Dockerfile`): `DockerSandbox`
  (`--network=none`, mem/CPU/pids limits, timeouts — the security boundary per ADR-0007)
  and a config-flagged `SubprocessSandbox` fallback.
- **Workspace lifecycle** (`app/workspace/`): git-backed project dirs, `base_commit`,
  `agent/run-<id>` branches, commit-per-task, cumulative diff.
- **Coder ReAct loop** (`app/agents/coder.py`): grounds via tools, edits files, runs
  commands in the sandbox, and self-corrects until `finish_task` or a budget/loop guard
  trips.
- **Deterministic verify** (`app/verify/runner.py`): auto-detects and runs checks
  (`compileall`, `pytest`, …) in the sandbox; a timeout counts as a failure.
- **Budgets** (`app/agents/budget.py`): step / wall-clock / token caps + no-progress
  detection.

**Verified end to end:** the coder builds a running, test-passing Python project entirely
inside the Docker sandbox (live `qwen2.5-coder:7b`, converged in 4 steps), confirmed by
the independent verify runner.

---

## Repository layout

```
.
├─ backend/                # Python service
│  ├─ app/
│  │  ├─ core/             # config, logging, tracing, errors
│  │  ├─ providers/        # LLM provider abstraction + Ollama adapter
│  │  ├─ tools/            # tool protocol, sandbox, fs/search/git/shell, authorization
│  │  ├─ agents/           # coder ReAct loop, budgets
│  │  ├─ verify/           # deterministic verify runner
│  │  ├─ workspace/        # git-backed workspace lifecycle
│  │  ├─ graph/ memory/ rag/ db/ api/   # placeholders for Phases 2–3+
│  │  └─ ...
│  └─ tests/               # hermetic + integration tests
├─ infra/                  # docker-compose, Postgres init, sandbox image
├─ scripts/                # bootstrap + smoke scripts
├─ docs/                   # ARCHITECTURE.md, ADRs, phased build plans
└─ workspaces/             # runtime project sandboxes (git-ignored)
```

---

## Quickstart

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv), Ollama, and
Docker (for the sandbox). ripgrep is optional (bundled in the sandbox image; a Python
fallback is used otherwise).

```bash
# 1. Pull the local models (ADR-0004)
ollama pull qwen2.5-coder:7b-instruct
ollama pull nomic-embed-text

# 2. Build the sandbox image (used by run_command and verify)
docker build -t aiswe-sandbox:latest infra/sandbox

# 3. Install the backend
cd backend
uv venv
uv pip install -e ".[dev]"

# 4. Verify
uv run ruff check .          # lint
uv run mypy                  # type-check (strict)
uv run pytest                # hermetic tests
uv run pytest -m integration # live tests (needs Ollama + Docker), incl. the e2e (~3 min)

# 5. Smoke: a validated structured LLM call
uv run python ../scripts/smoke_llm.py
```

Optional local infra (Postgres + pgvector + Langfuse):

```bash
docker compose -f infra/docker-compose.yml up -d
```

---

## Configuration

Settings are env-driven (see [`backend/.env.example`](backend/.env.example)); nested keys
use the `__` delimiter. Swapping a model or provider is config-only:

```bash
MODELS__CODER__MODEL=llama3.1:8b     # change the coder model
SANDBOX__BACKEND=subprocess          # fallback when Docker is unavailable
LANGFUSE__ENABLED=true               # turn on tracing
```

---

## Testing

- **Hermetic** (`uv run pytest`) — no external services; runs by default.
- **Integration** (`uv run pytest -m integration`) — requires live Ollama and/or Docker;
  opt-in so the default run stays fast and deterministic.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations (config, providers, structured output, logging/tracing, infra) | ✅ Complete |
| 1 | Core coder loop (tools, sandbox, git workspace, verify, budgets) | ✅ Complete |
| 2 | LangGraph orchestration + HITL (`plan→…→finalize`, checkpointer, interrupts) | 📋 Specced |
| 3 | Repository indexing, hybrid RAG, memory | 📋 Specced |
| 4 | Adversarial reviewer + iteration | 📋 Planned |
| 5 | Eval harness | 📋 Planned |
| 6 | Mission-control UI | 📋 Planned |
| 7 | Cloud provider swap + scale | 📋 Planned |

Full plans: [`docs/build-plans/`](docs/build-plans/).

---

## License

MIT.
