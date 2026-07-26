# AI Software Engineering Workspace

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests: 246 hermetic + 27 integration](https://img.shields.io/badge/tests-246%20hermetic%20%2B%2027%20integration-brightgreen)
![Typing: mypy --strict](https://img.shields.io/badge/mypy-strict-blue)
![Lint: ruff](https://img.shields.io/badge/lint-ruff-black)
![Runs locally on Ollama](https://img.shields.io/badge/runs-locally%20on%20Ollama-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

*(Badges are static, self-reported facts from this repo — there is no CI pipeline yet; the
numbers are reproducible via `uv run pytest` / `ruff` / `mypy`.)*

A **supervised coding-agent workspace**: a LangGraph state machine that drives a small set of
capability-bounded agents through a `plan → code → verify → review` loop over a **git-backed
workspace**, grounded in the real repository, with sandboxed execution, human-in-the-loop
gates, and full observability.

It is **not** a chatbot, and **not** a one-prompt app generator. Agents are defined by tool
boundaries, context isolation, and verification — not by human job titles.

> **Status:** Phases 0–5 are complete and verified — Foundations · Core coder loop ·
> Orchestration + HITL · RAG + memory · Review + self-correction · Eval harness. Phases 6–7
> are specced in [`docs/build-plans/`](docs/build-plans/) and not yet implemented.

📖 **Full documentation → [`docs/`](docs/README.md)** · deep design, the [runtime flow](docs/runtime-flow.md),
and an as-built writeup of every phase.

---

## At a glance

You give it a coding request. It **plans** the work grounded in your real repo, **writes** the
code one task at a time in a sandbox, **verifies** by running your actual tests, has a separate
**reviewer** critique the diff with fresh eyes, and **finalizes** — pausing for a human at
exactly the points you choose. It learns across runs (memory) and is scored against a task suite.

| Phase | In plain English | Deep dive |
|---|---|---|
| **0 · Foundations** | one swappable, schema-safe way to call any LLM, with logging/tracing | [→](docs/phases/phase-0-foundations.md) |
| **1 · Coder loop** | an agent that edits files, runs commands in a sandbox, and checks its own work with real tests | [→](docs/phases/phase-1-coder-loop.md) |
| **2 · Orchestration** | a LangGraph state machine wiring the agents together, with human-in-the-loop gates and crash-safe checkpoints | [→](docs/phases/phase-2-orchestration.md) |
| **3 · Grounding** | it reads your *actual* code (hybrid RAG) and remembers decisions + past runs, instead of guessing | [→](docs/phases/phase-3-rag-and-memory.md) |
| **4 · Review** | an independent reviewer catches problems and sends targeted fixes back — a real self-correction loop | [→](docs/phases/phase-4-review.md) |
| **5 · Evals** | the whole thing is scored against a fixed task suite, so any change can be proven better or worse | [→](docs/phases/phase-5-evals.md) |

It runs **entirely locally** on one small model (`qwen2.5-coder:7b` + `nomic-embed-text`) — and
swapping in a stronger/hosted model is a **config change, no code** (see [Configuration](#configuration)).

### How the phases connect

Each phase ships a working foundation the next one builds on — nothing is thrown away, and the
graph's *shape* never changes after Phase 2 (later phases upgrade seams, not topology).
`✅ shipped & verified · 📋 specced`

```
 ✅ 0 · FOUNDATIONS ───── one swappable, schema-safe way to call any LLM + logging/tracing
        │                 └▶ the substrate every agent call sits on
        ▼
 ✅ 1 · CODER LOOP ────── tools + sandbox + git workspace + a coder that verifies its own work
        │                 └▶ a *proven* single agent, ready to wrap in a graph
        ▼
 ✅ 2 · ORCHESTRATION ── LangGraph: plan→gate→coder→verify→review→finalize · HITL · checkpoints
        │                 └▶ the runnable pipeline (grounded on ripgrep, for now)
        ▼
 ✅ 3 · GROUNDING ────── hybrid RAG over your *real* code + long-term & episodic memory
        │                 └▶ upgrades the plan/coder grounding seam — graph untouched
        ▼
 ✅ 4 · REVIEW ───────── fresh-context reviewer + bounded, targeted self-correction loop
        │                 └▶ closes the quality loop — an independent second opinion
        ▼
 ✅ 5 · EVALS ────────── scored task suite + deterministic regression gate + recorded baseline
        │                 └▶ makes quality measurable: "better or worse", not just "different"
        ▼
 📋 6 · MISSION-CONTROL UI ─ Next.js: live graph, streaming, diff viewer, HITL cards
        │                 └▶ a UI built on a measured, trustworthy core
        ▼
 📋 7 · CLOUD + SCALE ── hosted-model swap, task queue, auth, horizontal-ready
                          └▶ run it for more than one person, on more than one machine
```

Read the order rationale in the **[roadmap](docs/build-plans/ROADMAP.md)**; the as-built story
of each shipped phase is in the **[phase writeups](docs/phases/)**.

---

## How it works

One run flows through six nodes; any node can pause for a human:

```
  user request  →  PLAN  →  HUMAN_GATE  →  CODER  ⇄  VERIFY  →  REVIEW  →  FINALIZE
```

- **PLAN** and **CODER** ground in your *real code* (hybrid RAG via `retrieve`) + **memory**
  before acting — they don't guess.
- **CODER ⇄ VERIFY**: on a failed test — or a `blocker`/`major` review finding — the coder
  re-enters in targeted **"fix mode"**, then re-verifies. **REVIEW** is a fresh-context
  reviewer that sees only the diff, never the coder's reasoning.
- **Any node can escalate to HUMAN_GATE** (budget exhausted / retries used / command needs
  approval); **FINALIZE** records the run's outcome to episodic memory for next time.

**See the full annotated diagram → [`docs/runtime-flow.md`](docs/runtime-flow.md).**

Three deliberately separated sources of truth keep it honest and cheap on a 16 GB laptop:

| Concern | Lives in |
|---|---|
| Control flow + small structured artifacts | LangGraph state (checkpointed) |
| The actual code | Git-backed workspace on disk |
| Retrievable knowledge (code chunks, memory) | Vector store + Postgres |

The architecture is an orchestrator over **3 real LLM roles** (planner, coder, reviewer) + a
**deterministic verify** node — not a role-play "team" of seven agents. Full rationale in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the [decision records](docs/adr/).

---

## Validation & known limitations

Everything below is verified in the current repository — nothing aspirational.

**Quality gates (every phase):**

| Gate | Status |
|---|---|
| `ruff check` (lint) | ✅ clean |
| `mypy` (strict) | ✅ clean (121 source files) |
| Hermetic tests (`pytest`) | ✅ **246 passing** — no external services |
| Integration tests (`pytest -m integration`) | ✅ **27 passing** — live Ollama / Docker / Postgres |
| Phase 5 eval baseline | ✅ recorded (`backend/evals/baseline.json`) |

**What the pipeline has *proven* works** (shown end-to-end through the real graph, not just unit tests):

- RAG demonstrably changes behavior — the coder reuses a hidden helper only with retrieval on.
- Memory carries across runs — a prior run's outcome measurably shapes the next run's plan.
- The self-correction loop is mechanically correct — severity-gated, isolated review; targeted
  fixes; bounded cycles that escalate to a human instead of looping.
- The full green happy path (`plan → coder → verify PASS → review approve → finalize succeeded`)
  runs clean on tasks within the model's reach.

**Known limitations — these come from the local `qwen2.5-coder:7b-instruct` model, *not* a broken pipeline:**

- The live 7B reviewer **does not reliably catch subtle defects unprompted** (Phase-5
  `defect_detection_rate` = 0.00). This is a model-capability ceiling — every wiring point fires
  correctly and the read-only grounding tools are available; the model just doesn't use them
  deeply. Strengthening the prompt did **not** change it.
- Multi-step task convergence is weak (Phase-5 `task_success_rate` ≈ 0.33) — notably, on some
  "failed" tasks the *feature under test still worked* (helper reused, memory carried); the run
  failed only because the 7B couldn't finish the coding.

**The lever:** model choice is **config-only** (the provider abstraction) — point the coder/reviewer
at a stronger or hosted model via `MODELS__CODER__MODEL` / `MODELS__REVIEWER__MODEL` with **zero
code change**, and the Phase-5 baseline will show the improvement in hard numbers.

*Full evidence and the honest findings live in the phase docs — especially
[Phase 4 (review)](docs/phases/phase-4-review.md) and [Phase 5 (evals)](docs/phases/phase-5-evals.md).*

---

## Where to start

Pick the path that fits why you're here:

- 🧭 **New here — what is this?** → [At a glance](#at-a-glance) above, then the
  **[runtime flow](docs/runtime-flow.md)** (one annotated diagram of a single run).
- 🎯 **Evaluating the engineering (interview / review)?** →
  [Validation & known limitations](#validation--known-limitations) above, then the
  **[phase writeups](docs/phases/)** — the build story with real evidence: the
  [RAG A/B](docs/phases/phase-3-rag-and-memory.md), the [honest 7B reviewer finding](docs/phases/phase-4-review.md),
  and the [eval baseline](docs/phases/phase-5-evals.md).
- ▶️ **Want to run it?** → [Quickstart](#quickstart) below.
- 🔍 **Going deep on design?** → **[Architecture](docs/ARCHITECTURE.md)** and the
  **[decision records (ADRs)](docs/adr/)**.

## Documentation map

Everything lives in **[`docs/`](docs/README.md)** (that's the hub — start there to browse). Two
kinds of phase docs, deliberately kept separate:

| | |
|---|---|
| **[Docs hub](docs/README.md)** | the index / map of all documentation |
| **[Runtime flow](docs/runtime-flow.md)** | what happens inside one run, node by node |
| **[Architecture](docs/ARCHITECTURE.md)** | the deep design and rationale |
| **[Phase writeups →](docs/phases/)** *(as-built)* | what actually shipped + how each phase (0–5) was verified |
| **[Build plans →](docs/build-plans/)** *(specs)* | the forward-looking spec written *before* each phase |
| **[Roadmap](docs/build-plans/ROADMAP.md)** | the full 0–7 phase sequence and why this order |
| **[ADRs](docs/adr/)** | load-bearing decisions + the alternatives rejected |

Every phase writeup carries `← Hub · ← Previous · Next →` navigation, so you can read the whole
story front-to-back or jump straight to what you need.

---

## Quickstart

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv), Ollama, and Docker (for
the sandbox). ripgrep is optional (bundled in the sandbox image; a Python fallback is used
otherwise).

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
uv run pytest                # hermetic tests (fast, no external services)
uv run pytest -m integration # live tests (needs Ollama + Docker + Postgres)

# 5. Smoke the whole graph with the real model, end to end
#    (on a 16 GB CPU box, lower the context + use the subprocess sandbox to fit RAM)
$env:OLLAMA__DEFAULT_NUM_CTX=4096; $env:SANDBOX__BACKEND="subprocess"; uv run python ../scripts/smoke_graph.py

# 6. Seed long-term memory from the repo's ADRs, then run the live validation / eval suite
#    (needs pgvector). Each script is a self-contained evidence harness — see the phase docs.
uv run python ../scripts/seed_memory.py my-project
uv run python ../scripts/rag_validate.py abtest    # does RAG change agent behavior?
uv run python ../scripts/memory_e2e.py             # does a past run change future planning?
uv run python ../scripts/review_e2e.py             # does the reviewer catch a seeded defect?
uv run python ../scripts/run_evals.py              # score all 5 tasks, gate on precision@k
```

Postgres is needed for the pgvector-backed features and the Postgres-checkpointer test (SQLite
is the default and needs no external service):

```bash
docker compose -f infra/docker-compose.yml up -d postgres   # just Postgres + pgvector
docker compose -f infra/docker-compose.yml up -d            # optional: full infra incl. Langfuse
```

Inspect the compiled graph visually (optional):

```bash
uvx --with-editable . --from "langgraph-cli[inmem]" langgraph dev
```

---

## Configuration

Settings are env-driven (see [`backend/.env.example`](backend/.env.example)); nested keys use
the `__` delimiter. **Swapping a model or provider is config-only** — the single lever for
moving past the local-7B ceiling documented in the phase writeups:

```bash
MODELS__CODER__MODEL=llama3.1:8b     # change the coder model
MODELS__REVIEWER__MODEL=llama3.1:8b  # change the reviewer model independently
SANDBOX__BACKEND=subprocess          # fallback when Docker is unavailable
CHECKPOINTER__BACKEND=postgres       # durable checkpointer: sqlite (default) or postgres
LANGFUSE__ENABLED=true               # turn on tracing
OLLAMA__DEFAULT_NUM_CTX=4096         # smaller context = less RAM (helps on a 16 GB box)
PLANNER__MEMORY_LONG_TERM_K=5        # conventions/decisions injected into planning
GRAPH__MAX_REVIEW_CYCLES=2           # review/fix cycles before escalating to a human
REVIEWER__GROUNDING_STEPS=4          # bounded read-only grounding rounds before a Review
```

---

## Repository layout

```
.
├─ backend/
│  ├─ app/
│  │  ├─ core/             # config, logging, tracing, errors, clock
│  │  ├─ providers/        # LLM provider abstraction + Ollama adapter
│  │  ├─ tools/            # tool protocol, sandbox, fs/search/git/shell, authorization
│  │  ├─ agents/           # planner, coder ReAct loop, reviewer, budgets
│  │  ├─ graph/            # LangGraph state, nodes, routing, checkpointer, instrumentation
│  │  ├─ verify/           # deterministic verify runner
│  │  ├─ workspace/        # git-backed workspace lifecycle
│  │  ├─ rag/              # chunker, embeddings, vector store, BM25, hybrid retriever, indexer
│  │  ├─ memory/           # long-term (semantic) + episodic memory + ADR ingestion writer
│  │  └─ evals/            # Phase-5 eval harness: task suite, runner, metrics, regression gate
│  ├─ evals/baseline.json  # recorded metric baseline the regression gate diffs against
│  └─ tests/               # hermetic + integration tests
├─ infra/                  # docker-compose, Postgres init, sandbox image
├─ scripts/                # bootstrap, smoke, seed_memory, and live validation/eval harnesses
├─ docs/                   # docs index, ARCHITECTURE, runtime-flow, phase writeups, ADRs, build-plans
└─ workspaces/             # runtime project sandboxes (git-ignored)
```

---

## Testing

- **Hermetic** (`uv run pytest`) — no external services; runs by default (**246 tests**),
  including the eval scoring path tested from hand-built reports with no live model.
- **Integration** (`uv run pytest -m integration`) — opt-in (**27 tests**): Docker sandbox,
  live Ollama, Postgres checkpointer, live e2e coder run, the RAG/memory stack against real
  pgvector, and the live reviewer schema-validity test.
- **Scored eval suite** — runs as a script (`scripts/run_evals.py`), not a pytest test; see
  [Phase 5](docs/phases/phase-5-evals.md).

Every phase is gated on ruff + mypy (strict) + the hermetic suite; the deep validation of each
is written up in its [phase doc](docs/README.md).

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations (config, providers, structured output, logging/tracing, infra) | ✅ Complete |
| 1 | Core coder loop (tools, sandbox, git workspace, verify, budgets) | ✅ Complete |
| 2 | LangGraph orchestration + HITL (`plan→…→finalize`, checkpointer, interrupts) | ✅ Complete |
| 3 | Repository indexing, hybrid RAG, memory | ✅ Complete |
| 4 | Autonomous review & self-correction (fresh-context reviewer, bounded fix loop) | ✅ Complete |
| 5 | Eval harness (scored task suite, deterministic-gated regression, recorded baseline) | ✅ Complete |
| 6 | Mission-control UI (Next.js: live graph, streaming, diff viewer, HITL cards) | 📋 Planned |
| 7 | Cloud provider swap + scale (hosted providers, task queue, horizontal) | 📋 Planned |

Full plans: [`docs/build-plans/`](docs/build-plans/).

---

## License

MIT.
