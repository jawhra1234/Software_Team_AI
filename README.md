# AI Software Engineering Workspace

A **supervised coding-agent workspace**: a LangGraph state machine that drives a small
set of capability-bounded agents through a `plan → code → verify → review` loop over a
**git-backed workspace**, grounded in the real repository, with sandboxed execution,
human-in-the-loop gates, and full observability.

It is **not** a chatbot, and **not** a one-prompt app generator. Agents are defined by
tool boundaries, context isolation, and verification — not by human job titles. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the decision records in
[`docs/adr/`](docs/adr/) for the full design and rationale.

> **Status:** Phase 0 (Foundations), Phase 1 (Core coder loop), and Phase 2 (LangGraph
> orchestration + HITL) are complete and verified. Phases 3–7 are specced in
> [`docs/build-plans/`](docs/build-plans/) and not yet implemented.

---

## Architecture flow

Two different "flows" matter here, and it's easy to conflate them: the **runtime flow**
is what happens *inside a single run* (right now, at any point in the project); the
**phase flow** is the *order the system itself was built in*. Both are below.

### Runtime flow — what happens inside one run (built in Phase 2)

```
  user request
       │
       ▼
  ┌─────────────┐
  │    PLAN     │  ground in the real repo (read-only tools) → draft a Plan
  └──────┬──────┘  blocking question → pause and ask the human directly
         │
         ▼  (semi/manual autonomy: needs approval)
  ┌─────────────┐
  │ HUMAN_GATE  │  approve → continue │ revise → back to PLAN │ abort → FINALIZE
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │    CODER    │◀── loops here until every task is done
  │             │     (or, in "fix mode", until verify/review's specific
  │             │      feedback is addressed)
  └──────┬──────┘  commits to git each step; may pause to ask approval
         │          before running a shell command
         ▼  (all tasks done)
  ┌─────────────┐
  │   VERIFY    │  no LLM — just runs the project's real tests/build
  └──────┬──────┘
         │  fails → back to CODER ("fix mode") │ passes ↓
         ▼
  ┌─────────────┐
  │   REVIEW    │  approve → continue │ changes_requested → back to CODER
  └──────┬──────┘  (today: a simple rule; Phase 4 swaps in a real reviewer)
         ▼  (approved)
  ┌─────────────┐
  │  FINALIZE   │  diff summary + final status (succeeded/failed/cancelled)
  └─────────────┘

  At any point, any node can instead escalate to HUMAN_GATE — budget
  exhausted, retries used up, or a command needs approval — and the human
  can retry, accept the current state as-is, or abort the run.
```

- **`plan`** — grounds itself in the real repo (read-only tools), drafts a task list, and
  only pauses to ask the human a question when it's genuinely blocking.
- **`human_gate`** — the *one* place a human is asked anything: approve the plan, resolve
  an escalation, or sign off on the final result. What it asks for depends on the
  **autonomy level** (`auto` / `semi` / `manual`).
- **`coder`** — does the actual work, one task at a time, inside a sandbox. If verify or
  review comes back with a problem, it re-enters in "fix mode" targeting that specific
  feedback instead of redoing everything.
- **`verify`** — no LLM involved. Runs the project's real tests/build and reports pass/fail.
- **`review`** — approves or requests changes to the diff (today a simple rule; Phase 4
  swaps this for a real second-opinion reviewer).
- **`finalize`** — closes the run out with a diff summary and a final status.

### Phase flow — how the project itself was built

```
  0 ──▶ 1 ──▶ 2 ──▶ 3 ──▶ 4 ──▶ 5 ──▶ 6 ──▶ 7
  ✅    ✅    ✅    📋    📋    📋    📋    📋
```

| # | Phase | What it adds |
|---|---|---|
| 0 | Foundations | config, LLM provider abstraction, `structured_call`, logging/tracing, infra |
| 1 | Core Coder Loop | tools, sandbox, git-backed workspace, deterministic verify, budgets |
| 2 | LangGraph + HITL | state graph, planner, the 6-node graph, dual checkpointers, interrupts |
| 3 | RAG + Memory | repo symbol index, hybrid retrieval, long-term/episodic memory |
| 4 | Real Reviewer | fresh-context adversarial review, replacing `review_stub` |
| 5 | Eval Harness | internal task suite, quality metrics, regression tracking |
| 6 | Mission-Control UI | Next.js: live graph, streaming, diff viewer, HITL cards |
| 7 | Cloud + Scale | hosted providers, task queue, horizontal scale |

Each phase hands the next one something concrete to build on — this is the part a
table can't show, so it's spelled out here:

- **Phase 0 → 1:** a working, swappable LLM call (`structured_call`, config-driven models,
  logging/tracing) — the substrate every agent call sits on.
- **Phase 1 → 2:** a *proven* single-agent loop (tools, sandbox, git, verify) — Phase 2
  doesn't reinvent it, it wraps it in a graph node.
- **Phase 2 → 3:** a working orchestrator that currently grounds `plan`/`coder` with
  ripgrep — Phase 3 upgrades that one seam to real hybrid RAG without touching the graph.
- **Phase 3 → 4:** real retrieval + memory — Phase 4's reviewer uses it for context instead
  of blindly trusting the diff.
- **Phase 4 → 5:** a real quality signal (reviewer verdicts) — Phase 5 measures it against
  a task suite so later changes can be judged "better" or "worse," not just "different."
- **Phase 5 → 6:** a measured, trustworthy core — only now does it make sense to build a UI
  on top of it.
- **Phase 6 → 7:** a working local product — Phase 7 is what it takes to run it for more
  than one person, on more than one machine.

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

The full graph — `plan · human_gate · coder · verify · review · finalize` — is
implemented and compiled as a real LangGraph `StateGraph` (Phase 2). Everything runs
locally on **one primary Ollama model** (`qwen2.5-coder:7b`), with a config-only path to
swap in OpenRouter / Gemini / Groq / OpenAI later.

---

## What's implemented (Phases 0–2)

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

### Phase 2 — LangGraph orchestration + HITL
- **State schema** (`app/graph/state.py`): the full `AgentState` graph state — `Plan`,
  `Task`, `VerifyResult`, `Review`, `HITLRequest`/`HITLResponse`, `Budget`, `FileRef` —
  with custom reducers (`merge_by_path`, `merge_counts`) so file changes and retry counts
  accumulate correctly across graph steps. State holds only references and truncated
  output, never raw file contents (enforced by tests).
- **The compiled graph** (`app/graph/build_graph.py`): a real LangGraph `StateGraph` with
  6 nodes — `plan → human_gate → coder → verify → review → finalize` — wired with
  conditional routing (`app/graph/routing.py`) that matches the architecture's state
  diagram exactly, and a recursion limit set above the run's step budget so escalation,
  not a raw `GraphRecursionError`, is always the terminal path.
- **Planner** (`app/agents/planner.py`, `app/graph/nodes/plan.py`): bounded read-only
  grounding (`list_dir`/`read_file`/`search_code`) followed by structured `Plan`
  emission; pauses via `interrupt()` for genuinely blocking clarification questions.
  The `Plan` schema's string-list fields use a tolerant `StrList` coercion so a small
  local model emitting list-of-objects (a common quirk) doesn't fail validation.
- **Coder node** (`app/graph/nodes/coder.py`): wraps the Phase-1 coder loop into the
  graph — selects the next task, or (in "fix mode") synthesizes an ad hoc task directly
  from failing verify/review feedback; commits at task boundaries; derives
  `changed_files` from git.
- **Verify & review-stub nodes**: `verify` (`app/graph/nodes/verify.py`) is the same
  deterministic runner from Phase 1, now wired with retry-then-escalate logic;
  `review_stub` (`app/graph/nodes/review_stub.py`) is a rule-based placeholder (approves
  when files changed) that exercises both routing branches ahead of the real adversarial
  reviewer in Phase 4.
- **Human-in-the-loop** (`app/graph/nodes/human_gate.py`): one multiplexed node handling
  `plan_approval`, `escalation`, and `final_accept`, plus a direct in-tool interrupt for
  `command_approval` before `run_command` — gated by three autonomy levels (`manual` /
  `semi` / `auto`).
- **Durable checkpointing** (`app/graph/checkpointer.py`): SQLite (clone-and-run default)
  or Postgres (`ADR-0010`), selected by config alone. A killed process resumes at the
  exact interrupt it paused at — verified against **both** backends, including a live
  Postgres restart test.
- **Budgets & instrumentation** (`app/graph/instrument.py`): a run-wide budget circuit
  breaker (steps / wall-clock / tokens) wraps every node, distinct from the Phase-1
  per-task budget; event hooks (`app/graph/events.py`) for future streaming.
- **LangGraph Studio** (`langgraph.json`, `app/graph/studio.py`): the compiled graph is
  loadable via `langgraph dev` for visual inspection.

**Verified end to end (tests):** happy-path run through all 6 nodes with zero interrupts
(auto autonomy); plan-approval interrupt/resume and plan-revise loop (semi); the full
autonomy matrix including the `final_accept` gate (manual); a verify-fail → fix → pass
loop; a run-wide budget exhaustion escalating cleanly instead of crashing; and checkpoint
recovery across a simulated process restart on **both** SQLite and live Postgres.

**Verified end to end (live model):** `scripts/smoke_graph.py` drives the whole compiled
graph with the real `qwen2.5-coder:7b` model — plan → coder → verify → review → finalize —
and it produces correct, test-passing code (`calc.py` + a passing `test_calc.py`), reaching
`status: succeeded`, including the coder recovering from a failed command mid-task.

---

## Repository layout

```
.
├─ backend/                # Python service
│  ├─ app/
│  │  ├─ core/             # config, logging, tracing, errors, clock
│  │  ├─ providers/        # LLM provider abstraction + Ollama adapter
│  │  ├─ tools/            # tool protocol, sandbox, fs/search/git/shell, authorization
│  │  ├─ agents/           # planner, coder ReAct loop, budgets, shared tool-call parsing
│  │  ├─ graph/            # LangGraph state, nodes, routing, checkpointer, instrumentation
│  │  │  └─ nodes/         # plan, coder, verify, review_stub, finalize, human_gate
│  │  ├─ verify/           # deterministic verify runner
│  │  ├─ workspace/        # git-backed workspace lifecycle
│  │  ├─ memory/ rag/ db/ api/   # placeholders for Phase 3+
│  │  └─ ...
│  ├─ tests/               # hermetic + integration tests
│  └─ langgraph.json       # LangGraph Studio entry point
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
uv run pytest -m integration # live tests (needs Ollama + Docker + Postgres), incl. e2e (~4 min)

# 5. Smoke: a validated structured LLM call
uv run python ../scripts/smoke_llm.py

# 6. Smoke: drive the whole graph with the real model, end to end
#    (on a 16 GB CPU box, lower the context + use the subprocess sandbox to fit RAM)
$env:OLLAMA__DEFAULT_NUM_CTX=4096; $env:SANDBOX__BACKEND="subprocess"; uv run python ../scripts/smoke_graph.py

# 7. Inspect the compiled graph visually (optional)
uvx --with-editable . --from "langgraph-cli[inmem]" langgraph dev
```

Postgres is needed for the Postgres-checkpointer durability test (SQLite is the default
and needs no external service). Start just that service without touching anything else
on the machine:

```bash
docker compose -f infra/docker-compose.yml up -d postgres
```

Optional full local infra (Postgres + pgvector + Langfuse):

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
CHECKPOINTER__BACKEND=postgres       # durable checkpointer: sqlite (default) or postgres
GRAPH__RECURSION_LIMIT=200           # graph-level recursion cap (above the run step budget)
OLLAMA__REQUEST_TIMEOUT_S=600        # generous for local CPU; lower for GPU/hosted
OLLAMA__DEFAULT_NUM_CTX=4096         # smaller context = less RAM (helps on a 16 GB box)
```

---

## Testing

- **Hermetic** (`uv run pytest`) — no external services; runs by default (165 tests).
- **Integration** (`uv run pytest -m integration`) — requires live Ollama, Docker, and/or
  Postgres depending on the test; opt-in so the default run stays fast and deterministic
  (7 tests: Docker sandbox, live Ollama, Postgres checkpointer, live e2e coder run).

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations (config, providers, structured output, logging/tracing, infra) | ✅ Complete |
| 1 | Core coder loop (tools, sandbox, git workspace, verify, budgets) | ✅ Complete |
| 2 | LangGraph orchestration + HITL (`plan→…→finalize`, checkpointer, interrupts) | ✅ Complete |
| 3 | Repository indexing, hybrid RAG, memory | 📋 Specced |
| 4 | Adversarial reviewer + iteration | 📋 Planned |
| 5 | Eval harness | 📋 Planned |
| 6 | Mission-control UI | 📋 Planned |
| 7 | Cloud provider swap + scale | 📋 Planned |

Full plans: [`docs/build-plans/`](docs/build-plans/).

---

## License

MIT.
