# AI Software Engineering Workspace

A **supervised coding-agent workspace**: a LangGraph state machine that drives a small
set of capability-bounded agents through a `plan → code → verify → review` loop over a
**git-backed workspace**, grounded in the real repository, with sandboxed execution,
human-in-the-loop gates, and full observability.

It is **not** a chatbot, and **not** a one-prompt app generator. Agents are defined by
tool boundaries, context isolation, and verification — not by human job titles. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the decision records in
[`docs/adr/`](docs/adr/) for the full design and rationale.

> **Status:** Phase 0 (Foundations), Phase 1 (Core coder loop), Phase 2 (LangGraph
> orchestration + HITL), Phase 3 (Repository indexing, hybrid RAG, and memory), Phase 4
> (Autonomous review & self-correction), and Phase 5 (Eval harness) are complete and
> verified. Phases 6–7 are specced in [`docs/build-plans/`](docs/build-plans/) and not
> yet implemented.

---

## Architecture flow

Two different "flows" matter here, and it's easy to conflate them: the **runtime flow**
is what happens *inside a single run* (right now, at any point in the project); the
**phase flow** is the *order the system itself was built in*. Both are below.

### Runtime flow — what happens inside one run (Phase 2 graph + Phase 3 grounding + Phase 4 review)

```
  ┌───────────────────── shared knowledge · Postgres + pgvector ──────────────────────┐
  │  Code index — hybrid RAG (vector + BM25 → RRF)                                     │
  │  Long-term memory — durable conventions & decisions (semantic)                    │
  │  Episodic memory — outcomes of past runs (relational)                             │
  └───────────────────────────────────────────────────────────────────────────────────┘
     ▲ read code · `retrieve` (PLAN + CODER)   ▲ read memory (PLAN)   ▼ write outcome (FINALIZE)

  user request
       │
       ▼
  ┌─────────────┐  ① inject Project Conventions + Previous Attempts (memory),
  │    PLAN     │  ② ground in real code via `retrieve` → draft a Plan
  └──────┬──────┘  blocking question → pause and ask the human directly
         │
         ▼  (semi/manual autonomy: needs approval)
  ┌─────────────┐
  │ HUMAN_GATE  │  approve → continue │ revise → back to PLAN │ abort → FINALIZE
  └──────┬──────┘
         ▼
  ┌─────────────┐◀── loops here until every task is done (or, in "fix mode",
  │    CODER    │    until verify/review's specific feedback is addressed) ·
  │             │    grounds in real code via `retrieve` on demand
  └──────┬──────┘  commits to git each step; may pause to ask approval
         │          before running a shell command
         ▼  (all tasks done)
  ┌─────────────┐
  │   VERIFY    │  no LLM — just runs the project's real tests/build
  └──────┬──────┘
         │  fails → back to CODER ("fix mode") │ passes ↓
         ▼
  ┌─────────────┐  fresh-context LLM reviewer — sees ONLY plan + diff + verify
  │   REVIEW    │  result (never coder's reasoning); blocker/major → fix cycle
  └──────┬──────┘  (targeted, no re-plan) │ minor/nit → advisory, never blocks
         ▼  (approved)
  ┌─────────────┐
  │  FINALIZE   │  diff summary + final status (succeeded/failed/cancelled)
  └─────────────┘  → writes the run outcome to episodic memory

  At any point, any node can instead escalate to HUMAN_GATE — budget
  exhausted, retries used up, or a command needs approval — and the human
  can retry, accept the current state as-is, or abort the run.
```

The **shared knowledge** layer (top) is Phase 3: `PLAN` and `CODER` read real code on
demand through the `retrieve` tool; `PLAN` also gets durable **memory** injected before it
drafts; and `FINALIZE` records the run's outcome back to episodic memory so future runs can
learn from it. Everything else is the Phase-2 orchestration.

- **`plan`** — before drafting, it's automatically handed relevant **memory** (durable
  decisions/conventions + how earlier runs on this project went); it then grounds in the
  real repo (hybrid RAG via the `retrieve` tool + read-only tools), drafts a task list, and
  only pauses to ask the human a question when it's genuinely blocking.
- **`human_gate`** — the *one* place a human is asked anything: approve the plan, resolve
  an escalation, or sign off on the final result. What it asks for depends on the
  **autonomy level** (`auto` / `semi` / `manual`).
- **`coder`** — does the actual work, one task at a time, inside a sandbox, grounding in
  real code on demand via `retrieve`. If verify or review comes back with a problem, it
  re-enters in "fix mode" targeting that specific feedback instead of redoing everything.
- **`verify`** — no LLM involved. Runs the project's real tests/build and reports pass/fail.
- **`review`** — a real, fresh-context reviewer (Phase 4, `ADR-0006`) with its own **isolated**
  view: only the approved plan, the diff, and the verify result — never the coder's
  reasoning. It grounds read-only (`retrieve`/`search_code`/`read_file`/`list_dir`), assigns
  each finding a severity, and only `blocker`/`major` findings send the run back to the coder
  with a **targeted** fix task (not a re-plan); `minor`/`nit` are advisory and never block.
- **`finalize`** — closes the run out with a diff summary and a final status, and records the
  run's outcome to episodic memory.

### Phase flow — how the project itself was built

```
  0 ──▶ 1 ──▶ 2 ──▶ 3 ──▶ 4 ──▶ 5 ──▶ 6 ──▶ 7
  ✅    ✅    ✅    ✅    ✅    ✅    📋    📋
```

| # | Phase | What it adds |
|---|---|---|
| 0 | Foundations | config, LLM provider abstraction, `structured_call`, logging/tracing, infra |
| 1 | Core Coder Loop | tools, sandbox, git-backed workspace, deterministic verify, budgets |
| 2 | LangGraph + HITL | state graph, planner, the 6-node graph, dual checkpointers, interrupts |
| 3 | RAG + Memory | repo symbol index, hybrid retrieval, long-term/episodic memory |
| 4 | Autonomous Review & Self-Correction | fresh-context adversarial reviewer + bounded fix loop, replacing `review_stub` |
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
- **Phase 3 → 4:** real retrieval + memory — Phase 4's reviewer uses the same read-only
  grounding tools for context instead of blindly trusting the diff text.
- **Phase 4 → 5:** a real quality signal (reviewer verdicts, severities, cycle counts) —
  Phase 5 measures it against a task suite so later changes can be judged "better" or
  "worse," not just "different."
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
implemented and compiled as a real LangGraph `StateGraph` (Phase 2), its `plan`/`coder`
grounding seam is backed by **hybrid RAG over the real repository** plus **long-term and
episodic memory** (Phase 3), and `review` is a real **fresh-context adversarial reviewer**
driving a bounded self-correction loop (Phase 4) — all three roles running on **one primary
Ollama model** (`qwen2.5-coder:7b`) with `nomic-embed-text` for embeddings, and a config-only
path to swap in OpenRouter / Gemini / Groq / OpenAI later.

---

## What's implemented (Phases 0–5)

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
  deterministic runner from Phase 1, now wired with retry-then-escalate logic; `review_stub`
  was a rule-based placeholder (approved whenever files changed) that exercised both routing
  branches ahead of the real adversarial reviewer — **replaced in Phase 4** by
  `app/graph/nodes/review.py` (see below); the node's contract (reads diff+plan+verify,
  writes `review`) was preserved exactly, so no graph/routing changes were needed.
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

### Phase 3 — Repository indexing, hybrid RAG, and memory

This phase upgrades the single grounding seam from Phase 2 (ripgrep) into real hybrid
retrieval, and adds durable memory — **without touching the graph's shape**. RAG and memory
are wired as **opt-in dependencies** (default `None`), so the hermetic test suite stays fast
and independent of Postgres.

- **Structural chunker** (`app/rag/chunker.py`): tree-sitter splits source into
  symbol-aware chunks (functions/classes with their signatures), not blind line windows —
  so a retrieved chunk is a whole, meaningful unit.
- **Embeddings** (`app/rag/embeddings.py`): `nomic-embed-text` (768-dim) via the provider
  abstraction, cached by chunk `content_hash` so re-indexing an unchanged chunk costs
  nothing. `nomic` is asymmetric, so documents and queries get their required
  `search_document:` / `search_query:` task prefixes (measurably tighter margins — see below).
- **Vector store** (`app/rag/vector_store.py`): pgvector cosine similarity, namespaced per
  project, with a symbol-name lookup for exact-symbol queries.
- **Keyword index** (`app/rag/keyword_index.py`): in-memory BM25Plus — exact symbol names
  dominate code queries, where pure vectors are weak.
- **Hybrid retriever** (`app/rag/retriever.py`): runs the vector and BM25 arms independently
  and fuses them with **Reciprocal Rank Fusion** — *no cross-encoder reranker* (ADR-0008);
  RRF combines both without tuning score scales.
- **Incremental indexer** (`app/rag/indexer.py`): content-hash diffing so a reindex only
  touches changed files; drives the `retrieve` tool and the graph's `RetrievalCapture`.
- **`retrieve` tool** (`app/tools/retrieve.py`): exposes hybrid retrieval to the agents; the
  planner and coder now call it as their **first grounding step**, and results are captured
  into `retrieved_context` on graph state (`app/graph/retrieval.py`).
- **Long-term memory** (`app/memory/long_term.py`): a namespaced pgvector store of durable
  facts (decisions, learned repo conventions), retrieved by **semantic** similarity — kept
  separate from the churny code index. Populated by a deterministic writer
  (`app/memory/ingest.py` + `scripts/seed_memory.py`) that ingests the repo's own ADRs as
  `decision` memories, so the store has real content on day one.
- **Episodic memory** (`app/memory/episodic.py`): run outcomes written at `finalize`.
  Retrieval isn't blind recency — `relevant()` re-ranks a recent window by **lexical**
  overlap with the request plus a failure bonus (past failures are the useful lesson), since
  summaries are mostly filenames/symbols/errors.
- **Memory in the planner** (`app/graph/planning_context.py`): before planning, the plan node
  searches long-term (semantic) + episodic (lexical) and injects two clearly-labeled sections
  — **Project Conventions** and **Previous Attempts** — into the planner's prompt, each shown
  only when non-empty. Repository code is **not** preloaded here: it stays on-demand via the
  `retrieve` tool, so tokens stay bounded and the agentic loop is preserved. All reads are
  best-effort (a memory outage degrades a section, never fails the run).
- **Evaluation** (`app/rag/evaluation.py`): retrieval-quality metrics (hit-rate / MRR) for
  regression tracking.
- **File-size guard** (`app/tools/fs.py`): a 1 MiB cap on `write_file` / `edit_file`, added
  after live validation surfaced a runaway-edit failure mode (below).

**Verified — retrieval works on the real codebase:** indexing this repo (103 files → 773
chunks) and querying it, hybrid retrieval returns the right symbols for both **exact** symbol
queries and **semantic paraphrases** that share no keywords with the target — with the
`nomic` task prefixes improving the query↔target cosine margin (e.g. `+0.580 → +0.614` on the
validation fixture).

#### Does RAG actually change agent behavior? (RAG OFF vs RAG ON)

Components passing tests isn't proof that retrieval *helps the agent*. So the whole LangGraph
pipeline (plan → retrieve → coder → verify → review → finalize → memory) was run live against
a controlled fixture: a checkout task whose correct answer depends on a **non-guessable helper**
(`apply_levy`, with a bespoke surcharge/rounding rule) that lives in the indexed repo but is
**never named in the task**. The task is only solvable if the agent *discovers* the helper.

| | **RAG OFF** | **RAG ON** |
|---|---|---|
| `retrieve` | `ok: false` (no index, by design) | **`ok: true` — called repeatedly** |
| `retrieved_context` | **0 chunks** | **10 chunks** — incl. `apply_levy`, `test_exact_values` |
| Found the hidden helper? | **No** — never sees `apply_levy` | **Yes** — wrote correct `from pricing_rules import apply_levy` / `return apply_levy(...)` |
| Node timeline | plan → escalate → finalize | plan `[retrieved 5]` → coder → retrieve → coder `[retrieved 10]` → … |

**What we concluded:**

1. **RAG demonstrably changes behavior.** With retrieval on, the coder grounds on the real
   helper and writes the correct, reuse-based solution; with it off, it never finds the helper
   and dead-ends. This is the core Phase-3 thesis, shown end-to-end through the actual graph,
   not just in unit tests.
2. **The local 7B model is the ceiling, not the design.** On this hard task both arms ended
   `failed` for **model-quality** reasons (not RAG/orchestration defects): the planner
   sometimes emitted an invalid task `kind`, and the coder, after writing correct code,
   corrupted the file by misusing `edit_file` as a full-file rewrite. The harness (budgets,
   no-progress detection, HITL escalation) caught every one cleanly instead of hanging. Both
   quirks are now **hardened** (see below), and the same pipeline runs fully green on a task
   within the model's reach.
3. **Real bugs were found and fixed** — hardening, not benchmark-tuning:
   - a 1 MiB cap on `write_file`/`edit_file` turns a runaway edit loop into an immediate tool
     failure the model can react to (was: a file ballooning to hundreds of MB);
   - a tolerant `Task.kind` coercion maps an off-enum value (e.g. a tool name) to a safe
     default instead of failing the whole Plan;
   - integration tests use isolated memory tables so a 2-dim test embedder can't clash with
     the real 768-dim store.

The provider abstraction (`app/providers/`) means a fully-green end-to-end run is a
**config-only swap** to a frontier hosted model; the local `qwen2.5-coder:7b` remains the
zero-cost default for development, and its stumbles double as a live demonstration that the
guardrails work under a weak model.

#### Does past experience change future planning? (memory across runs)

A second live harness (`scripts/memory_e2e.py`) proves the memory loop end-to-end by running
the **same project twice**:

- **Run 1** plans with an empty episodic store → only **Project Conventions** is injected
  (Previous Attempts is correctly omitted). At `finalize`, the run's outcome is written to
  episodic memory.
- **Run 2** (a related request) now retrieves that record: its planner prompt carries **both**
  sections — Project Conventions *and* a **Previous Attempts** entry naming run 1 — proving a
  prior run measurably shapes later planning. Repository `retrieve` still fires independently.

**Full green happy path** (`scripts/memory_e2e.py simple`): on a trivial self-contained task,
the whole pipeline runs clean — `plan → coder → verify PASS → review approve → finalize
succeeded` — with memory wired in and a `succeeded` record written back. This is the same
architecture as the hard task; only the task difficulty (not the design) decides the outcome.

Validation harnesses: `scripts/rag_validate.py` (`part1` retrieval, `abtest` RAG off/on) and
`scripts/memory_e2e.py` (cross-run memory; `simple` for the green happy path).

### Phase 4 — Autonomous review & self-correction

This phase replaces the Phase-2 `review_stub` (a rule that approved whenever files changed)
with a real, fresh-context adversarial reviewer (`ADR-0006`) and turns the existing
`review → coder(fix) → verify → review` loop into a genuine, bounded self-correction cycle —
**without changing the graph's topology or any node's contract**. The reviewer is the third
and last real LLM role in the "3 roles + deterministic verify" design (see *Why this design*
above); it plugs into the same seam the stub occupied.

- **Reviewer agent** (`app/agents/reviewer.py`): shaped exactly like the planner — a bounded,
  **read-only** grounding loop (`retrieve` / `search_code` / `read_file` / `list_dir` — never
  `write_file`/`edit_file`/`run_command`) followed by a `structured_call` that emits a
  `Review`. Its system prompt states five judging priorities in order — **correctness,
  security, test gaps, architecture (incl. duplicated logic elsewhere in the repo),
  maintainability** — and is explicit that style/formatting/naming are *never* blockers.
- **Isolation is structural, not a convention** (`app/graph/nodes/review.py`): the reviewer's
  input is built from exactly three sources — the approved `Plan`, the diff
  (`state["diff_summary"]`, already head/tail-truncated by the coder), and the `VerifyResult`
  — and the code path never reads `state["coder_scratch"]`. This isn't a prompt instruction
  the model could ignore; the node simply never has access to the coder's reasoning to leak.
- **The verdict is not trusted blindly.** A small model can say "approved" while listing a
  blocker, or "changes_requested" over a pure nit. The node **deterministically overrides**
  the effective verdict from the issues' own severities: any `blocker`/`major` forces
  `changes_requested` regardless of the model's claim; otherwise (no blocker/major) it's
  `approved` — so "only blocker/major trigger another cycle" is a property of the code, not
  a hope about model compliance. `rejected` (the approach is fundamentally wrong) and a
  malformed/unparseable review both escalate to a human rather than looping or silently
  approving.
- **Targeted fix hand-off** (`app/graph/nodes/coder.py`'s `build_fix_task`): when changes are
  requested, the coder's fix task is built **only from the blocker/major issues** (file, what's
  wrong, the reviewer's suggestion) — never a re-plan, and never diluted by minor/nit noise,
  which stay visible in `review.issues`/`summary` for tracing but are advisory-only.
  A fix always re-enters `verify` before the next `review`, so a fix that breaks the tests is
  caught, not rubber-stamped.
- **Bounded and observable**: the same cycle cap the stub already had (`GRAPH__MAX_REVIEW_CYCLES`,
  default 2) still governs the loop; every cycle logs `review_produced` (verdict + issue
  count) and, when the node overrides a mistaken verdict, `review_verdict_overridden` — so
  the whole review/fix history of a run is inspectable, not just its final outcome.

**Verified (hermetic, 16 tests, `test_reviewer.py` + `test_graph_nodes_review.py`):** bounded
grounding and structured emission (mirroring the planner's own test suite); the severity
override in both directions (a false "approved" with a blocker is forced to
`changes_requested`; a false "changes_requested" over only nits is forced to `approved`);
the cycle cap escalates instead of looping forever; a `rejected` verdict escalates
immediately regardless of remaining cycles; a malformed/unparseable reviewer response
escalates rather than fabricating a fake approval; the final-accept gate for manual autonomy
is preserved; and — the signature test — a `coder_scratch` message containing a planted
secret string never appears in any message sent to the reviewer, proving the isolation
invariant structurally, not just by convention.

**Verified (live model, `test_reviewer_integration.py`):** the real `qwen2.5-coder:7b`
reviewer reliably returns a **schema-valid** `Review` end to end through `structured_call`'s
repair-retry (no exception), with every issue's severity landing in the valid enum — proving
the reviewer's structured-output contract holds under a live model, not just a scripted one.

#### Does the live reviewer actually catch a real defect? (live validation + an honest limitation)

The hermetic tests above prove the **loop mechanics** are correct by feeding the node scripted
findings. They cannot prove that a live 7B model will *notice* a subtle defect on its own — so
a second live harness, `scripts/review_e2e.py`, was built to test exactly that: a scripted
planner and coder (deterministic, so the specific defect is reproducible) seed a real
**architecture/duplication defect** — the coder's first attempt reimplements an existing
helper's tax-rate math (`pricing_rules.apply_levy`) instead of reusing it — behind a
deliberately **weak** pre-written test that passes on both the buggy and the correct version
(`total > subtotal`), so only the reviewer, not `verify`, can catch it. The reviewer itself
runs **live** throughout.

| Run | Reviewer prompt | Result |
|---|---|---|
| 1st live run | original ("ground before judging" as guidance) | `verdict=approved`, **0 issues**, empty summary, **no grounding tool calls at all** |
| 2nd live run | strengthened — "grounding is **mandatory** before approving any non-trivial diff; check for duplicated logic" | identical outcome: `verdict=approved`, **0 issues**, empty summary, **no grounding tool calls** |

**What this shows, precisely:**
1. **The pipeline is not at fault.** Every wiring point fired correctly both runs — scripted
   plan → scripted (buggy) coder → real `verify` (passed, as designed) → **live** review →
   finalize `succeeded`. `retrieve`/`search_code`/`read_file`/`list_dir` were all available to
   the reviewer in its registry; it simply never called any of them.
2. **A genuine, reproducible model-capability limitation, not a prompt-wording problem.**
   Strengthening the system prompt to explicitly mandate grounding — the same class of fix
   that worked for the coder/planner in Phase 3 (`retrieve` added to their prompts) — made
   **no observed difference** here. The reviewer emitted its verdict via the forced structured
   "emit" tool call in ~2 minutes both times with an empty summary and zero issues, consistent
   with a small model satisfying the schema with minimal effort rather than working through
   the reasoning the prompt describes. This is a deeper ceiling than instruction wording: a
   7B model driven through a forced-schema tool call can be "technically compliant" while
   doing essentially no analysis.
3. **What remains proven despite this:** the loop's correctness does not depend on the model
   reliably noticing everything. The severity-override logic, the isolation invariant, the
   targeted fix hand-off, the bounded cycle + escalation, and the malformed-output fail-safe
   are all independently proven via the 16 hermetic tests using scripted reviewer responses
   that *do* contain findings — so when a reviewer (of any capability) does surface a
   blocker/major, the system is proven to route, fix, re-verify, and re-review it correctly.
   What isn't proven is that *this* local model will reliably surface a subtle architectural
   issue unprompted — an honest capability gap, not a design defect.
4. **Why no further prompt iteration was pursued.** Repeatedly re-wording the prompt hoping to
   eventually get a lucky pass would drift into tuning the benchmark rather than hardening the
   system — the same discipline applied in Phase 3's RAG validation. The one legitimate,
   targeted prompt fix was applied and honestly re-tested; it didn't change the outcome, and
   that result is reported here rather than concealed or re-run into a favorable draw.

**The smallest real fix, not yet applied:** since the provider abstraction already makes model
choice config-only, swapping the reviewer role to a stronger (e.g. hosted) model via
`MODELS__REVIEWER__MODEL`/`MODELS__REVIEWER__PROVIDER` is the natural next lever — a genuine
capability upgrade rather than another prompt tweak, and it requires zero code changes to
this phase's implementation.

Validation harnesses: `scripts/review_e2e.py` (live catch → fix → approve attempt, with the
finding above) — run alongside `scripts/rag_validate.py` and `scripts/memory_e2e.py` from
Phase 3 for the full live-validation suite.

### Phase 5 — Eval harness

Phases 3-4 were each proven by running a script by hand and reading the log. That's rigorous
once, but not **comparable**: there was no way to tell whether next week's prompt tweak made
things better or worse. Phase 5 turns those one-off scripts into a small, standing, **scored
regression suite** — every run reduces to hard numbers saved to a baseline, so any later
change (a prompt edit, a model swap, a config tweak) can be judged *better or worse*, not
just *different*. It adds no runtime node and changes no graph contract; it's a measurement
layer that drives the existing pipeline.

```
  ┌──────────── fixed task suite — 5 frozen fixtures (app/evals/tasks.py) ────────────┐
  │  happy_path · rag_required · defect_injection · cross_run_memory · retrieval@k     │
  └───────────────────────────────────────┬───────────────────────────────────────────┘
                                          │  each task →
                                          ▼
                          ┌──────────────────────────────┐
                          │   run the REAL pipeline live   │  plan→coder→verify→review→finalize
                          │   & capture the outcome        │  (retrieval task = precision@k only)
                          └───────────────┬───────────────┘
                                          ▼   per-task RunReport
                          ┌──────────────────────────────┐
                          │      aggregate metrics         │
                          │  • deterministic → GATE         │  retrieval precision@k
                          │  • stochastic   → trend         │  success / defect-detect / false-flag / cycles
                          └───────────────┬───────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │   diff vs backend/evals/        │  exit non-zero ONLY on a deterministic
                          │   baseline.json                 │  drop; stochastic metrics reported, never gated
                          └──────────────────────────────┘
```

- **The task suite** (`app/evals/tasks.py`): five frozen fixtures reusing what Phases 3-4
  already validated live — a happy-path build, a RAG-required task (hidden helper), a
  defect-injection task (the reviewer *should* catch a planted duplication — its coder is
  **scripted** so the defect reliably exists), a cross-run memory task (two runs), and a
  retrieval precision@k measurement. Each carries an automatic pass/fail check — none take
  arbitrary input, so their numbers are comparable across runs.
- **The runner** (`app/evals/runner.py`): streams one real graph run to a terminal state
  (auto-aborting escalation interrupts, since an eval has no human), reducing it to a
  task-agnostic capture — review verdicts per cycle, retrieved symbols, verify pass/retries,
  step count, wall-clock.
- **Metrics, split by reproducibility** (`app/evals/metrics.py`): a **deterministic** set
  (retrieval precision@k — same index + embeddings → same number, so it's *gate-worthy*) and
  a **stochastic** set (success rate, defect-detection rate, false-flag rate,
  cycles-to-converge, steps, latency). The 5-6 task counts are honest *indicative rates*,
  **not** claimed as statistical precision/recall.
- **Regression gate** (`app/evals/regression.py` + `scripts/run_evals.py`): diffs a run
  against `backend/evals/baseline.json` and exits non-zero **only** if a *deterministic*
  metric regressed — a local 7B jitters run to run, so hard-gating a stochastic metric would
  fire on noise. Stochastic metrics are printed as a before/after trend.
- **Tested hermetically** (`test_evals_metrics.py` / `_regression.py` / `_runner.py`): the
  whole scoring path — aggregation, the deterministic-gate-vs-stochastic-trend split, the
  baseline round-trip, and `run_graph` capturing a real graph run via `FakeProvider` — runs
  without a live model, from hand-built reports.

#### The recorded live baseline — and what the numbers honestly say

Running the suite once against the real `qwen2.5-coder:7b` (`scripts/run_evals.py
--update-baseline`, ~54 min on a 16 GB box) produced `backend/evals/baseline.json`:

| Metric | Value | Reading |
|---|---|---|
| **`retrieval_precision_at_k`** (deterministic, gated) | **0.75** | RAG surfaces the right symbol in 3 of 4 fixed queries |
| `task_success_rate` | 0.33 | 1 of 3 multi-step tasks fully converged |
| `defect_detection_rate` | 0.00 | the reviewer missed the planted defect (the Phase-4 finding, now quantified) |
| `false_flag_rate` | 0.00 | the reviewer did **not** wrongly block correct code |
| `memory_influence_rate` | 1.00 | a past run's memory correctly shaped the next run's planning |

**The key thing the baseline shows** is a clean separation between *what the architecture does*
and *what the local model can't do*. On two of the "failed" tasks the **feature under test
actually worked** — the coder reused the hidden helper (`reused_helper=True`), and cross-run
memory carried run 1's record into run 2's planning (`memory_influenced=True`) — but the run
still ended `failed` because the 7B couldn't fully converge the multi-step coding. So
`task_success_rate 0.33` and `defect_detection 0.00` are **model-capability numbers, not
design defects**, consistent with every Phase 3-4 finding.

This is recorded exactly as it came out — not massaged. It's now the reference every future
change is measured against: because model choice is a **config-only swap** (the provider
abstraction), pointing the coder/reviewer at a stronger model should move the stochastic
metrics up, and the baseline will show it in hard numbers. The deterministic
`retrieval_precision_at_k = 0.75` is the reproducible anchor the regression gate protects.

Validation harness: `scripts/run_evals.py` (`--category <name>` to run one category,
`--update-baseline` to record).

---

## Repository layout

```
.
├─ backend/                # Python service
│  ├─ app/
│  │  ├─ core/             # config, logging, tracing, errors, clock
│  │  ├─ providers/        # LLM provider abstraction + Ollama adapter
│  │  ├─ tools/            # tool protocol, sandbox, fs/search/git/shell, authorization
│  │  ├─ agents/           # planner, coder ReAct loop, reviewer, budgets, tool-call parsing
│  │  ├─ graph/            # LangGraph state, nodes, routing, checkpointer, instrumentation
│  │  │  └─ nodes/         # plan, coder, verify, review, finalize, human_gate
│  │  ├─ verify/           # deterministic verify runner
│  │  ├─ workspace/        # git-backed workspace lifecycle
│  │  ├─ rag/              # chunker, embeddings, vector store, BM25, hybrid retriever, indexer, eval
│  │  ├─ memory/           # long-term (semantic) + episodic memory + ADR ingestion writer
│  │  ├─ evals/            # Phase-5 eval harness: task suite, runner, metrics, regression gate
│  │  ├─ db/ api/          # placeholders for Phase 6+
│  │  └─ ...
│  ├─ evals/               # baseline.json — the recorded metric baseline the gate diffs against
│  ├─ tests/               # hermetic + integration tests
│  └─ langgraph.json       # LangGraph Studio entry point
├─ infra/                  # docker-compose, Postgres init, sandbox image
├─ scripts/                # bootstrap, smoke, seed_memory, and live validation/eval harnesses
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

# 7. Seed long-term memory from the repo's ADRs (needs pgvector)
uv run python ../scripts/seed_memory.py my-project

# 8. Validate live (needs pgvector). Each is a self-contained evidence harness:
uv run python ../scripts/rag_validate.py part1     # retrieval works on the real repo
uv run python ../scripts/rag_validate.py abtest    # does RAG change agent behavior?
uv run python ../scripts/memory_e2e.py             # does a past run change future planning?
uv run python ../scripts/memory_e2e.py simple      # full green pipeline on a trivial task
uv run python ../scripts/review_e2e.py             # does the reviewer catch a seeded defect?

# 9. Run the scored eval suite live + diff vs baseline (Phase 5; needs pgvector)
uv run python ../scripts/run_evals.py              # score all 5 tasks, gate on precision@k
uv run python ../scripts/run_evals.py --update-baseline   # record a new baseline

# 10. Inspect the compiled graph visually (optional)
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
PLANNER__MEMORY_LONG_TERM_K=5        # conventions/decisions injected into planning
PLANNER__MEMORY_EPISODIC_K=3         # past runs surfaced as "Previous Attempts"
MODELS__REVIEWER__MODEL=llama3.1:8b  # change the reviewer model independently of the coder
GRAPH__MAX_REVIEW_CYCLES=2           # review/fix cycles before escalating to a human
REVIEWER__GROUNDING_STEPS=4          # bounded read-only grounding rounds before emitting Review
```

---

## Testing

- **Hermetic** (`uv run pytest`) — no external services; runs by default (246 tests),
  including the Phase-5 eval scoring path (metrics, regression gate/trend split, and
  `run_graph` capture) tested from hand-built reports with no live model.
- **Integration** (`uv run pytest -m integration`) — requires live Ollama, Docker, and/or
  Postgres depending on the test; opt-in so the default run stays fast and deterministic
  (27 tests: Docker sandbox, live Ollama, Postgres checkpointer, live e2e coder run, the
  RAG/embeddings/retriever/memory + memory-in-planner stack against real pgvector, and the
  live reviewer schema-validity test). The scored eval suite itself runs as a script
  (`scripts/run_evals.py`), not a pytest test.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations (config, providers, structured output, logging/tracing, infra) | ✅ Complete |
| 1 | Core coder loop (tools, sandbox, git workspace, verify, budgets) | ✅ Complete |
| 2 | LangGraph orchestration + HITL (`plan→…→finalize`, checkpointer, interrupts) | ✅ Complete |
| 3 | Repository indexing, hybrid RAG, memory | ✅ Complete |
| 4 | Autonomous review & self-correction (fresh-context reviewer, bounded fix loop) | ✅ Complete |
| 5 | Eval harness (scored task suite, deterministic-gated regression, recorded baseline) — see [`PHASE-5.md`](docs/build-plans/PHASE-5.md) | ✅ Complete |
| 6 | Mission-control UI | 📋 Planned |
| 7 | Cloud provider swap + scale | 📋 Planned |

Full plans: [`docs/build-plans/`](docs/build-plans/).

---

## License

MIT.
