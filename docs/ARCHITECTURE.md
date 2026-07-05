# AI Software Engineering Workspace — Architecture Specification

> **Status:** Approved for implementation
> **Audience:** Implementing engineers
> **Scope:** Complete system design. This document is the single source of truth for *what* to build. Rationale for individual decisions lives in `docs/adr/`. Build sequencing lives in `docs/build-plans/`.

---

## 1. What this system is (and is not)

A **supervised coding-agent workspace**: a LangGraph state machine that drives a small set of capability-bounded agents through a `plan → code → verify → review` loop over a **git-backed workspace**, grounded in the real repository via code RAG, with human-in-the-loop gates and full observability.

- It is **not** a chatbot.
- It is **not** a one-prompt app generator.
- It is **not** a role-play "team" of seven AI job-titles. The "team of AI engineers" is a **UI narrative**, not the execution architecture. See [ADR-0001](adr/0001-capability-bounded-agents.md).

### Design tenets

1. **Agents are justified by tool boundaries, context isolation, genuine parallelism, or a different model — never by human job titles.**
2. **Three sources of truth, deliberately separated** (see §2). Control flow in state; code on disk; knowledge in the vector store.
3. **Close the loop.** Generated code is run (tests/build/lint) and failures feed back. This is what separates a tool from a demo.
4. **Ground before writing.** Read real files; never hallucinate structure.
5. **Small state.** No file contents and no full command output in graph state — only references and truncated tails. Critical on a 16 GB machine.
6. **One primary local model.** Optimize around a single Ollama model; avoid model-reload thrash. Provider/model is config, not architecture.

---

## 2. Three sources of truth

| Concern | Lives in | Why |
|---|---|---|
| Control flow + small structured artifacts (plan, task list, statuses, verdicts) | **LangGraph state** (checkpointed) | Small, serializable, drives the graph, cheap to checkpoint |
| The actual code | **Git-backed workspace on disk** | State stays tiny; git gives diffs, rollback, history for free |
| Retrievable knowledge (repo chunks, decisions, cross-session memory) | **Vector store + Postgres** | Repo ≫ context window; the RAG/memory substrate |

Agents communicate **through state (control) and the filesystem (code)** — never by chat-passing messages to each other. No event bus, no agent-to-agent messaging. See [ADR-0002](adr/0002-three-sources-of-truth.md).

---

## 3. High-level diagram

```
                    ┌──────────────────────────────────────────┐
   User ──HTTP/WS──▶ │  FastAPI  (SSE/WS streaming, HITL resume) │
                    └───────────────────┬──────────────────────┘
                                        │  invoke / stream / resume(interrupt)
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                   LangGraph  (Orchestrator)                    │
        │                                                                │
        │   plan ─(HITL: approve)─▶ coder ⇄ (next task) ─▶ verify        │
        │     ▲                       ▲                     │            │
        │     │ revise                │ fix                 │ pass/fail   │
        │     └──── human_gate ◀── escalation ◀── budget    ▼            │
        │              │                              review ─▶ finalize │
        └───────┬──────┴───────────────┬───────────────┬────────┬───────┘
                │                       │               │        │
         ┌──────▼─────┐         ┌───────▼──────┐  ┌─────▼──────┐ │
         │  Provider  │         │    Tools     │  │ Memory/RAG │ │
         │ abstraction│         │ fs·ripgrep·  │  │ pgvector + │ │
         │(Ollama/... )│         │ git·sandbox  │  │ nomic-embed│ │
         └──────┬─────┘         │ run_command  │  └────────────┘ │
                │               └──────┬───────┘         ┌────────▼─────┐
          ┌─────▼────┐          ┌──────▼───────────────┐ │ Checkpointer │
          │  Ollama  │          │ Sandbox (Docker:     │ │ (Postgres/   │
          │Qwen2.5-  │          │ no-net, cwd-jail,    │ │  SQLite dev) │
          │Coder 7B  │          │ timeout, allowlist)  │ └──────────────┘
          └──────────┘          └──────────────────────┘
```

---

## 4. LangGraph node specifications

The graph is **6 nodes**: `plan · human_gate · coder · verify · review · finalize`.
HITL is one multiplexed `human_gate` node plus one in-tool interrupt for command approval.

### Topology

```
START → plan → human_gate[plan_approval] ─approve→ coder ⇄ (next task)
                    │revise→ plan                    │ all done
                    │abort → finalize                ▼
                                                    verify ─pass→ review ─approved→ finalize → END
                                                      │fail(retry)     │changes(retry)
                                                      └──→ coder ←──────┘
   budget/loop/exhausted ──→ human_gate[escalation] ─(retry|accept|abort)→ coder|plan|finalize
```

### 4.1 `plan`
| Field | Spec |
|---|---|
| **Purpose** | Turn a raw request into a validated, executable `Plan` (spec + architecture notes + ordered task list). Absorbs intake/clarification and architecture. |
| **Inputs** | `user_request`, project context (repo summary + symbol index), prior `plan` if revising. |
| **Outputs** | A `Plan` object; optionally a clarification `interrupt`. |
| **State read** | `user_request`, `project_id`, `plan`, `review`/`verify_result` (if re-planning), `retrieved_context`, `autonomy_level`, `budget`. |
| **State written** | `plan` (spec embedded), `needs_clarification`, `clarification_answers`, `budget`, `node_history`. |
| **Tools** | `retrieve`, `search_code`, `list_dir`, `read_file` (read-only — **no writes in plan**). |
| **Prompt responsibilities** | (1) Detect blocking ambiguity → structured `open_questions` + `interrupt` only if blocking; else record `assumptions` and proceed. (2) Architecture + tech-stack grounded in existing repo. (3) Decompose into `Task[]` with `target_paths`, `acceptance_criteria`, `depends_on`, `kind`. (4) Emit validated JSON via `structured_call`. |
| **Transitions** | Blocking questions & autonomy ≠ `auto` → `interrupt` (resume re-enters `plan`). Else → `human_gate[plan_approval]`. |
| **Failure/retry** | Malformed JSON → repair-prompt retry ×2, then `human_gate[escalation]`. Empty task list = failure. Idempotent: re-running overwrites `plan` (version++). |

### 4.2 `human_gate` (multiplexed HITL)
| Field | Spec |
|---|---|
| **Purpose** | Single reusable pause point. `hitl_request.kind ∈ {plan_approval, escalation, final_accept}`. |
| **Inputs** | `hitl_request` payload (kind, context summary, options). |
| **Outputs** | `hitl_response` (decision + optional edits/notes). |
| **State read** | `plan`, `diff_summary`, `verify_result`, `review`, `errors`, `autonomy_level`. |
| **State written** | `hitl_response`; for `plan_approval` with edits: patched `plan`. |
| **Tools** | None. |
| **Prompt responsibilities** | None — pure `interrupt(payload)`. Payload is rendered by the UI, not the LLM. |
| **Transitions** | `plan_approval`: approve→`coder`; revise→`plan`; abort→`finalize(cancelled)`. `escalation`: retry→`coder`/`plan`; accept-as-is→`finalize`; abort→`finalize(failed)`. `final_accept`: accept→`finalize`; request-changes→`coder`. |
| **Failure/retry** | Resume durable via checkpointer; crash while paused resumes at the same interrupt. Invalid response schema → re-prompt UI, don't advance. |

### 4.3 `coder`
| Field | Spec |
|---|---|
| **Purpose** | Executes the current task via an internal ReAct tool loop until acceptance criteria are met or a step/token budget trips. Also runs `docs` tasks. |
| **Inputs** | Current `Task`, retrieved context, workspace path, prior `verify_result`/`review` when in fix mode. |
| **Outputs** | File mutations on disk (git-staged), updated `Task.status`, `changed_files`, per-task summary. |
| **State read** | `plan`, `current_task_id`, `workspace_path`, `work_branch`, `retrieved_context`, `verify_result`, `review`, `budget`, `autonomy_level`. |
| **State written** | `changed_files`, `diff_summary`, `plan.tasks[i].status/attempts`, `current_task_id`, `coder_scratch` (pruned), `budget`, `errors`, `node_history`. |
| **Tools** | `read_file`, `edit_file` (patch preferred), `write_file`, `list_dir`, `search_code`, `retrieve`, `run_command` (sandboxed), `git_status/diff/add/commit`, `finish_task`. |
| **Prompt responsibilities** | Ground before writing. Smallest change that satisfies acceptance criteria. Self-check via targeted commands. `finish_task` with summary when done. In fix mode, address specific issues only. |
| **Transitions** | Task done + tasks remain → `coder`. All done → `verify`. Budget/loop exceeded → `human_gate[escalation]`. |
| **Failure/retry** | Tool errors fed back into loop (self-correct). `run_command` non-zero exit is data, not a crash. Inner loop capped by `max_steps_per_task`; per-task `attempts` capped (default 3). |

### 4.4 `verify` (deterministic — no LLM)
| Field | Spec |
|---|---|
| **Purpose** | Objective ground truth. Runs test/build/lint/typecheck commands in the sandbox; returns structured results. |
| **Inputs** | `workspace_path`, verify command set (auto-detected from repo). |
| **Outputs** | `VerifyResult` (per-check pass/fail, exit codes, truncated stdout/stderr tails). |
| **State read** | `workspace_path`, `plan`. |
| **State written** | `verify_result`, `retries.verify`, `node_history`. |
| **Tools** | `run_command` (sandboxed) only. No LLM call. |
| **Prompt responsibilities** | None. Command detection is rule-based (presence of `package.json`, `pyproject.toml`, …), configurable per project. |
| **Transitions** | All pass → `review`. Any fail & `retries.verify < cap` → `coder` (fix mode). Fail & cap reached → `human_gate[escalation]`. |
| **Failure/retry** | Timeout = fail (guards infinite loops). Infra error (not a test failure) → retry ×1 then escalate. Output truncated head+tail. |

### 4.5 `review` (adversarial, fresh context)
| Field | Spec |
|---|---|
| **Purpose** | Independent critique of the *diff* against the *requirements*, with no visibility into the Coder's reasoning. |
| **Inputs** | `git diff base_commit..HEAD`, `spec`/`plan`, `verify_result`. |
| **Outputs** | `Review` (verdict + issues by severity + suggestions). |
| **State read** | `plan`, `verify_result`, `workspace_path`, `base_commit`. **Does NOT read `coder_scratch`.** |
| **State written** | `review`, `retries.review`, `node_history`. |
| **Tools** | `git_diff`, `read_file`, `search_code`, `retrieve` (read-only). |
| **Prompt responsibilities** | Judge correctness, requirement coverage, security, obvious quality. Severity `blocker/major/minor/nit`. `changes_requested` only if a `blocker`/`major` exists — do not nitpick-block. Structured output. |
| **Transitions** | `approved` → `finalize` (or `human_gate[final_accept]` in manual autonomy). `changes_requested` & retries left → `coder`. `rejected`/exhausted → `human_gate[escalation]`. |
| **Failure/retry** | Review→fix cycles capped (default 2). Malformed output → repair retry ×2. |

### 4.6 `finalize`
| Field | Spec |
|---|---|
| **Purpose** | Terminal node. Produce final artifact (diff, summary, commit/branch), write episodic memory, close the run. |
| **Inputs** | Final `status`, workspace git state. |
| **Outputs** | Final `diff_summary`, artifact records, run outcome. |
| **State read** | Everything relevant for the summary. |
| **State written** | `status` (`succeeded/failed/cancelled`), artifact refs, episodic memory write. |
| **Tools** | `git_commit`/`git_diff`, memory write. |
| **Prompt responsibilities** | Optional short LLM run summary; otherwise deterministic. |
| **Transitions** | → `END`. |
| **Failure/retry** | Best-effort; failures logged, don't block run closure. |

---

## 5. State schema

```python
# ---- value objects (Pydantic) ----
class Budget(BaseModel):
    max_tokens: int; max_steps: int; max_wall_clock_s: int
    tokens_used: int = 0; steps_used: int = 0; started_at: str  # ISO, injected

class FileRef(BaseModel):                 # NEVER carries file contents
    path: str
    status: Literal["added","modified","deleted","unchanged"]
    blob_sha: str | None = None

class Task(BaseModel):
    id: str; title: str; description: str
    kind: Literal["create","modify","test","docs","fix"]
    target_paths: list[str] = []
    acceptance_criteria: list[str] = []
    depends_on: list[str] = []
    status: Literal["pending","in_progress","done","failed","skipped"] = "pending"
    attempts: int = 0

class Plan(BaseModel):
    version: int = 1
    summary: str
    functional_requirements: list[str]
    non_functional: list[str] = []
    constraints: list[str] = []
    assumptions: list[str] = []
    open_questions: list[str] = []
    architecture_notes: str = ""
    tech_stack: dict[str, str] = {}
    tasks: list[Task]

class CheckResult(BaseModel):
    name: str; cmd: str; passed: bool; exit_code: int
    stdout_tail: str; stderr_tail: str

class VerifyResult(BaseModel):
    passed: bool; checks: list[CheckResult]; summary: str

class ReviewIssue(BaseModel):
    severity: Literal["blocker","major","minor","nit"]
    file: str | None; line: int | None
    description: str; suggestion: str | None = None

class Review(BaseModel):
    verdict: Literal["approved","changes_requested","rejected"]
    issues: list[ReviewIssue] = []; summary: str

class RetrievedChunk(BaseModel):          # ephemeral, not long-term persisted
    path: str; symbol: str | None; score: float; content: str

class HITLRequest(BaseModel):
    kind: Literal["plan_approval","escalation","final_accept","command_approval"]
    context: str; options: list[str]; payload: dict = {}

class HITLResponse(BaseModel):
    decision: str; edits: dict = {}; note: str | None = None

class ErrorRecord(BaseModel):
    node: str; kind: str; message: str; ts: str

# ---- the graph State ----
class AgentState(TypedDict):
    # identity
    run_id: str; project_id: str; thread_id: str
    # request / planning
    user_request: str
    intent: NotRequired[str]
    needs_clarification: NotRequired[bool]
    clarification_answers: Annotated[list[str], add]
    plan: NotRequired[Plan]                 # spec fields live inside Plan
    # execution
    current_task_id: NotRequired[str | None]
    workspace_path: str
    base_commit: NotRequired[str]
    work_branch: NotRequired[str]
    changed_files: Annotated[list[FileRef], merge_by_path]
    diff_summary: NotRequired[str]
    coder_scratch: Annotated[list[AnyMessage], add_messages]   # pruned aggressively
    # quality gates
    verify_result: NotRequired[VerifyResult]
    review: NotRequired[Review]
    retrieved_context: NotRequired[list[RetrievedChunk]]        # ephemeral per step
    # control / HITL / observability
    autonomy_level: Literal["manual","semi","auto"]
    hitl_request: NotRequired[HITLRequest]
    hitl_response: NotRequired[HITLResponse]
    budget: Budget
    retries: Annotated[dict[str,int], merge_counts]            # keyed by node
    errors: Annotated[list[ErrorRecord], add]
    node_history: Annotated[list[str], add]                    # loop detection + tracing
    status: Literal["planning","running","paused","succeeded","failed","cancelled"]
```

**Reducer rules:** `changed_files` merges by path (latest status wins); `coder_scratch`/`errors`/`node_history`/`clarification_answers` append; `retries` sums; everything else overwrites.
**Invariant:** no field ever holds file contents (only `FileRef` + on-disk truth) or full command output (only truncated tails). Keeps every checkpoint small on 16 GB.

---

## 6. Tool abstraction layer

```
Tool (protocol):
  name: str
  description: str                 # shown to the LLM
  args_schema: type[BaseModel]     # validated before execution
  requires_approval: bool          # gates command_approval in `semi` mode
  run(args, ctx: ToolContext) -> ToolResult

ToolResult: ok: bool; output: str; error: str | None; meta: dict
ToolContext: workspace_path, git handle, sandbox handle, run_id, budget, tracer
```

**Every call flows through one pipeline:**
`LLM tool-call → schema validate → authorize (path-jail + command allow/deny-list) → [HITL command_approval if requires_approval and autonomy=semi] → execute in sandbox → truncate output → trace → ToolResult back to loop.`

**Tool catalog** (domain-level, provider-agnostic): `read_file`, `edit_file` (patch/search-replace), `write_file`, `list_dir`, `search_code` (ripgrep + tree-sitter symbols), `retrieve` (RAG), `run_command` (sandboxed, `requires_approval=True`), `git_status/diff/add/commit/checkout/branch`, `finish_task`.

The LangChain binding is a thin adapter (`to_langchain_tool(tool)`). Domain tools never import LangChain, so swapping the agent framework touches one file. See [ADR-0007](adr/0007-sandboxed-execution.md).

---

## 7. Provider abstraction

```
LLMProvider (interface):
  chat(messages, tools=None, **params) -> ChatResponse        # text + tool_calls
  structured(messages, schema: type[BaseModel]) -> BaseModel  # validate + repair-retry
  stream(messages, **params) -> Iterator[Chunk]
  embed(texts: list[str]) -> list[Vector]
  capabilities: {supports_tools, supports_json, max_context}
```

- **Factory + per-role config:** `providers.get(role)` where `role ∈ {planner, coder, reviewer, embed}`. Locally all point at one Ollama model; in cloud upgrade `planner` independently — graph untouched.
- **`structured()` is the critical utility:** validates against Pydantic schema; on failure re-prompts with the validation error (×2) before raising. Tames Ollama's flaky JSON/tool support.
- **Ollama adapter specifics:** sets `num_ctx`, `keep_alive` (avoid reloads), `temperature` per role; declares `supports_tools` per model to fall back to JSON-mode prompting when native tool-calling is absent.
- **Swap = config/env only.** No `if provider == ...` in graph/agent/tool code.

See [ADR-0003](adr/0003-provider-abstraction.md) and [ADR-0004](adr/0004-ollama-model-choice.md).

---

## 8. Workspace lifecycle

1. **Create/attach:** `workspaces/<project_id>/`; `git init` for new, or register existing repo. Record `base_commit`.
2. **Run start:** create `work_branch = agent/run-<run_id>` off `base_commit`.
3. **Execute:** Coder mutates files; `git add` continuously; **commit at each verified task boundary**.
4. **Verify/Review:** run inside sandbox mounted RW only on that workspace; Reviewer reads `base_commit..HEAD`.
5. **Finalize:** succeeded → leave branch + final diff artifact for human merge/PR (never auto-merge to main); cancelled/failed → branch retained for inspection.
6. **GC:** background reaper prunes old workspaces/branches by age + keep-list.

Everything is jailed to the workspace dir and executed in the sandbox (Docker `--network=none`, CPU/mem/time limits).

---

## 9. Repository indexing & RAG flow

**Index (on attach + incremental on change):**
```
walk repo (respect .gitignore)
 → tree-sitter parse → symbol table (file, class, func, line)
 → chunk by function/class boundary (not fixed windows)
 → nomic-embed-text (Ollama) → pgvector upsert (namespace=project_id)
 → build BM25 keyword index
incremental: reindex only files whose content-hash changed
```
**Retrieve (hybrid):**
```
query → [BM25 keyword] + [vector search] → Reciprocal-Rank-Fusion → top-k
      → return chunks + symbol locations (NO cross-encoder reranker — avoids a
        second model reloading on a 16 GB box)
```
Consumed by `plan` and `coder`. Keyword + vector fusion matters because code retrieval is dominated by exact symbol names. Retrieved chunks land in `retrieved_context` (ephemeral). See [ADR-0008](adr/0008-hybrid-code-rag.md).

---

## 10. Git workflow

- **Branch per run** (`agent/run-<id>`) off recorded `base_commit`.
- **Commit per verified task**; accumulating diff *is* the review artifact and rollback unit.
- **Reviewer** operates on `git diff base_commit..HEAD`.
- **Accept** → hand a clean branch/PR to the human (never auto-merge to `main`). **Rollback** → `git reset`/`revert`.

This is what makes "iteratively improve an existing project" real rather than aspirational.

---

## 11. Human-in-the-loop flow

- **Mechanism:** node calls `interrupt(HITLRequest)` → LangGraph checkpoints and pauses → API surfaces `hitl_request` over the stream → user responds → `graph.invoke(Command(resume=HITLResponse), config)` continues from the checkpoint.
- **Gates:** clarification (inside `plan`), plan approval (`human_gate`), **command approval** (in-tool, before `run_command` when `autonomy=semi`), escalation (`human_gate`), final acceptance (`human_gate`, only when `autonomy=manual`).
- **Autonomy levels:** `manual` = all gates; `semi` = plan + destructive commands + escalation; `auto` = escalation only.
- Durable: a process crash while paused resumes at the identical interrupt.

See [ADR-0009](adr/0009-hitl-autonomy-levels.md).

---

## 12. UI interaction flow

```
create/attach project → submit task
 → open SSE/WS stream
 → live events: node transitions · streamed tokens · tool calls · verify output · running diff
 → HITL cards appear inline (Approve plan / Approve command / Accept changes / Escalation)
 → user acts → POST /runs/{id}/resume → stream continues
 → terminal: final diff viewer + run summary + Langfuse trace link
```

Phases 0–5: LangGraph Studio + a minimal event stream. Phase 6: custom Next.js "mission control" (graph view, diff viewer, approval cards, timeline). The "team of AI engineers" story is told here by visualizing Planner/Coder/Reviewer transitions.

---

## 13. Data stores

| Store | Role | Local default | Production |
|---|---|---|---|
| Relational | projects, runs, events, artifacts, tasks, reviews | Postgres (Docker) | Managed Postgres |
| Vector | code chunks + memory (pgvector) | pgvector in same Postgres | Managed Postgres+pgvector |
| Checkpointer | LangGraph durable state | SQLite (clone-and-run) → Postgres | Postgres checkpointer |
| Object/diff artifacts | final diffs, run summaries | filesystem | object storage |

See [ADR-0010](adr/0010-postgres-pgvector-checkpointer.md).

---

## 14. Cross-cutting concerns

- **Observability:** structured JSON logging (`structlog`) with a run/trace id threaded through every node and tool; **Langfuse (self-hosted)** for traces, latency, token cost, and eval dashboards (offline-capable). See [ADR-0006-note]. *(Langfuse chosen over LangSmith for offline/self-host.)*
- **Error handling:** LLM transport → tenacity backoff; structured output → repair-retry; tool errors → fed back to agent; graph → recursion cap + budgets + no-progress detection; recovery → checkpoint resume; escalation → HITL after bounded retries.
- **Budgets:** `Budget` caps tokens/steps/wall-clock per run and `max_steps_per_task` per coder loop. Exceeding any cap routes to escalation.
- **Security:** sandboxed exec (Docker `--network=none`, resource + time limits), command allow/deny-list, workspace path jail, no secrets in prompts/state/logs, prompt-injection awareness (requirement text and ingested repo are untrusted input), API auth for cloud. See [ADR-0007](adr/0007-sandboxed-execution.md).

---

## 15. Scalability seams (designed now, exploited later)

- Stateless backend + externalized state (Postgres checkpointer) → horizontal scale.
- Long runs → task queue (Arq/Celery + Redis).
- Per-role model routing for cost/quality.
- `Send`-based parallel Coder fan-out for independent tasks — **dormant locally** (serializes on one Ollama), live in cloud.

---

## 16. Explicitly cut / deferred (anti-scope-creep)

| Item | Verdict |
|---|---|
| Cross-encoder reranker in RAG | **Cut** — second model = reload thrash; RRF is enough |
| Per-task incremental verify | **Deferred** — start with one full `verify` after all tasks |
| Episodic/cross-session memory | **Deferred** to post-Phase-5 (keep `finalize` write-hook as stub) |
| Separate `Spec` object | **Cut** — folded into `Plan` |
| LLM router / supervisor agent | **Cut** — conditional edges are the router |
| Parallel Coder fan-out | **Seam only** — do not build the machinery in early phases |
| Role-based frontend/backend/DB/QA/docs agents | **Cut** — merged into one `coder` + `verify` + `review` |

See [ADR-0001](adr/0001-capability-bounded-agents.md).
