# Runtime flow — what happens inside one run

[← Back to README](../README.md) · [Docs index](README.md)

This is the compiled graph as it stands today: the [Phase-2](phases/phase-2-orchestration.md)
six-node skeleton, grounded in real code + memory ([Phase 3](phases/phase-3-rag-and-memory.md)),
with a real fresh-context reviewer closing the loop ([Phase 4](phases/phase-4-review.md)).
[Phase 5](phases/phase-5-evals.md) (the eval harness) isn't a node here — it *wraps* this whole
flow, running it over a fixed task suite and scoring the results.

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

## How to read the diagram, by phase

- **[Phase 2](phases/phase-2-orchestration.md)** built the six-node skeleton (`plan ·
  human_gate · coder · verify · review · finalize`) and the escalation-to-human path.
- **[Phase 3](phases/phase-3-rag-and-memory.md)** added the **shared knowledge** band on top:
  `PLAN` and `CODER` read real code on demand via `retrieve`; `PLAN` also gets durable
  **memory** injected before it drafts; `FINALIZE` writes the run's outcome back to episodic
  memory so future runs learn from it.
- **[Phase 4](phases/phase-4-review.md)** turned `REVIEW` from a rule-of-thumb stub into a
  real, isolated LLM reviewer driving the targeted `blocker`/`major` → fix cycle.
- **[Phase 5](phases/phase-5-evals.md)** doesn't appear here — it runs this entire flow over a
  task suite and scores it.

## The nodes

- **`plan`** — before drafting, it's automatically handed relevant **memory** (durable
  decisions/conventions + how earlier runs on this project went); it then grounds in the real
  repo (hybrid RAG via `retrieve` + read-only tools), drafts a task list, and only pauses to
  ask the human a question when it's genuinely blocking.
- **`human_gate`** — the *one* place a human is asked anything: approve the plan, resolve an
  escalation, or sign off on the final result. What it asks depends on the **autonomy level**
  (`auto` / `semi` / `manual`).
- **`coder`** — does the actual work, one task at a time, inside a sandbox, grounding in real
  code on demand via `retrieve`. If verify or review returns a problem, it re-enters in "fix
  mode" targeting that specific feedback instead of redoing everything.
- **`verify`** — no LLM involved. Runs the project's real tests/build and reports pass/fail.
- **`review`** — a real, fresh-context reviewer with an **isolated** view: only the approved
  plan, the diff, and the verify result — never the coder's reasoning. It grounds read-only,
  assigns each finding a severity, and only `blocker`/`major` findings send the run back to the
  coder with a **targeted** fix task (not a re-plan); `minor`/`nit` are advisory and never block.
- **`finalize`** — closes the run out with a diff summary and a final status, and records the
  run's outcome to episodic memory.

---

For the deeper design rationale, see [`ARCHITECTURE.md`](ARCHITECTURE.md) and the
[decision records](adr/).
