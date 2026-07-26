# Documentation

[← Back to project README](../README.md)

The central map. Start with the project [README](../README.md) for the overview, evidence, and
quickstart; come here to go deep.

> **Two kinds of phase docs — don't conflate them:**
> - **`docs/phases/`** — the **as-built** record: what actually shipped in each phase and how it
>   was verified (incl. honest limitations).
> - **`docs/build-plans/`** — the **original forward-looking specifications**, written *before*
>   each phase was built.
>
> Each as-built doc links to its plan, and vice-versa.

## How it works

| Doc | What it covers |
|---|---|
| **[Runtime flow](runtime-flow.md)** | what happens inside a single run — the full node-by-node diagram |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | the deep design: three sources of truth, agent roles, provider abstraction, and the reasoning |

## What was built, phase by phase (as-built)

Each doc follows the same structure — *what it adds · why · architecture · implementation ·
configuration · testing & validation · live validation · what worked · known limitations · key
decisions · status*.

| Phase | As-built writeup | Original plan (spec) |
|---|---|---|
| 0 · Foundations | [phase-0-foundations.md](phases/phase-0-foundations.md) | [PHASE-0](build-plans/PHASE-0.md) |
| 1 · Core coder loop | [phase-1-coder-loop.md](phases/phase-1-coder-loop.md) | [PHASE-1](build-plans/PHASE-1.md) |
| 2 · Orchestration + HITL | [phase-2-orchestration.md](phases/phase-2-orchestration.md) | [PHASE-2](build-plans/PHASE-2.md) |
| 3 · RAG + memory | [phase-3-rag-and-memory.md](phases/phase-3-rag-and-memory.md) | [PHASE-3](build-plans/PHASE-3.md) |
| 4 · Review + self-correction | [phase-4-review.md](phases/phase-4-review.md) | [PHASE-4](build-plans/PHASE-4.md) |
| 5 · Eval harness | [phase-5-evals.md](phases/phase-5-evals.md) | [PHASE-5](build-plans/PHASE-5.md) |

## Planning & decisions

| Doc | What it covers |
|---|---|
| **[Roadmap](build-plans/ROADMAP.md)** | the full phase sequence (0–7) and why this order |
| **[Build plans](build-plans/)** | the detailed spec written *before* each phase was built |
| **[ADRs](adr/)** ([index](adr/README.md)) | the load-bearing decisions and the alternatives rejected |
