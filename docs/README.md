# Documentation

[← Back to project README](../README.md)

The map of everything. Start with the project [README](../README.md) for the overview and
quickstart; come here to go deep.

## How it works

- **[Runtime flow](runtime-flow.md)** — what happens inside a single run, node by node, with
  the full diagram.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the deep design: the three sources of truth, the
  agent roles, the provider abstraction, and the reasoning behind them.

## What was built, phase by phase

Each doc is the **as-built writeup** — what shipped in that phase and how it was verified. (For
the forward-looking *plan* of each phase, see the paired build plan.)

| Phase | As-built writeup | Plan |
|---|---|---|
| 0 · Foundations | [phase-0-foundations.md](phases/phase-0-foundations.md) | [PHASE-0](build-plans/PHASE-0.md) |
| 1 · Core coder loop | [phase-1-coder-loop.md](phases/phase-1-coder-loop.md) | [PHASE-1](build-plans/PHASE-1.md) |
| 2 · Orchestration + HITL | [phase-2-orchestration.md](phases/phase-2-orchestration.md) | [PHASE-2](build-plans/PHASE-2.md) |
| 3 · RAG + memory | [phase-3-rag-and-memory.md](phases/phase-3-rag-and-memory.md) | [PHASE-3](build-plans/PHASE-3.md) |
| 4 · Review + self-correction | [phase-4-review.md](phases/phase-4-review.md) | [PHASE-4](build-plans/PHASE-4.md) |
| 5 · Eval harness | [phase-5-evals.md](phases/phase-5-evals.md) | [PHASE-5](build-plans/PHASE-5.md) |

## Planning & decisions

- **[Roadmap](build-plans/ROADMAP.md)** — the full phase sequence (0–7) and why this order.
- **[Build plans](build-plans/)** — the detailed spec written *before* each phase was built.
- **[Architecture Decision Records](adr/)** — the load-bearing choices and the alternatives
  rejected ([index](adr/README.md)).
