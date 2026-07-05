# Architecture Decision Records

Each ADR captures one significant decision: context, the decision, consequences, and the alternatives rejected. Format is a lightweight MADR.

| # | Decision | Status |
|---|---|---|
| [0001](0001-capability-bounded-agents.md) | Capability-bounded agents, not role-based multi-agent | Accepted |
| [0002](0002-three-sources-of-truth.md) | Three separated sources of truth; no file contents in state | Accepted |
| [0003](0003-provider-abstraction.md) | Config-driven provider abstraction with per-role model config | Accepted |
| [0004](0004-ollama-model-choice.md) | Single primary local model: `qwen2.5-coder:7b` | Accepted |
| [0005](0005-deterministic-verify.md) | Deterministic (non-LLM) verify node closing the loop | Accepted |
| [0006](0006-fresh-context-reviewer.md) | Adversarial reviewer with isolated context | Accepted |
| [0007](0007-sandboxed-execution.md) | Sandboxed command execution + tool authorization pipeline | Accepted |
| [0008](0008-hybrid-code-rag.md) | Hybrid (BM25 + vector) code RAG, no reranker | Accepted |
| [0009](0009-hitl-autonomy-levels.md) | HITL via `interrupt` with three autonomy levels | Accepted |
| [0010](0010-postgres-pgvector-checkpointer.md) | Postgres + pgvector + LangGraph checkpointer (SQLite dev) | Accepted |
