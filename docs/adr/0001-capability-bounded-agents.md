# ADR-0001: Capability-bounded agents, not role-based multi-agent

**Status:** Accepted

## Context
The initial vision proposed a "team" of role-based agents (PM, architect, frontend, backend, DB, QA, docs). On a 16 GB laptop running one Ollama model, every "agent" is the same model in a different costume, invoked sequentially. Role-based agent orgs (ChatDev/MetaGPT-style) demo well but perform poorly and compound errors across handoffs. The hard problems in coding agents are context management, grounding, verification, and safe execution — not role division.

## Decision
An agent is justified **only** by one of: (a) a distinct toolset, (b) an isolated context window, (c) genuine parallelism, or (d) a different model. Under that test we keep exactly three LLM roles — `planner`, `coder`, `reviewer` — plus a deterministic `verify` node and a mostly-deterministic orchestrator (the graph itself). Frontend/backend/DB/QA/docs collapse into the single `coder` loop (prompt/context differences, not architecture). The "team of AI engineers" survives as a **UI narrative only**.

## Consequences
- Far fewer lossy handoffs; lower latency and token cost; honest on local hardware.
- Strong interview signal: demonstrates judgment about *when not* to use multi-agent.
- `reviewer` and `planner` remain separate because context isolation genuinely earns its keep.
- Parallel `coder` fan-out is designed as a seam but dormant locally.

## Alternatives rejected
- **Role-agent pipeline (original vision):** error compounding, fake specialists, unusable parallelism locally.
- **Single monolithic loop (Aider-style):** robust but thin on the LangGraph/multi-agent/HITL story the project must showcase.
