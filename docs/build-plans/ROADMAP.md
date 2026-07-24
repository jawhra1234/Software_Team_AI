# Roadmap

Sequenced to kill risk early and minimize technical debt: build the riskiest core (tool + verify loop) first, structure it with LangGraph second, then add retrieval, review, evals, and UI. Never build UI or parallelism on an unproven core.

| Phase | Objective | Detailed plan | Status |
|---|---|---|---|
| **0 — Foundations** | Config, provider abstraction, `structured_call`, logging/tracing, local infra | [PHASE-0.md](PHASE-0.md) | Specced |
| **1 — Core Coder loop** ⭐ | Tools + sandbox + git workspace + coder ReAct loop + deterministic verify | [PHASE-1.md](PHASE-1.md) | Specced |
| **2 — LangGraph orchestration + HITL** | `plan→human_gate→coder→verify→review→finalize` graph, Postgres checkpointer, plan-approval interrupt, State schema | [PHASE-2.md](PHASE-2.md) | Specced |
| **3 — Grounding: RAG + memory** | tree-sitter symbol index, hybrid BM25+vector retrieval, `retrieve` tool, long-term memory | [PHASE-3.md](PHASE-3.md) | Specced |
| **4 — Autonomous Review & Self-Correction** | Fresh-context reviewer (diff+plan+verify only, read-only grounding), targeted fix-loop bounded + escalating | [PHASE-4.md](PHASE-4.md) | Specced |
| **5 — Eval harness** ⭐ | Internal SWE-bench-lite-style task suite, metrics, Langfuse dashboards, regression tracking | _to be written_ | Planned |
| **6 — Mission-control UI** | Next.js: live graph, streaming, diff viewer, HITL cards, timeline | _to be written_ | Planned |
| **7 — Cloud & scale** | Hosted provider swap, task queue, auth, deploy, horizontal-ready | _to be written_ | Planned |

⭐ = the two phases most portfolio projects skip and that most impress senior interviewers (closed-loop verification and measurable evals).

### Sequencing rationale (why this order)
- **0 before everything:** the provider abstraction + `structured_call` are touched by every node; settle them before the graph exists.
- **1 before 2:** prove the tool/sandbox/verify core in isolation, so Phase 2 debugs *orchestration* only, not orchestration + core at once.
- **3 after 1–2:** RAG is only meaningful once a working executor can act on retrieved context.
- **4 after 3:** the reviewer needs a real diff and grounding to critique.
- **5 after 4:** you can't measure quality until the full loop exists; evals then gate every later change.
- **6 after 5:** build UI on a stable, measured core.
- **7 last:** cloud/scale seams were pre-cut throughout; exploit them only once the product is proven.
