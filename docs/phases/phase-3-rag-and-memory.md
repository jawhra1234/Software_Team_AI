# Phase 3 — Repository indexing, hybrid RAG, and memory

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 2 — Orchestration](phase-2-orchestration.md) · [Next: Phase 4 — Review →](phase-4-review.md)

> *This is an **as-built** writeup. For the original forward-looking specification, see the [Phase 3 build plan](../build-plans/PHASE-3.md).*

Real grounding: the agents read your *actual* code (hybrid retrieval) and remember decisions +
past runs — instead of guessing. Added **without touching the graph's shape**.

## What this phase adds

A tree-sitter code index, hybrid (vector + BM25 → RRF) retrieval, a `retrieve` tool, long-term
(semantic) and episodic (relational) memory, and memory injection into the planner.

## Why it was needed

Phase 2 grounded planning/coding with ripgrep only — fine for greenfield, weak for editing a real
repository. To reuse existing helpers instead of reinventing them (and to learn across runs), the
agents need real retrieval and memory. This is the seam that makes "iteratively improve an
existing project" real.

## Architecture / how it works

- **Hybrid retrieval** ([ADR-0008](../adr/0008-hybrid-code-rag.md)): tree-sitter chunks source on
  function/class boundaries; `nomic-embed-text` (768-dim) embeds them into pgvector; a BM25Plus
  keyword index runs alongside; the two arms are fused with **Reciprocal Rank Fusion — no
  cross-encoder reranker**. Keyword matters because exact symbol names dominate code queries;
  vectors matter for paraphrase.
- **Opt-in dependencies:** RAG and memory default to `None` in `build_graph`, so hermetic tests
  stay fast and Postgres-free; live runs build the stack explicitly.
- **Memory, two tiers:** *long-term* is a namespaced pgvector store of durable facts (retrieved
  by **semantic** similarity), populated deterministically from the repo's own ADRs; *episodic*
  is a relational log of run outcomes, retrieved by **lexical** relevance (overlap + a failure
  bonus, since past failures are the useful lesson).
- **Memory in the planner:** before planning, the plan node injects two labeled sections —
  **Project Conventions** (long-term) and **Previous Attempts** (episodic) — each only when
  non-empty. Repository code is **not** preloaded; it stays on-demand via `retrieve`, so tokens
  stay bounded. All reads are best-effort (a memory outage degrades a section, never fails a run).

## Implementation

- `app/rag/` — `chunker.py`, `embeddings.py` (content-hash cached, `search_document:`/`search_query:`
  prefixes), `vector_store.py` (pgvector), `keyword_index.py` (BM25Plus), `retriever.py` (RRF),
  `indexer.py` (incremental), `evaluation.py` (precision@k — reused by Phase 5).
- `app/tools/retrieve.py` + `app/graph/retrieval.py` — the `retrieve` tool + capture into state.
- `app/memory/long_term.py`, `episodic.py`, `ingest.py` (ADR writer); `scripts/seed_memory.py`.
- `app/graph/planning_context.py` — assembles the injected memory sections.
- `app/tools/fs.py` — 1 MiB `write_file`/`edit_file` size guard (added after live validation).

## Configuration

```bash
RAG__TOP_K=6                            # chunks returned per retrieve() call
RAG__CANDIDATE_K=20                     # per-arm pool before RRF fusion
RAG__RRF_K=60                           # RRF constant
PLANNER__MEMORY_LONG_TERM_K=5           # conventions/decisions injected into planning
PLANNER__MEMORY_EPISODIC_K=3            # past runs surfaced as "Previous Attempts"
PLANNER__MEMORY_MAX_SECTION_CHARS=1200  # per-section cap (bounds injected tokens)
```

## Testing and validation

- **Hermetic:** chunker (no mid-function splits, symbol table), embeddings (batch==single,
  prefixing), vector store (namespace isolation), hybrid retrieval (exact + paraphrase, no
  reranker in the path), incremental reindex, and the memory read/write logic.
- **Integration (real pgvector):** vector round-trips, retriever precision@k, memory stores.

## Live validation

Two live harnesses, run against the real model:

**Does RAG change agent behavior? (`scripts/rag_validate.py abtest`)** — a checkout task whose
answer depends on a **non-guessable helper** (`apply_levy`) that lives in the indexed repo but is
never named in the task.

| | RAG OFF | RAG ON |
|---|---|---|
| `retrieve` | `ok: false` (no index) | **`ok: true` — repeatedly** |
| `retrieved_context` | **0 chunks** | **10 chunks** (incl. `apply_levy`) |
| Found the hidden helper? | **No** | **Yes** — wrote `from pricing_rules import apply_levy` |

**Does past experience change future planning? (`scripts/memory_e2e.py`)** — the same project run
twice: run 1 writes an episodic record at finalize; run 2's planner prompt then carries a
**Previous Attempts** entry naming run 1. `scripts/memory_e2e.py simple` shows the full green happy
path (`plan → coder → verify PASS → review approve → finalize succeeded`) on a trivial task.

## What worked

RAG demonstrably changes behavior (the coder reused the hidden helper only with retrieval on), and
memory measurably carries across runs (run 1 shaped run 2's plan) — both shown end-to-end through
the real graph, not just in unit tests.

## Known limitations / honest findings

- **The local 7B is the ceiling, not the design.** On the *hard* task both RAG arms ended `failed`
  for model-quality reasons (invalid task `kind`; the coder corrupting a file via `edit_file`
  misuse) — the harness caught every one cleanly, and the *same* pipeline runs fully green on a
  task within the model's reach.
- **Real bugs found and fixed** (hardening, not benchmark-tuning): a 1 MiB fs size guard (was: a
  file ballooning to hundreds of MB); a tolerant `Task.kind` coercion (off-enum → safe default);
  isolated test memory tables (a 2-dim test embedder can't clash with the real 768-dim store).
- **Long-term memory has a writer (ADRs) but no automatic run-time distiller** — writing learned
  facts from each run is deferred to a later phase.

## Key engineering decisions

- [ADR-0008 — Hybrid (BM25 + vector) code RAG, no reranker](../adr/0008-hybrid-code-rag.md)
- [ADR-0004 — `nomic-embed-text` for embeddings](../adr/0004-ollama-model-choice.md)
- [ADR-0010 — Postgres + pgvector](../adr/0010-postgres-pgvector-checkpointer.md)

## Current status

✅ **Complete and verified.**

---

**Navigation:** [← Documentation Hub](../README.md) · [← Previous: Phase 2 — Orchestration](phase-2-orchestration.md) · [Next: Phase 4 — Review →](phase-4-review.md)
