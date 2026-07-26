# Phase 3 — Repository indexing, hybrid RAG, and memory

[← Back to README](../../README.md) · [Docs index](../README.md) · [Build plan](../build-plans/PHASE-3.md)

> Real grounding: the agents read your *actual* code (hybrid retrieval) and remember decisions + past runs — instead of guessing. Added **without touching the graph's shape**.

## What was built

RAG and memory are wired as **opt-in dependencies** (default `None`), so the hermetic test
suite stays fast and independent of Postgres.

- **Structural chunker** (`app/rag/chunker.py`) — tree-sitter splits source into symbol-aware
  chunks (functions/classes with their signatures), not blind line windows, so a retrieved
  chunk is a whole, meaningful unit.
- **Embeddings** (`app/rag/embeddings.py`) — `nomic-embed-text` (768-dim) via the provider
  abstraction, cached by chunk `content_hash` so re-indexing an unchanged chunk costs nothing.
  `nomic` is asymmetric, so documents and queries get their `search_document:` /
  `search_query:` task prefixes (measurably tighter margins — see below).
- **Vector store** (`app/rag/vector_store.py`) — pgvector cosine similarity, namespaced per
  project, with a symbol-name lookup for exact-symbol queries.
- **Keyword index** (`app/rag/keyword_index.py`) — in-memory BM25Plus; exact symbol names
  dominate code queries, where pure vectors are weak.
- **Hybrid retriever** (`app/rag/retriever.py`) — runs the vector and BM25 arms independently
  and fuses them with **Reciprocal Rank Fusion** — *no cross-encoder reranker*
  ([ADR-0008](../adr/0008-hybrid-code-rag.md)); RRF combines both without tuning score scales.
- **Incremental indexer** (`app/rag/indexer.py`) — content-hash diffing so a reindex only
  touches changed files; drives the `retrieve` tool and the graph's `RetrievalCapture`.
- **`retrieve` tool** (`app/tools/retrieve.py`) — exposes hybrid retrieval to the agents; the
  planner and coder call it as their **first grounding step**, and results are captured into
  `retrieved_context` on graph state (`app/graph/retrieval.py`).
- **Long-term memory** (`app/memory/long_term.py`) — a namespaced pgvector store of durable
  facts (decisions, learned repo conventions), retrieved by **semantic** similarity, kept
  separate from the churny code index. Populated by a deterministic writer
  (`app/memory/ingest.py` + `scripts/seed_memory.py`) that ingests the repo's own ADRs as
  `decision` memories, so the store has real content on day one.
- **Episodic memory** (`app/memory/episodic.py`) — run outcomes written at `finalize`.
  Retrieval isn't blind recency: `relevant()` re-ranks a recent window by **lexical** overlap
  with the request plus a failure bonus (past failures are the useful lesson), since summaries
  are mostly filenames/symbols/errors.
- **Memory in the planner** (`app/graph/planning_context.py`) — before planning, the plan node
  searches long-term (semantic) + episodic (lexical) and injects two labeled sections —
  **Project Conventions** and **Previous Attempts** — into the planner's prompt, each shown
  only when non-empty. Repository code is **not** preloaded here — it stays on-demand via the
  `retrieve` tool, so tokens stay bounded. All reads are best-effort (a memory outage degrades
  a section, never fails the run).
- **File-size guard** (`app/tools/fs.py`) — a 1 MiB cap on `write_file` / `edit_file`, added
  after live validation surfaced a runaway-edit failure mode (below).

## How it was verified

**Retrieval works on the real codebase:** indexing this repo (103 files → 773 chunks) and
querying it, hybrid retrieval returns the right symbols for both **exact** symbol queries and
**semantic paraphrases** that share no keywords with the target — with the `nomic` task
prefixes improving the query↔target cosine margin (e.g. `+0.580 → +0.614` on the fixture).

### Does RAG actually change agent behavior? (RAG OFF vs RAG ON)

Components passing tests isn't proof that retrieval *helps the agent*. So the whole pipeline
was run live against a controlled fixture: a checkout task whose correct answer depends on a
**non-guessable helper** (`apply_levy`, a bespoke surcharge/rounding rule) that lives in the
indexed repo but is **never named in the task** — solvable only if the agent *discovers* it.

| | **RAG OFF** | **RAG ON** |
|---|---|---|
| `retrieve` | `ok: false` (no index, by design) | **`ok: true` — called repeatedly** |
| `retrieved_context` | **0 chunks** | **10 chunks** — incl. `apply_levy` |
| Found the hidden helper? | **No** | **Yes** — wrote `from pricing_rules import apply_levy` / `return apply_levy(...)` |
| Node timeline | plan → escalate → finalize | plan `[retrieved 5]` → coder → retrieve → coder `[retrieved 10]` → … |

**Conclusions:**

1. **RAG demonstrably changes behavior.** With retrieval on, the coder grounds on the real
   helper and writes the correct, reuse-based solution; with it off, it never finds the helper
   and dead-ends — shown end-to-end through the actual graph, not just in unit tests.
2. **The local 7B is the ceiling, not the design.** On this hard task both arms ended `failed`
   for **model-quality** reasons (invalid task `kind`; the coder corrupting a file by misusing
   `edit_file`), and the harness caught every one cleanly instead of hanging. Both quirks are
   now **hardened**.
3. **Real bugs found and fixed** — hardening, not benchmark-tuning: a 1 MiB `write_file` /
   `edit_file` cap (was: a file ballooning to hundreds of MB); a tolerant `Task.kind` coercion
   (off-enum value → safe default, not a failed Plan); isolated test memory tables (a 2-dim
   test embedder can't clash with the real 768-dim store).

The provider abstraction means a fully-green run is a **config-only swap** to a stronger model;
the local 7B's stumbles double as a live demonstration that the guardrails work under a weak one.

### Does past experience change future planning? (memory across runs)

A second harness (`scripts/memory_e2e.py`) runs the **same project twice**:

- **Run 1** plans with an empty episodic store → only **Project Conventions** is injected. At
  `finalize`, the run's outcome is written to episodic memory.
- **Run 2** (a related request) retrieves that record: its planner prompt now carries **both**
  sections — Project Conventions *and* a **Previous Attempts** entry naming run 1 — proving a
  prior run measurably shapes later planning. Repository `retrieve` still fires independently.

**Full green happy path** (`scripts/memory_e2e.py simple`): on a trivial task, the whole
pipeline runs clean — `plan → coder → verify PASS → review approve → finalize succeeded` — with
memory wired in. Same architecture as the hard task; only task difficulty decides the outcome.

**Validation harnesses:** `scripts/rag_validate.py` (`part1` retrieval, `abtest` RAG off/on)
and `scripts/memory_e2e.py` (cross-run memory; `simple` for the green happy path).

## Key decisions

- [ADR-0008 — Hybrid (BM25 + vector) code RAG, no reranker](../adr/0008-hybrid-code-rag.md)
- [ADR-0004 — `nomic-embed-text` for embeddings](../adr/0004-ollama-model-choice.md)
- [ADR-0010 — Postgres + pgvector](../adr/0010-postgres-pgvector-checkpointer.md)

---

[← Phase 2 — Orchestration](phase-2-orchestration.md) · Next: [Phase 4 — Review →](phase-4-review.md)
