# Phase 3 — Repository indexing, hybrid RAG, and memory

> **Goal:** Give the working graph real **grounding**. Implement repository indexing (tree-sitter symbol table + structural chunking), hybrid BM25+vector retrieval fused with RRF (`ADR-0008`), the `retrieve` tool, and the memory tiers (`ARCHITECTURE.md §"Memory"`), so `plan` and `coder` operate on retrieved real code and the system can meaningfully edit an *existing* repository.
>
> **Why this phase comes fourth (after the graph):** RAG is only meaningful once a working executor can act on retrieved context. Phase 2's graph runs on ripgrep-level grounding; Phase 3 upgrades grounding without changing topology or node contracts. This ordering is what makes "iteratively improve an existing project" real (ties to git workflow, ADR-0002).

## Objectives
1. Repository indexer: walk repo (respect `.gitignore`), tree-sitter parse → symbol table, chunk by function/class boundary.
2. Embeddings via `nomic-embed-text` (ADR-0004) → pgvector upsert, namespaced per `project_id`.
3. BM25 keyword index over the same chunks.
4. Hybrid retriever: BM25 + vector fused via Reciprocal-Rank-Fusion; **no cross-encoder reranker** (ADR-0008).
5. `retrieve` tool + upgraded `search_code` backed by the symbol index (replacing/augmenting Phase-1 ripgrep).
6. Incremental reindex by content hash on workspace changes.
7. Memory tiers: **working** (already state+checkpointer, Phase 2), **semantic/long-term** (pgvector, namespaced), **episodic** (Postgres relational + traces). Turn the Phase-2 `finalize` memory hook into real writes.
8. Feed `retrieved_context` into `plan` and `coder`; keep it ephemeral in state.

## Scope
**In:** `rag/indexer.py` (walk + tree-sitter + chunker), `rag/embeddings.py` (nomic-embed via provider), `rag/vector_store.py` (pgvector upsert/query, per-project namespace), `rag/keyword_index.py` (BM25), `rag/retriever.py` (hybrid + RRF), `tools/retrieve.py`, symbol-backed `search_code`, incremental reindex trigger on git changes, `memory/long_term.py` (semantic store: ADRs/decisions/conventions), `memory/episodic.py` (run outcomes to Postgres), wiring `retrieve` into `plan`/`coder` prompts, populating `retrieved_context`.
**Out (later phases):** cross-encoder reranker (**cut**, ADR-0008), cross-session behavioral learning beyond stored facts (deferred, ADR/§16), the reviewer's use of retrieval beyond read-only context (reviewer arrives Phase 4; it already may call `retrieve`), UI surfacing of retrieval (Phase 6), distributed/managed vector store (Phase 7 — pgvector local now, ADR-0010).

## Prerequisites
- Phase 2 complete: working graph, state, checkpointer, HITL, `finalize` memory-hook stub.
- Phase 0 infra: Postgres + pgvector healthy; `nomic-embed-text` pulled.
- tree-sitter grammars available for target languages (at minimum Python + TS/JS to match the workspace targets).

## Work breakdown & deliverables
| # | Task | Deliverable |
|---|---|---|
| 3.1 | `rag/chunker.py` — tree-sitter parse → symbol table (file/class/func/line) + chunks on function/class boundaries | Structural chunks + symbol index |
| 3.2 | `rag/embeddings.py` — batch embed chunks via `nomic-embed-text` through the Phase-0 provider `embed()` | Deterministic embedding fn |
| 3.3 | `rag/vector_store.py` — pgvector schema + upsert/query, **namespace = project_id** | Vector store adapter |
| 3.4 | `rag/keyword_index.py` — BM25 over chunks (per project) | Keyword index |
| 3.5 | `rag/retriever.py` — hybrid: run BM25 + vector, fuse via RRF, return top-k chunks + symbol locations | `retrieve(query, k)` core |
| 3.6 | `rag/indexer.py` — full index on attach: walk (`.gitignore`-aware) → chunk → embed → upsert → build BM25 | `index_project(project_id)` |
| 3.7 | Incremental reindex — content-hash per file; reindex only changed files on git change | `reindex_changed()` |
| 3.8 | `tools/retrieve.py` — `retrieve` tool through the Phase-1 authorization/trace pipeline (read-only) | LLM-callable retrieval |
| 3.9 | Symbol-backed `search_code` — augment Phase-1 ripgrep with symbol lookups | Upgraded search tool |
| 3.10 | `memory/long_term.py` — semantic store for decisions/conventions (namespaced pgvector), read + write API | Long-term memory |
| 3.11 | `memory/episodic.py` — write run outcomes (what changed, verdicts) to Postgres; realize the `finalize` hook | Episodic memory writes |
| 3.12 | Wire `retrieved_context` — `plan` and `coder` request retrieval, populate ephemeral `retrieved_context`, prune after use | Grounded planning/coding |
| 3.13 | Retrieval-quality harness — fixed queries over a known repo with expected symbols/files | Precision@k measurement |

## Testing strategy
- **Chunker tests:** functions/classes chunked intact (no mid-function splits); symbol table maps names → correct file/line; `.gitignore` respected.
- **Embedding tests:** stable dimensionality; batch == single-call parity; empty/binary files skipped gracefully.
- **Vector store tests:** upsert/query round-trip; **namespace isolation** — project A's chunks never returned for project B.
- **Hybrid retrieval tests:** exact-symbol query → keyword arm surfaces it (case vector-only would miss); semantic paraphrase query → vector arm surfaces it; RRF fusion orders sensibly. Assert **no reranker** in the path.
- **Retrieval-quality (precision@k):** on a known repo, a fixed query set hits expected files/symbols above a baseline threshold; tracked as a regression metric (feeds Phase-5 evals).
- **Incremental reindex tests:** change one file → only that file's chunks re-embedded (hash-gated); deleted file's chunks removed.
- **Grounding integration test:** `coder` editing an *existing* repo uses `retrieved_context` to reference real symbols (no hallucinated imports); compare against a ripgrep-only baseline run.
- **Memory tests:** long-term write→read round-trip (namespaced); episodic run record persisted at `finalize`; retrieved chunks remain ephemeral (not persisted in checkpoint state — invariant re-checked).

## Definition of Done
- `index_project` fully indexes a real repo (Python + TS/JS) into pgvector + BM25, namespaced per project.
- Hybrid retriever returns fused BM25+vector results (RRF, no reranker) with symbol locations; precision@k meets the baseline on the known-repo query set.
- Incremental reindex re-embeds only changed files.
- `retrieve` tool and symbol-backed `search_code` are callable by `plan`/`coder`, populating ephemeral `retrieved_context`.
- The graph edits an existing repository grounded in retrieved code (integration test beats the ripgrep-only baseline on hallucinated-reference rate).
- Long-term (semantic) and episodic memory writes work; `finalize` hook is real, not a stub.
- State invariant still holds (retrieved chunks ephemeral; no contents persisted). Lint/type-check/tests green; retrieval traced in Langfuse.

## Risks & mitigations
- **Embedding throughput on CPU / 16 GB** → batch embeddings; index off the run's critical path (on attach, not per step); cache by content hash so reindex is incremental; `nomic-embed-text` co-resides without model-reload thrash (ADR-0004).
- **tree-sitter grammar gaps / parse failures** → graceful fallback to line-window chunking for unsupported languages, logged; never fail the run on a parse error.
- **Retrieval pollution blowing the context budget** → strict top-k, dedup by path/symbol, truncate chunk size; prune `retrieved_context` immediately after the node consumes it.
- **pgvector recall/index tuning** → start with a sane index (e.g. IVFFlat/HNSW default) and the precision@k harness as the tuning signal; do not add a reranker (ADR-0008) — tune fusion weights instead.
- **Namespace leakage across projects** → enforce `project_id` filter at the store layer; explicit isolation test.
- **Stale index vs workspace drift** → content-hash gating + reindex-on-git-change; document that verify/review always read live files, not the index.

## What the next phase builds on this
Phase 4 (adversarial reviewer + iteration) replaces the Phase-2 `review` stub with the real fresh-context reviewer (ADR-0006), which consumes the `git diff` and may call the now-real `retrieve`/`search_code` for surrounding context — without changing graph topology, state, or the retrieval/memory contracts settled here. Phase 5 (eval harness) reuses the Phase-3 retrieval-quality harness as one of its regression metrics.
