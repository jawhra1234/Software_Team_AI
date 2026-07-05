# ADR-0008: Hybrid (BM25 + vector) code RAG, no reranker

**Status:** Accepted

## Context
The repo is larger than the context window, so grounding requires retrieval. Code retrieval differs from document retrieval: exact symbol names dominate, and fixed-size chunking destroys structure. A cross-encoder reranker would be a second model competing for RAM on a 16 GB box.

## Decision
Index by parsing with tree-sitter into a symbol table and **chunking on function/class boundaries**; embed with `nomic-embed-text`; upsert to pgvector (namespace per project); build a BM25 keyword index. Retrieve with **hybrid BM25 + vector fused via Reciprocal-Rank-Fusion**. **No cross-encoder reranker.** Reindex incrementally by content hash.

## Consequences
- Keyword arm nails exact symbol lookups; vector arm handles semantic queries.
- No second model → no reload thrash.
- Structural chunks keep functions/classes intact.
- Retrieved chunks are ephemeral in state (`retrieved_context`).

## Alternatives rejected
- **Vector-only:** misses exact symbol matches common in code.
- **Fixed-window chunking:** shreds semantic units.
- **Cross-encoder reranker:** marginal gain, real RAM cost on 16 GB.
