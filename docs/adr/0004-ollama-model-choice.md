# ADR-0004: Single primary local model — `qwen2.5-coder:7b`

**Status:** Accepted

## Context
16 GB RAM, no GPU, CPU inference. A 7–8B model at Q4 (~5–6 GB + KV cache) is comfortable; 14B Q4 (~9 GB) is possible but slow and margin-thin; 32B is impractical. Ollama reloads the model on switch, so running two different chat models thrashes.

## Decision
Use **one primary chat model for all LLM roles locally: `qwen2.5-coder:7b-instruct` (Q4_K_M)**. Use **`nomic-embed-text`** for embeddings (tiny, co-resident, no thrash). Do **not** introduce a second small chat model for routing — prefer rule-based routing or the primary model. Set `num_ctx` deliberately (~8K–16K) and `keep_alive` to avoid reloads.

## Consequences
- No model-reload thrash; predictable RAM.
- Directly satisfies the "one primary local model" constraint.
- Large context is RAM-expensive (KV cache) — which is precisely *why* RAG and context pruning exist.
- Cloud era can differentiate models per role (see ADR-0003) without local penalty.

## Alternatives rejected
- **14B/32B local:** too slow / infeasible on 16 GB.
- **Separate small router model:** causes reload thrash for marginal benefit.
- **Distinct models per role locally:** thrash; no quality gain from one 7B wearing three hats vs. reloading.
