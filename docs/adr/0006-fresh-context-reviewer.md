# ADR-0006: Adversarial reviewer with isolated context

**Status:** Accepted

## Context
An author reviewing their own work misses errors because their context is anchored to their reasoning. Human teams solve this with independent code review. This is a genuine case where a separate agent (context isolation) earns its keep.

## Decision
`review` is a separate LLM role that sees **only the diff (`base_commit..HEAD`) + requirements + verify result** — never `coder_scratch`. It assigns severities (`blocker/major/minor/nit`) and returns `changes_requested` only when a `blocker`/`major` exists (no nitpick-blocking). Review→fix cycles are capped (default 2) to prevent reviewer/coder ping-pong.

## Consequences
- Catches failure modes author-context misses.
- Cap prevents infinite review loops.
- Deliberately excluded from reading coder scratch — the isolation is the point.

## Note on observability
Traces/latency/token cost/eval dashboards use **Langfuse (self-hosted)** rather than LangSmith, chosen for offline/local operation and self-hosting as a demonstrable skill.

## Alternatives rejected
- **Coder reviews itself:** defeats the purpose; shared context hides bugs.
- **Reviewer sees full coder reasoning:** reintroduces the anchoring bias.
