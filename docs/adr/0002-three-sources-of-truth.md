# ADR-0002: Three separated sources of truth; no file contents in state

**Status:** Accepted

## Context
Naively storing file contents and full command output in LangGraph state bloats every checkpoint, explodes token cost, and is unworkable on 16 GB. Agents also need a reliable way to communicate without a lossy chat channel.

## Decision
Separate three sources of truth:
1. **LangGraph state (checkpointed):** control flow + small structured artifacts (plan, task list, statuses, verdicts). Never file contents; never full command output (truncated tails only).
2. **Git-backed workspace on disk:** the actual code — the ground truth for the `reviewer` diff and for rollback.
3. **Vector store + Postgres:** retrievable knowledge (repo chunks, decisions, memory).

Agents communicate through **state (control)** and the **filesystem (code)**. No agent-to-agent messaging, no event bus.

## Consequences
- Small, cheap, resumable checkpoints.
- Git diff is the natural review + rollback unit.
- Reproducible runs; clean seam for horizontal scaling (externalized state).
- Enforced by a state invariant: only `FileRef` (path + status + blob sha) in state.

## Alternatives rejected
- **Contents-in-state:** checkpoint bloat, token blowup, infeasible locally.
- **Event bus / message passing between agents:** lossy, harder to reason about, unnecessary at this scale.
