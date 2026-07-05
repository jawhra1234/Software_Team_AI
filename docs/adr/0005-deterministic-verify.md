# ADR-0005: Deterministic (non-LLM) verify node closing the loop

**Status:** Accepted

## Context
The single feature separating a coding *tool* from a coding *demo* is running generated code and feeding failures back. An LLM "QA agent" that only reasons about tests is unreliable and expensive; ground truth comes from actually executing tests/build/lint.

## Decision
`verify` is a **deterministic node with no LLM call**. It auto-detects and runs the project's test/build/lint/typecheck commands in the sandbox and returns a structured `VerifyResult` (per-check pass/fail, exit codes, truncated output tails). Failures route back to `coder` in fix mode (bounded retries), then to `human_gate[escalation]`.

## Consequences
- Objective, cheap, reproducible signal.
- The feedback loop drives agentic self-correction.
- Timeouts count as failures, guarding infinite loops.
- Command detection is rule-based and per-project configurable.

## Alternatives rejected
- **LLM QA agent:** non-deterministic, costly, hallucinates pass/fail.
- **No verification (trust the coder):** the demo trap; unacceptable.
