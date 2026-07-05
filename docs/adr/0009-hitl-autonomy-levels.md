# ADR-0009: HITL via `interrupt` with three autonomy levels

**Status:** Accepted

## Context
The system needs human oversight at the right moments without forcing approval on every step. Oversight must survive process crashes.

## Decision
Human-in-the-loop uses LangGraph `interrupt()`, which checkpoints and pauses; the API surfaces the request and resumes via `Command(resume=...)`. A single multiplexed `human_gate` node handles `plan_approval`, `escalation`, and `final_accept`; `command_approval` is an in-tool interrupt before `run_command`. Three **autonomy levels** decide which gates are live: `manual` (all), `semi` (plan + destructive commands + escalation), `auto` (escalation only).

## Consequences
- Durable: a crash while paused resumes at the identical interrupt.
- One reusable node instead of several gate nodes.
- Autonomy levels are a small, mature product touch demonstrable in interviews.

## Alternatives rejected
- **Approve-everything:** unusable friction.
- **Fully autonomous, no gates:** unsafe for code execution; no plan control.
- **Separate node per gate:** more graph surface for no benefit.
