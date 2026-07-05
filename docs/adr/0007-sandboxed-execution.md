# ADR-0007: Sandboxed command execution + tool authorization pipeline

**Status:** Accepted

## Context
The `coder` executes LLM-generated code via `run_command`. Untrusted requirement text and ingested repositories are prompt-injection vectors that could steer tools toward destructive or exfiltrating actions. This is both a real safety issue and a common interview probe.

## Decision
All `run_command` execution happens in a **sandbox** (Docker container per workspace: `--network=none`, CPU/mem limits, hard timeouts, workspace mounted RW only). Every tool call flows through one pipeline: **schema validate → authorize (workspace path-jail + command allow/deny-list) → optional HITL command approval (autonomy=semi) → execute in sandbox → truncate output → trace → result.** No secrets in prompts/state/logs.

## Consequences
- Contains blast radius of generated/injected code.
- Path-jail prevents traversal outside the workspace.
- `run_command.requires_approval=True` wires cleanly into HITL autonomy levels.
- Sandbox is also the reproducibility boundary for `verify`.

## Alternatives rejected
- **Run commands directly on host:** unacceptable safety risk.
- **Allowlist only, no container:** insufficient isolation for network/filesystem side effects.
