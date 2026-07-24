"""Reviewer agent (Task 4.1, ADR-0006).

A fresh-context adversarial reviewer: a bounded, **read-only** grounding loop
(``retrieve``/``search_code``/``read_file``/``list_dir``) followed by a
``structured_call`` that emits a :class:`~app.graph.state.Review`. Isolation is
the point (ADR-0006) — the reviewer sees only the approved plan, the
implementation diff, and the verify result. It never sees the coder's
scratchpad/reasoning, and it never edits code; it only produces structured
findings for the coder to act on.
"""

from __future__ import annotations

from app.agents.coder import ToolResultHook
from app.agents.toolcalls import extract_tool_calls
from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.state import Plan, Review, VerifyResult
from app.providers.base import ChatMessage, LLMProvider
from app.tools.authorization import AuthorizationPolicy, execute_tool
from app.tools.base import ToolContext, ToolRegistry

log = get_logger("agents.reviewer")

_SYSTEM_PROMPT = """\
You are the adversarial code reviewer for an AI software engineering workspace.

You review with fresh eyes: you do NOT see how the code was written, only the
result. Your job is to find real problems, not to nitpick.

What you can see: the approved plan (what was supposed to be built), the
implementation diff (what actually changed), and the verify/test result
(whether the automated checks passed). Nothing else.

Priorities, in order — judge the diff against these, not against style:
1. Correctness — does the change actually do what the plan asked? Any logic
   errors, edge cases, or behavior mismatches?
2. Security — injection, path traversal, secrets, unsafe deserialization,
   missing authorization/validation on inputs that need it.
3. Test failures or gaps — the diff should be covered by meaningful tests;
   flag missing coverage for non-trivial new behavior.
4. Architecture — violations of the plan's stated design, wrong layering,
   duplicated logic that already exists elsewhere in the repo.
5. Maintainability — genuinely confusing or fragile code (not taste).

Style, formatting, naming, and other nitpicks are NEVER blockers. If you notice
one, record it as `nit` severity — it is advisory only and must not stop the run.

Severity rubric for each issue:
- blocker: breaks correctness or introduces a real security problem.
- major: a significant defect that should be fixed before this is accepted
  (architecture violation, missing error handling on a critical path, a real
  correctness risk not severe enough to call a blocker).
- minor: worth fixing, not urgent, does not require another iteration.
- nit: style/formatting/naming — advisory only.

Ground before judging — this is mandatory, not optional: before you approve any
diff that adds or changes non-trivial logic, call `retrieve` or `search_code`
at least once to check whether equivalent logic already exists elsewhere in
the repo (architecture priority #4) and to see how related conventions are
used. Skipping this step is how duplicated logic slips through — do not
approve solely from the diff text. Use `read_file` / `list_dir`
(read-only — you cannot edit anything) for anything the diff alone doesn't
show. Only skip grounding for changes with no logic to check (e.g. pure
formatting, comments, or a one-line typo fix).

Verdict:
- "approved" — no blocker/major issues (minor/nit are fine and do not block).
- "changes_requested" — at least one blocker/major issue; describe each
  concretely (file, what's wrong, a concrete suggestion) so the fix is targeted.
- "rejected" — use only when the overall approach is fundamentally wrong and
  targeted fixes cannot address it (this escalates to a human; use rarely).

When you have enough context, stop calling tools and emit the Review.
"""

def _emit_instruction(max_issues: int) -> str:
    return (
        "You have enough context. Emit the final Review now. Each issue's `severity` "
        "must be exactly one of: blocker, major, minor, nit. Surface at most "
        f"{max_issues} issues, most severe first."
    )


class Reviewer:
    """Produces a :class:`Review` for a diff via bounded read-only grounding + structured output."""

    def __init__(self, provider: LLMProvider, registry: ToolRegistry, settings: Settings) -> None:
        self._provider = provider
        self._registry = registry
        self._settings = settings
        # Read-only tools never touch run_command, so approval is never needed here.
        self._policy = AuthorizationPolicy.from_settings(settings, autonomy="auto")

    def review_change(
        self,
        *,
        plan: Plan | None,
        diff: str,
        verify_result: VerifyResult | None,
        ctx: ToolContext,
        on_tool_result: ToolResultHook | None = None,
    ) -> Review:
        """``on_tool_result`` (mirrors Task 3.12) is per-call, not per-instance: the
        Reviewer is built once and reused across every ``review`` node invocation in
        a run, so a per-instance hook would leak state across calls."""
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=_render_review_input(plan, diff, verify_result)),
        ]
        tool_specs = self._registry.specs()

        for _ in range(self._settings.reviewer.grounding_steps):
            response = self._provider.chat(messages, tools=tool_specs)
            calls = extract_tool_calls(response.content, response.tool_calls, self._registry)
            if not calls:
                messages.append(ChatMessage(role="assistant", content=response.content))
                break
            messages.append(
                ChatMessage(role="assistant", content=response.content, tool_calls=calls)
            )
            for call in calls:
                result = execute_tool(self._registry, call.name, call.arguments, ctx, self._policy)
                if on_tool_result is not None:
                    on_tool_result(call.name, result)
                observation = result.output if result.ok else f"ERROR: {result.error}"
                messages.append(
                    ChatMessage(
                        role="tool", name=call.name, tool_call_id=call.id, content=observation
                    )
                )

        messages.append(
            ChatMessage(role="user", content=_emit_instruction(self._settings.reviewer.max_issues))
        )
        review = self._provider.structured(messages, Review)
        log.info("review_produced", verdict=review.verdict, issues=len(review.issues))
        return review


def _render_review_input(plan: Plan | None, diff: str, verify_result: VerifyResult | None) -> str:
    parts: list[str] = []
    if plan is not None:
        parts.append(f"Approved plan: {plan.summary}")
        if plan.tasks:
            parts.append("Planned tasks:")
            parts.extend(
                f"- {t.id} [{t.kind}] {t.title}: "
                f"{'; '.join(t.acceptance_criteria) or '(no explicit acceptance criteria)'}"
                for t in plan.tasks
            )
    else:
        parts.append("Approved plan: (none available)")

    parts.append(f"\nDiff (base commit..HEAD):\n{diff or '(no changes)'}")

    if verify_result is not None:
        parts.append(f"\nVerify result: {'PASSED' if verify_result.passed else 'FAILED'}")
        parts.append(verify_result.summary or "(no summary)")
    else:
        parts.append("\nVerify result: (not available)")

    return "\n".join(parts)
