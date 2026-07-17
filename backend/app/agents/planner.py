"""Planner agent (Task 2.2, ARCHITECTURE.md §4.1).

Turns a raw request into a validated :class:`Plan`: a bounded, read-only
grounding loop (``list_dir``/``read_file``/``search_code``) followed by a
``structured_call`` that emits the ``Plan`` schema. Only truly blocking
ambiguity should surface as ``open_questions``; everything else is resolved as
a recorded ``assumption`` so planning can proceed without a human round-trip.
"""

from __future__ import annotations

from app.agents.toolcalls import extract_tool_calls
from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.state import Plan
from app.providers.base import ChatMessage, LLMProvider
from app.tools.authorization import AuthorizationPolicy, execute_tool
from app.tools.base import ToolContext, ToolRegistry

log = get_logger("agents.planner")

_SYSTEM_PROMPT = """\
You are the planning agent for an AI software engineering workspace.

Your job: turn a request into a grounded, executable Plan (not code).

Rules:
- Ground first: use read_file / list_dir / search_code to see what already
  exists before deciding what to build. Never assume file contents.
- Prefer reasonable defaults over questions: when something is ambiguous but a
  sensible default exists, record it as an assumption and proceed.
- Only use open_questions for genuinely blocking ambiguity (e.g. contradictory
  requirements) where no reasonable default exists. Keep it empty otherwise.
- Break the work into small, independently-verifiable tasks with concrete
  acceptance criteria and explicit target_paths. Order tasks via depends_on.
- When you have gathered enough context, stop calling tools and emit the Plan.
"""

_EMIT_INSTRUCTION = "You have enough context. Emit the final Plan now."


class Planner:
    """Produces a :class:`Plan` for a user request via bounded grounding + structured output."""

    def __init__(self, provider: LLMProvider, registry: ToolRegistry, settings: Settings) -> None:
        self._provider = provider
        self._registry = registry
        self._settings = settings
        # Read-only tools never touch run_command, so approval is never needed here.
        self._policy = AuthorizationPolicy.from_settings(settings, autonomy="auto")

    def create_plan(
        self,
        *,
        user_request: str,
        ctx: ToolContext,
        prior_plan: Plan | None = None,
        clarification_answers: list[str] | None = None,
    ) -> Plan:
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=_render_request(user_request, prior_plan, clarification_answers),
            ),
        ]
        tool_specs = self._registry.specs()

        for _ in range(self._settings.planner.grounding_steps):
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
                observation = result.output if result.ok else f"ERROR: {result.error}"
                messages.append(
                    ChatMessage(
                        role="tool", name=call.name, tool_call_id=call.id, content=observation
                    )
                )

        messages.append(ChatMessage(role="user", content=_EMIT_INSTRUCTION))
        plan = self._provider.structured(messages, Plan)
        plan.version = prior_plan.version + 1 if prior_plan is not None else 1
        log.info("plan_created", version=plan.version, tasks=len(plan.tasks))
        return plan


def _render_request(
    user_request: str, prior_plan: Plan | None, clarification_answers: list[str] | None
) -> str:
    parts = [f"Request: {user_request}"]
    if prior_plan is not None:
        parts.append(f"Prior plan summary (revise from here): {prior_plan.summary}")
        if prior_plan.assumptions:
            parts.append("Prior assumptions: " + "; ".join(prior_plan.assumptions))
    if clarification_answers:
        parts.append("Human clarifications:")
        parts.extend(f"- {a}" for a in clarification_answers)
    return "\n".join(parts)
