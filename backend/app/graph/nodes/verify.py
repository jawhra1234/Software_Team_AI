"""Verify node (Task 2.4, ADR-0005).

Deterministic — no LLM. Wraps :class:`app.verify.runner.VerifyRunner`. On
failure, increments ``retries["verify"]`` (a delta the ``merge_counts`` reducer
sums) and escalates once ``GraphSettings.max_verify_retries`` is reached.
Always sets ``hitl_request`` explicitly (a real request or ``None``) so
downstream routing never reads a stale value from an earlier cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import CoderSettings, GraphSettings
from app.graph.state import AgentState, HITLRequest
from app.tools.sandbox import Sandbox
from app.verify.runner import VerifyRunner


def make_verify_node(
    sandbox: Sandbox, graph_settings: GraphSettings, coder_settings: CoderSettings
) -> Any:
    runner = VerifyRunner(sandbox, coder_settings)

    def _node(state: AgentState) -> dict[str, Any]:
        result = runner.run(Path(state["workspace_path"]))
        patch: dict[str, Any] = {"verify_result": result, "hitl_request": None}

        if result.passed:
            return patch

        current = state.get("retries", {}).get("verify", 0)
        patch["retries"] = {"verify": 1}
        if current + 1 >= graph_settings.max_verify_retries:
            patch["hitl_request"] = HITLRequest(
                kind="escalation",
                context=f"Verify failed after {current + 1} attempt(s): {result.summary}",
                options=["retry", "accept", "abort"],
                payload={"origin_node": "coder"},
            )
        return patch

    return _node
