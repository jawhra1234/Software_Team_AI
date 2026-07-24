"""Task 4.10 — reviewer against a live model: schema-valid Review, real grounding.

Live: the reviewer agent runs against the real Ollama model and (optionally,
if it chooses to) grounds via the real hybrid retriever over a real pgvector
index. Requires Ollama + Postgres; marked integration and skipped otherwise.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from app.agents.reviewer import Reviewer
from app.core.config import ReviewerSettings, Settings
from app.graph.retrieval import RetrievalCapture
from app.providers.factory import get_provider
from app.rag.embeddings import ChunkEmbedder
from app.rag.indexer import Indexer
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.tools.base import ToolContext
from app.tools.registry import build_planner_registry

from tests.conftest import ollama_available

pytestmark = pytest.mark.integration

_DSN = Settings(_env_file=None).postgres.dsn


def _postgres_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


_SKIP = not (ollama_available() and _postgres_reachable(_DSN))

_PRICING_HELPER = '''\
def apply_levy(amount, rate_pct):
    """Add the mandatory surcharge to an order total (company standard rule)."""
    return round(amount + amount * rate_pct / 100.0, 2)
'''

_DIFF = '''\
--- /dev/null
+++ b/checkout.py
@@ -0,0 +1,4 @@
+from pricing_rules import apply_levy
+
+def checkout_total(cart, rate_pct):
+    return apply_levy(sum(cart), rate_pct)
'''


@pytest.mark.skipif(_SKIP, reason="requires live Ollama and local Postgres")
def test_live_reviewer_emits_schema_valid_review_and_can_ground(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pricing_rules.py").write_text(_PRICING_HELPER, encoding="utf-8")

    project_id = f"reviewer-live-{uuid.uuid4().hex[:8]}"
    settings = Settings(_env_file=None, reviewer=ReviewerSettings(grounding_steps=3))
    embedder = ChunkEmbedder(provider=get_provider("embed", settings))
    # A dedicated table (not the shared `rag_chunks_test`, used by other suites with
    # a small fake-embedder dim): this test uses the real 768-dim nomic embedder.
    store = VectorStore(_DSN, dim=settings.rag.embedding_dim, table="rag_chunks_reviewer_live_test")
    Indexer(store, embedder).index_project(project_id, repo)
    retriever = Retriever(store, embedder)

    reviewer = Reviewer(get_provider("reviewer", settings), build_planner_registry(), settings)
    ctx = ToolContext(workspace_path=repo, run_id="t", retriever=retriever, project_id=project_id)
    capture = RetrievalCapture()

    try:
        review = reviewer.review_change(
            plan=None,
            diff=_DIFF,
            verify_result=None,
            ctx=ctx,
            on_tool_result=capture.observe,
        )
    finally:
        store.clear_project(project_id)

    # The core proof: a live 7B reliably produces a *schema-valid* Review (no
    # exception means structured_call's repair-retry succeeded against a real model).
    assert review.verdict in ("approved", "changes_requested", "rejected")
    for issue in review.issues:
        assert issue.severity in ("blocker", "major", "minor", "nit")

    # If the reviewer chose to ground (not forced — a live model's call), the
    # retrieve tool must have actually worked, proving the wiring end to end.
    if capture.chunks:
        assert any(c.symbol == "apply_levy" for c in capture.chunks)
