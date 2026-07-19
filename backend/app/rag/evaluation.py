"""Retrieval-quality harness: precision@k over a fixed query set (Task 3.13).

Indexes a known repo, runs a fixed set of (query, expected_symbol) pairs through
the hybrid retriever, and reports precision@k — the fraction of queries whose
expected symbol appears in the top-k results. This is the regression signal
ADR-0008 calls for tuning fusion weights against (never a reranker), and feeds
Phase 5's eval harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.rag.indexer import Indexer
from app.rag.retriever import Retriever


@dataclass(frozen=True)
class RetrievalCase:
    """One query with the symbol expected to appear in its top-k results."""

    query: str
    expected_symbol: str


@dataclass
class CaseResult:
    case: RetrievalCase
    hit: bool
    rank: int | None  # 1-indexed position of the expected symbol, if found


@dataclass
class PrecisionReport:
    results: list[CaseResult]

    @property
    def precision_at_k(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hit) / len(self.results)

    def misses(self) -> list[RetrievalCase]:
        return [r.case for r in self.results if not r.hit]


def evaluate_retrieval(
    retriever: Retriever, project_id: str, cases: list[RetrievalCase], k: int = 5
) -> PrecisionReport:
    """Run each case's query and check whether its expected symbol is in the top-k."""
    results: list[CaseResult] = []
    for case in cases:
        hits = retriever.retrieve(project_id, case.query, k=k)
        rank = next(
            (i for i, h in enumerate(hits, start=1) if h.symbol == case.expected_symbol), None
        )
        results.append(CaseResult(case=case, hit=rank is not None, rank=rank))
    return PrecisionReport(results=results)


def index_and_evaluate(
    indexer: Indexer,
    retriever: Retriever,
    project_id: str,
    repo_root: Path,
    cases: list[RetrievalCase],
    k: int = 5,
) -> PrecisionReport:
    """Convenience: index ``repo_root`` fresh, then evaluate ``cases`` against it."""
    indexer.index_project(project_id, repo_root)
    retriever.invalidate(project_id)  # drop any stale in-memory BM25 cache
    return evaluate_retrieval(retriever, project_id, cases, k=k)
