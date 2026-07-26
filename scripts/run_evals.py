"""Run the Phase-5 eval suite live and diff against the saved baseline.

Drives the real graph (live models) across the fixed task suite, aggregates the
per-task reports into suite metrics, and compares them to
``backend/evals/baseline.json``. Exits non-zero **only** if a *deterministic*
metric regressed (retrieval precision@k) — stochastic live-model metrics are
printed as a before/after trend but never fail the build, because a 7B jitters
run to run.

    python scripts/run_evals.py                 # run all tasks, compare to baseline
    python scripts/run_evals.py --category rag_required
    python scripts/run_evals.py --update-baseline   # save this run as the new baseline

Requires live Ollama (qwen2.5-coder + nomic-embed-text) + Postgres/pgvector.
"""

from __future__ import annotations

# ruff: noqa: E402  — path bootstrap must precede app imports.
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "backend"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.clock import now_iso
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.evals import tasks as task_mod
from app.evals.metrics import DETERMINISTIC_KEYS, aggregate
from app.evals.regression import compare, load_baseline, save_baseline
from app.evals.report import RunReport
from app.evals.tasks import ALL_TASKS, TaskContext
from app.rag.factory import build_rag_stack
from app.tools.sandbox import get_sandbox

_BASELINE_PATH = _HERE.parent / "backend" / "evals" / "baseline.json"
_BACKEND_ROOT = _HERE.parent / "backend"


def _fmt(value: object) -> str:
    if value is None:
        return "  n/a"
    if isinstance(value, float):
        return f"{value:6.3f}"
    return f"{value:>5}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase-5 eval suite.")
    parser.add_argument("--category", help="run only tasks in this category")
    parser.add_argument("--update-baseline", action="store_true",
                        help="save this run's metrics as the new baseline")
    args = parser.parse_args(argv)

    settings = get_settings()  # lru_cached; mutation persists to graph builders
    settings.models.coder.temperature = 0.0
    settings.models.reviewer.temperature = 0.0
    configure_logging(settings)

    rag = build_rag_stack(settings)
    ctx = TaskContext(
        settings=settings, sandbox=get_sandbox(settings.sandbox), rag=rag,
        retrieval_repo=_BACKEND_ROOT / "app",
    )

    selected = ALL_TASKS
    if args.category:
        selected = [t for t in ALL_TASKS if _category_of(t) == args.category]
        if not selected:
            print(f"no tasks in category {args.category!r}")
            return 2

    print("=" * 70)
    print("PHASE-5 EVAL SUITE (live)")
    print("=" * 70)
    reports: list[RunReport] = []
    for task in selected:
        name = task.__name__
        print(f"\n>>> {name} ...", flush=True)
        try:
            report = task(ctx)
        except Exception as exc:  # a harness error is recorded, not fatal to the suite
            report = RunReport(task_id=name, category=_category_of(task), error=str(exc))
            print(f"    ERROR: {exc}")
        reports.append(report)
        _print_report(report)

    metrics = aggregate(reports)
    baseline = load_baseline(_BASELINE_PATH)
    result = compare(metrics, baseline)

    _print_summary(metrics.to_dict(), result)

    if args.update_baseline:
        save_baseline(_BASELINE_PATH, metrics, stamped_at=now_iso())
        print(f"\nbaseline updated -> {_BASELINE_PATH}")
        return 0

    if result.has_gated_regression:
        keys = ", ".join(d.key for d in result.gated_regressions)
        print(f"\nREGRESSION (deterministic): {keys} — exiting non-zero.")
        return 1
    print("\nno deterministic regression.")
    return 0


def _category_of(task: object) -> str:
    # Each task function's category is the constant matching its name prefix; derive it
    # by running nothing — just map by the task's own module-level naming convention.
    mapping = {
        task_mod.happy_path_task: "happy_path",
        task_mod.rag_required_task: "rag_required",
        task_mod.defect_injection_task: "defect_injection",
        task_mod.cross_run_memory_task: "cross_run_memory",
        task_mod.retrieval_task: "retrieval",
    }
    return mapping.get(task, "unknown")  # type: ignore[arg-type]


def _print_report(r: RunReport) -> None:
    if r.error:
        print(f"    [{r.category}] ERROR: {r.error}")
        return
    bits = [f"status={r.status}"]
    if r.verify_passed is not None:
        bits.append(f"verify={'PASS' if r.verify_passed else 'FAIL'}")
    if r.review_verdicts:
        bits.append(f"review={r.review_verdicts}")
    if r.expected_symbol_reused is not None:
        bits.append(f"reused_helper={r.expected_symbol_reused}")
    if r.memory_influenced is not None:
        bits.append(f"memory_influenced={r.memory_influenced}")
    if r.precision_at_k is not None:
        bits.append(f"precision@k={r.precision_at_k:.3f} ({r.notes})")
    bits.append(f"steps={r.steps}")
    bits.append(f"{r.wall_clock_s:.0f}s")
    print(f"    [{r.category}] " + "  ".join(bits))


def _print_summary(current: dict[str, object], result: object) -> None:
    print("\n" + "=" * 70)
    print("SUITE METRICS (D = deterministic / gate-worthy)")
    print("=" * 70)
    deltas = {d.key: d for d in result.deltas}  # type: ignore[attr-defined]
    print(f"  {'metric':<28} {'baseline':>9} {'current':>9}  Δ")
    for key, value in current.items():
        d = deltas.get(key)
        mark = "D" if key in DETERMINISTIC_KEYS else " "
        base = _fmt(d.baseline) if d else "  n/a"
        arrow = {"up": "▲", "down": "▼", "same": "=", "n/a": " "}[d.direction] if d else " "
        print(f"{mark} {key:<28} {base:>9} {_fmt(value):>9}  {arrow}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
