"""Run every model script, collect per-model summaries, and write report.md."""
from __future__ import annotations

import glob
import importlib
import json
import os
import time
import traceback
from typing import Any, Dict, List

from utils import RESULTS_DIR, ensure_dirs

# Order matters only for the report table.
MODEL_MODULES: List[str] = [
    "model_xgboost",
    "model_svm",
    "model_logistic_regression",
    "model_naive_bayes",
    "model_random_forest",
    "model_knn",
    "model_voting",
    "model_stacking",
]


def run_all() -> List[Dict[str, Any]]:
    """Import and execute `main()` on each model module in sequence."""
    ensure_dirs()
    results = []
    for name in MODEL_MODULES:
        print(f"\n######## Running {name} ########")
        t0 = time.time()
        try:
            mod = importlib.import_module(name)
            mod.main()
            status = "ok"
            err = ""
        except Exception as exc:  # keep going if one model fails
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        results.append({
            "module": name,
            "status": status,
            "error": err,
            "elapsed_sec": round(time.time() - t0, 2),
        })
    return results


def _load_summaries() -> List[Dict[str, Any]]:
    """Collect every *_summary.json written by the individual model scripts."""
    summaries = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_summary.json"))):
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def build_report(run_info: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> str:
    """Render a Markdown report with CV means, best params, and test metrics."""
    lines = ["# BoW Multi-class Classification — Combined Report", ""]

    # Run status
    lines += ["## Run status", ""]
    lines += ["| Model | Status | Elapsed (s) | Error |",
              "|---|---|---|---|"]
    for r in run_info:
        lines.append(f"| {r['module']} | {r['status']} | {r['elapsed_sec']} | {r['error']} |")
    lines.append("")

    if not summaries:
        lines += ["_No model summaries were produced._", ""]
        return "\n".join(lines)

    # CV means on the untuned pipeline (baseline, not the selection metric).
    lines += ["## 5-fold CV on baseline pipeline (untuned, reporting only)", ""]
    lines += ["| Model | Accuracy | Precision | Recall | F1 |",
              "|---|---|---|---|---|"]
    for s in summaries:
        cv = s["cv_means"]
        lines.append(f"| {s['model']} | {_fmt(cv['accuracy'])} | {_fmt(cv['precision'])} "
                     f"| {_fmt(cv['recall'])} | {_fmt(cv['f1'])} |")
    lines.append("")

    # Test metrics after grid-search refit
    lines += ["## Test-set metrics (best model from grid search)", ""]
    lines += ["| Model | Accuracy | Precision | Recall | F1 | CV best F1 |",
              "|---|---|---|---|---|---|"]
    for s in summaries:
        t = s["test_metrics"]
        lines.append(f"| {s['model']} | {_fmt(t['accuracy'])} | {_fmt(t['precision'])} "
                     f"| {_fmt(t['recall'])} | {_fmt(t['f1'])} | {_fmt(s['cv_best_f1'])} |")
    lines.append("")

    # Best hyperparameters
    lines += ["## Best hyperparameters (grid search)", ""]
    for s in summaries:
        lines.append(f"### {s['model']}")
        lines.append("```json")
        lines.append(json.dumps(s["best_params"], indent=2, default=str))
        lines.append("```")
    lines.append("")

    # Winner by test F1
    winner = max(summaries, key=lambda s: s["test_metrics"]["f1"])
    lines += ["## Winner", "",
              f"**{winner['model']}** — test F1 = {_fmt(winner['test_metrics']['f1'])}.", ""]

    return "\n".join(lines)


def main() -> None:
    run_info = run_all()
    summaries = _load_summaries()
    report = build_report(run_info, summaries)

    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    # also mirror to repo root for convenience
    with open("report.md", "w") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\nReport saved to: {report_path} and ./report.md")


if __name__ == "__main__":
    main()
