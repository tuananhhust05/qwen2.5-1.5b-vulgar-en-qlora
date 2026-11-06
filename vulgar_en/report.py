"""Aggregate result JSONs into results/summary.json and print a Markdown table."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

METRICS = {
    "is_vulgar_accuracy": ("is_vulgar", "accuracy"),
    "is_vulgar_precision": ("is_vulgar", "precision"),
    "is_vulgar_recall": ("is_vulgar", "recall"),
    "is_vulgar_f1": ("is_vulgar", "f1"),
    "severity_macro_f1": ("severity_macro_f1",),
    "vulgar_terms_f1": ("vulgar_terms", "f1"),
}


def _get(report: dict, path: tuple[str, ...]) -> float:
    value = report
    for key in path:
        value = value[key]
    return float(value)


def flatten(report: dict) -> dict:
    row = {name: _get(report, path) for name, path in METRICS.items()}
    if "json_valid_rate" in report:
        row["json_valid_rate"] = report["json_valid_rate"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    results = Path(args.results)
    load = lambda name: json.loads((results / name).read_text(encoding="utf-8"))  # noqa: E731

    summary: dict = {}
    baselines = load("baselines.json")
    summary["lexicon"] = flatten(baselines["lexicon"])
    summary["tfidf_lr"] = flatten(baselines["tfidf_lr"])
    base = load("base_zero_shot.json")
    summary["base_zero_shot"] = flatten(base) | {"ci95": base["ci95"], "examples": base["examples"]}

    runs = {}
    for path in sorted(results.glob("finetuned_seed*.json")):
        seed = re.search(r"seed(\d+)", path.name).group(1)
        report = json.loads(path.read_text(encoding="utf-8"))
        runs[seed] = flatten(report) | {
            "ci95": report["ci95"],
            "throughput": report.get("throughput"),
            "latency": report.get("latency"),
            "by_source": {source: flatten(item) for source, item in report["by_source"].items()},
        }
    summary["finetuned_runs"] = runs
    if runs:
        summary["finetuned_mean_std"] = {
            name: {
                "mean": statistics.mean(run[name] for run in runs.values()),
                "std": statistics.stdev(run[name] for run in runs.values()) if len(runs) > 1 else 0.0,
            }
            for name in list(METRICS) + ["json_valid_rate"]
        }
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    columns = ["is_vulgar_accuracy", "is_vulgar_f1", "severity_macro_f1", "vulgar_terms_f1", "json_valid_rate"]
    print("| Model | " + " | ".join(columns) + " |")
    print("| --- |" + " ---: |" * len(columns))
    for name in ("lexicon", "tfidf_lr", "base_zero_shot"):
        row = summary[name]
        print(f"| {name} | " + " | ".join(f"{row[c]:.4f}" if c in row else "-" for c in columns) + " |")
    if runs:
        agg = summary["finetuned_mean_std"]
        print(
            f"| qlora ({len(runs)} seeds) | "
            + " | ".join(f"{agg[c]['mean']:.4f} ± {agg[c]['std']:.4f}" for c in columns)
            + " |"
        )


if __name__ == "__main__":
    main()
