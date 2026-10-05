from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from vulgar_en.evaluate import evaluate
from vulgar_en.nvidia import configure_nvidia_runtime


def _delta(before: float, after: float) -> dict[str, float]:
    return {"base": before, "finetuned": after, "delta": after - before}


def compare(config_path: str, adapter_dir: str | None, split_file: str | None, limit: int | None) -> dict:
    configure_nvidia_runtime()
    base_report = evaluate(
        config_path, adapter_dir, split_file, use_adapter=False, limit=limit
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tuned_report = evaluate(
        config_path, adapter_dir, split_file, use_adapter=True, limit=limit
    )
    return {
        "examples": tuned_report["examples"],
        "is_vulgar_f1": _delta(base_report["is_vulgar"]["f1"], tuned_report["is_vulgar"]["f1"]),
        "is_vulgar_accuracy": _delta(
            base_report["is_vulgar"]["accuracy"], tuned_report["is_vulgar"]["accuracy"]
        ),
        "severity_macro_f1": _delta(base_report["severity_macro_f1"], tuned_report["severity_macro_f1"]),
        "vulgar_terms_f1": _delta(base_report["vulgar_terms"]["f1"], tuned_report["vulgar_terms"]["f1"]),
        "parse_failures": {
            "base": base_report["parse_failures"],
            "finetuned": tuned_report["parse_failures"],
        },
        "base": {key: value for key, value in base_report.items() if key != "failures"},
        "finetuned": {key: value for key, value in tuned_report.items() if key != "failures"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base model against the fine-tuned adapter")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--out", default="outputs/compare_base_vs_finetuned.json")
    args = parser.parse_args()
    report = compare(args.config, args.adapter, args.split, args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"base", "finetuned"}}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
