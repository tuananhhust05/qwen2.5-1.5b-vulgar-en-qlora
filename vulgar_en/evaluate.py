from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

from vulgar_en.config import load_config
from vulgar_en.infer import detect, load_detector
from vulgar_en.nvidia import configure_nvidia_runtime
from vulgar_en.schema import DetectionLabel, label_from_dict, normalize_terms


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _term_f1(gold: DetectionLabel, pred: DetectionLabel) -> tuple[float, float, float]:
    gold_set = set(gold.vulgar_terms)
    pred_set = set(normalize_terms(pred.vulgar_terms))
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    if not gold_set:
        return 0.0, 0.0, 0.0
    true_positive = len(gold_set & pred_set)
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gold_set)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def evaluate(
    config_path: str,
    adapter_dir: str | None,
    split_file: str | None,
    use_adapter: bool = True,
    limit: int | None = None,
) -> dict:
    config = load_config(config_path)
    path = Path(split_file or config.data.test_file)
    rows = _load_rows(path)
    if limit is not None:
        rows = rows[: max(0, limit)]
    tokenizer, model = load_detector(
        config, adapter_dir or config.infer.adapter_dir, use_adapter=use_adapter
    )
    gold_flags: list[int] = []
    pred_flags: list[int] = []
    gold_severity: list[str] = []
    pred_severity: list[str] = []
    term_scores: list[tuple[float, float, float]] = []
    failures: list[dict] = []
    for row in rows:
        gold = label_from_dict(row["label"])
        try:
            pred = detect(tokenizer, model, row["input_text"], config)
        except RuntimeError as exc:
            failures.append({"input_text": row["input_text"], "error": str(exc)})
            pred = DetectionLabel(False, "none", ())
        gold_flags.append(int(gold.is_vulgar))
        pred_flags.append(int(pred.is_vulgar))
        gold_severity.append(gold.severity)
        pred_severity.append(pred.severity)
        term_scores.append(_term_f1(gold, pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_flags, pred_flags, average="binary", zero_division=0
    )
    term_precision = sum(item[0] for item in term_scores) / max(len(term_scores), 1)
    term_recall = sum(item[1] for item in term_scores) / max(len(term_scores), 1)
    term_f1 = sum(item[2] for item in term_scores) / max(len(term_scores), 1)
    report = {
        "examples": len(rows),
        "parse_failures": len(failures),
        "is_vulgar": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": float(sum(g == p for g, p in zip(gold_flags, pred_flags)) / max(len(rows), 1)),
        },
        "severity_macro_f1": float(
            f1_score(gold_severity, pred_severity, average="macro", zero_division=0)
        ),
        "vulgar_terms": {
            "precision": term_precision,
            "recall": term_recall,
            "f1": term_f1,
        },
        "classification_report": classification_report(
            gold_flags, pred_flags, target_names=["clean", "vulgar"], zero_division=0
        ),
        "failures": failures[:10],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vulgar-language detector")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="outputs/eval_report.json")
    args = parser.parse_args()
    configure_nvidia_runtime()
    report = evaluate(args.config, args.adapter, args.split, use_adapter=not args.no_adapter, limit=args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(report)
    serializable["classification_report"] = report["classification_report"]
    out_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=True), encoding="utf-8")
    print(report["classification_report"])
    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
