"""Task metrics (pure Python + scikit-learn, no GPU)."""

from __future__ import annotations

import random
from collections import defaultdict

from sklearn.metrics import f1_score, precision_recall_fscore_support

from vulgar_en.schema import SEVERITIES, DetectionLabel, normalize_terms


def _binary(gold: list[DetectionLabel], pred: list[DetectionLabel]) -> dict[str, float]:
    y_true = [int(item.is_vulgar) for item in gold]
    y_pred = [int(item.is_vulgar) for item in pred]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "accuracy": accuracy}


def _severity_macro_f1(gold: list[DetectionLabel], pred: list[DetectionLabel]) -> float:
    return float(
        f1_score(
            [item.severity for item in gold],
            [item.severity for item in pred],
            labels=list(SEVERITIES),
            average="macro",
            zero_division=0,
        )
    )


def _terms(gold: list[DetectionLabel], pred: list[DetectionLabel]) -> dict[str, float]:
    """Micro P/R/F1 over (example, term) pairs: clean-vs-clean agreement earns no credit."""
    true_positive = predicted = expected = 0
    for g, p in zip(gold, pred):
        gold_set = set(g.vulgar_terms)
        pred_set = set(normalize_terms(p.vulgar_terms)) if p.is_vulgar else set()
        true_positive += len(gold_set & pred_set)
        predicted += len(pred_set)
        expected += len(gold_set)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score(gold: list[DetectionLabel], pred: list[DetectionLabel]) -> dict:
    return {
        "examples": len(gold),
        "is_vulgar": _binary(gold, pred),
        "severity_macro_f1": _severity_macro_f1(gold, pred),
        "severity_accuracy": sum(g.severity == p.severity for g, p in zip(gold, pred)) / max(len(gold), 1),
        "vulgar_terms": _terms(gold, pred),
    }


def bootstrap(gold: list[DetectionLabel], pred: list[DetectionLabel], resamples: int = 1000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(gold)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        index = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in index]
        p = [pred[i] for i in index]
        values["is_vulgar_f1"].append(_binary(g, p)["f1"])
        values["severity_macro_f1"].append(_severity_macro_f1(g, p))
        values["vulgar_terms_f1"].append(_terms(g, p)["f1"])
    interval = {}
    for key, samples in values.items():
        samples.sort()
        interval[key] = [samples[int(0.025 * resamples)], samples[int(0.975 * resamples) - 1]]
    return interval


def full_report(gold: list[DetectionLabel], pred: list[DetectionLabel], sources: list[str]) -> dict:
    report = score(gold, pred)
    report["ci95"] = bootstrap(gold, pred)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(sources):
        grouped[source].append(index)
    report["by_source"] = {
        source: score([gold[i] for i in index], [pred[i] for i in index])
        for source, index in sorted(grouped.items())
    }
    return report
