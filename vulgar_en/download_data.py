from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from vulgar_en.schema import DetectionLabel, normalize_terms

MAX_CHARS = 1800
HATEXPLAIN_URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"
HATEXPLAIN_LABELS = {
    0: "hatespeech",
    1: "normal",
    2: "offensive",
    "0": "hatespeech",
    "1": "normal",
    "2": "offensive",
    "hatespeech": "hatespeech",
    "normal": "normal",
    "offensive": "offensive",
}


def _clean_text(text: str) -> str:
    return " ".join(str(text).replace("\x00", " ").split())[:MAX_CHARS]


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "to", "of", "in", "on", "at",
    "for", "with", "as", "by", "from", "is", "are", "was", "were", "be", "been",
    "am", "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "my", "your", "his", "her", "their", "our", "me", "him", "them",
    "do", "does", "did", "not", "no", "so", "too", "very", "just", "about",
    "into", "over", "after", "before", "up", "down", "out", "off", "then",
    "than", "can", "will", "would", "should", "could", "have", "has", "had",
    "what", "when", "where", "who", "how", "why", "which", "all", "any",
}


def _usable_term(token: str) -> bool:
    cleaned = token.strip().lower()
    return len(cleaned) >= 2 and cleaned not in STOPWORDS and any(char.isalpha() for char in cleaned)


def _label(is_vulgar: bool, severity: str, terms: Iterable[str]) -> DetectionLabel | None:
    normalized = normalize_terms([term for term in terms if _usable_term(term)])
    if not is_vulgar:
        return DetectionLabel(False, "none", ())
    if not normalized:
        return None
    if severity not in {"mild", "strong", "explicit"}:
        severity = "strong"
    return DetectionLabel(True, severity, normalized)  # type: ignore[arg-type]


def terms_from_char_positions(text: str, positions: Iterable[Any]) -> list[str]:
    indexes: list[int] = []
    for pos in positions:
        try:
            index = int(pos)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(text):
            indexes.append(index)
    indexes = sorted(set(indexes))
    if not indexes:
        return []
    spans: list[str] = []
    start = previous = indexes[0]
    for index in indexes[1:]:
        if index == previous + 1:
            previous = index
            continue
        spans.append(text[start : previous + 1])
        start = previous = index
    spans.append(text[start : previous + 1])
    return [span.strip() for span in spans if span.strip()]


def from_toxic_spans_row(row: dict[str, Any]) -> tuple[str, DetectionLabel] | None:
    raw_post = row.get("text_of_post")
    if not raw_post and isinstance(row.get("text"), str):
        raw_post = row.get("text")
    original = str(raw_post or "")
    if not original.strip():
        return None
    span_scores = _as_mapping(row.get("text") if not isinstance(row.get("text"), str) else row.get("spans"))
    terms = [str(term) for term, score in span_scores.items() if float(score or 0) >= 0.5]
    if not terms:
        terms = terms_from_char_positions(original, row.get("position") or row.get("positions") or [])
    text = _clean_text(original)
    types = _as_mapping(row.get("type"))
    profane = float(types.get("profane/obscene") or types.get("profane_obscene") or 0)
    insult = float(types.get("insult") or 0)
    if not terms:
        return (text, DetectionLabel(False, "none", ()))
    if profane >= 0.5:
        severity = "explicit"
    elif profane > 0 or insult >= 0.5:
        severity = "strong"
    else:
        severity = "mild"
    label = _label(True, severity, terms)
    if label is None:
        return None
    return text, label


def _hatexplain_majority(row: dict[str, Any]) -> str:
    annotators = row.get("annotators")
    if isinstance(annotators, list):
        raw_labels = [item.get("label") for item in annotators if isinstance(item, dict)]
    else:
        raw_labels = _as_mapping(annotators).get("label") or []
    mapped = [HATEXPLAIN_LABELS.get(item, str(item).strip().lower()) for item in raw_labels]
    if not mapped:
        return "normal"
    return Counter(mapped).most_common(1)[0][0]


def _hatexplain_terms(row: dict[str, Any]) -> list[str]:
    tokens = [str(token) for token in (row.get("post_tokens") or [])]
    rationales = row.get("rationales") or []
    if not tokens:
        return []
    votes = [0] * len(tokens)
    valid = 0
    for rationale in rationales:
        if len(rationale) != len(tokens):
            continue
        valid += 1
        for index, flag in enumerate(rationale):
            votes[index] += int(flag)
    if valid == 0:
        return []
    threshold = max(1, (valid + 1) // 2)
    return [tokens[index] for index, vote in enumerate(votes) if vote >= threshold]


def from_hatexplain_row(row: dict[str, Any]) -> tuple[str, DetectionLabel] | None:
    tokens = [str(token) for token in (row.get("post_tokens") or [])]
    text = _clean_text(" ".join(tokens))
    if not text:
        return None
    majority = _hatexplain_majority(row)
    if majority == "normal":
        return text, DetectionLabel(False, "none", ())
    terms = _hatexplain_terms(row)
    severity = "explicit" if majority == "hatespeech" else "strong"
    label = _label(True, severity, terms)
    if label is None:
        return None
    return text, label


def from_tweet_eval_row(row: dict[str, Any]) -> tuple[str, DetectionLabel] | None:
    text = _clean_text(str(row.get("text") or ""))
    if not text:
        return None
    if int(row.get("label", 1)) != 0:
        return None
    return text, DetectionLabel(False, "none", ())


def _iter_splits(dataset: Any) -> Iterable[dict[str, Any]]:
    if hasattr(dataset, "keys"):
        for split in dataset.keys():
            for row in dataset[split]:
                yield dict(row)
        return
    for row in dataset:
        yield dict(row)


def _load_source(name: str, config: str | None = None) -> Any:
    from datasets import load_dataset

    if config is None:
        return load_dataset(name)
    return load_dataset(name, config)


def _load_hatexplain() -> list[dict[str, Any]]:
    with urllib.request.urlopen(HATEXPLAIN_URL, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        return []
    return [item for item in payload.values() if isinstance(item, dict)]


def download_sources() -> dict[str, list[dict[str, Any]]]:
    collected: dict[str, list[dict[str, Any]]] = {
        "toxic_spans": [],
        "hatexplain": [],
        "tweet_eval_offensive": [],
    }
    toxic = _load_source("heegyu/toxic-spans")
    for row in _iter_splits(toxic):
        converted = from_toxic_spans_row(row)
        if converted is None:
            continue
        text, label = converted
        collected["toxic_spans"].append(
            {"input_text": text, "label": json.loads(label.to_json()), "source": "toxic_spans"}
        )
    for row in _load_hatexplain():
        converted = from_hatexplain_row(row)
        if converted is None:
            continue
        text, label = converted
        collected["hatexplain"].append(
            {"input_text": text, "label": json.loads(label.to_json()), "source": "hatexplain"}
        )
    tweets = _load_source("cardiffnlp/tweet_eval", "offensive")
    for row in _iter_splits(tweets):
        converted = from_tweet_eval_row(row)
        if converted is None:
            continue
        text, label = converted
        collected["tweet_eval_offensive"].append(
            {
                "input_text": text,
                "label": json.loads(label.to_json()),
                "source": "tweet_eval_offensive",
            }
        )
    return collected


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique[row["input_text"]] = row
    return list(unique.values())


def balance_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    vulgar = [row for row in rows if row["label"]["is_vulgar"]]
    clean = [row for row in rows if not row["label"]["is_vulgar"]]
    rng = random.Random(seed)
    rng.shuffle(clean)
    if len(clean) > int(len(vulgar) * 1.2):
        clean = clean[: max(1, int(len(vulgar) * 1.2))]
    mixed = vulgar + clean
    rng.shuffle(mixed)
    return mixed


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def download(raw_dir: Path, seed: int) -> dict[str, int]:
    by_source = download_sources()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    merged: list[dict[str, Any]] = []
    for source, rows in by_source.items():
        write_jsonl(raw_dir / f"{source}.jsonl", rows)
        counts[source] = len(rows)
        merged.extend(rows)
    labeled = balance_rows(_dedupe(merged), seed)
    write_jsonl(raw_dir / "labeled.jsonl", labeled)
    counts["labeled"] = len(labeled)
    counts["vulgar"] = sum(1 for row in labeled if row["label"]["is_vulgar"])
    counts["clean"] = sum(1 for row in labeled if not row["label"]["is_vulgar"])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download English vulgar/offensive public datasets")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = download(Path(args.raw_dir), args.seed)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
