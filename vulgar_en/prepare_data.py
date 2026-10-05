from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from vulgar_en.config import load_config
from vulgar_en.data_bank import all_examples
from vulgar_en.prompts import build_messages
from vulgar_en.schema import DetectionLabel, label_from_dict


def _record(text: str, label: DetectionLabel) -> dict:
    assistant = label.to_json()
    return {
        "input_text": text,
        "label": json.loads(assistant),
        "messages": build_messages(text, assistant),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_labeled(path: Path) -> list[tuple[str, DetectionLabel]]:
    if not path.exists():
        return []
    rows: list[tuple[str, DetectionLabel]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = str(payload.get("input_text") or "").strip()
            if not text:
                continue
            rows.append((text, label_from_dict(payload.get("label") or {})))
    return rows


def collect_examples(config_path: str) -> list[tuple[str, DetectionLabel]]:
    config = load_config(config_path)
    merged: dict[str, DetectionLabel] = {}
    if config.data.include_seed:
        for text, label in all_examples():
            merged[text] = label
    for text, label in _load_labeled(Path(config.data.labeled_file)):
        merged[text] = label
    return list(merged.items())


def prepare(config_path: str) -> dict[str, int]:
    config = load_config(config_path)
    examples = collect_examples(config_path)
    if not examples:
        raise FileNotFoundError("No training examples. Run python -m vulgar_en.download_data or keep include_seed true.")
    texts = [item[0] for item in examples]
    labels = [item[1] for item in examples]
    stratify = [int(item.is_vulgar) for item in labels]
    holdout = config.data.val_ratio + config.data.test_ratio
    train_texts, rest_texts, train_labels, rest_labels = train_test_split(
        texts,
        labels,
        test_size=holdout,
        random_state=config.data.seed,
        stratify=stratify,
    )
    relative_test = config.data.test_ratio / holdout
    rest_stratify = [int(item.is_vulgar) for item in rest_labels]
    eval_texts, test_texts, eval_labels, test_labels = train_test_split(
        rest_texts,
        rest_labels,
        test_size=relative_test,
        random_state=config.data.seed,
        stratify=rest_stratify,
    )
    splits = {
        config.data.train_file: [_record(text, label) for text, label in zip(train_texts, train_labels)],
        config.data.eval_file: [_record(text, label) for text, label in zip(eval_texts, eval_labels)],
        config.data.test_file: [_record(text, label) for text, label in zip(test_texts, test_labels)],
    }
    for path, rows in splits.items():
        _write_jsonl(Path(path), rows)
    return {path: len(rows) for path, rows in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/eval/test JSONL splits")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    counts = prepare(args.config)
    for path, count in counts.items():
        print(f"{path}: {count}")


if __name__ == "__main__":
    main()
