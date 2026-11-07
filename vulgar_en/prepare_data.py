from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from vulgar_en.config import load_config
from vulgar_en.data_bank import all_examples
from vulgar_en.prompts import build_messages
from vulgar_en.schema import DetectionLabel, label_from_dict

SEED_SOURCE = "seed_bank"


def _record(text: str, label: DetectionLabel, source: str) -> dict:
    assistant = label.to_json()
    return {
        "input_text": text,
        "label": json.loads(assistant),
        "source": source,
        "messages": build_messages(text, assistant),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def dedupe_key(text: str) -> str:
    """Case/whitespace/punctuation-insensitive key used to drop near-identical posts."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _load_labeled(path: Path) -> list[tuple[str, DetectionLabel, str]]:
    if not path.exists():
        return []
    rows: list[tuple[str, DetectionLabel, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = str(payload.get("input_text") or "").strip()
            if not text:
                continue
            source = str(payload.get("source") or "labeled")
            rows.append((text, label_from_dict(payload.get("label") or {}), source))
    return rows


def collect_examples(config_path: str) -> tuple[list[tuple[str, DetectionLabel, str]], list[tuple[str, DetectionLabel, str]], dict]:
    """Return (public, seed, stats). Public rows are real posts; seed rows are synthetic."""
    config = load_config(config_path)
    public: dict[str, tuple[str, DetectionLabel, str]] = {}
    conflicts: set[str] = set()
    for text, label, source in _load_labeled(Path(config.data.labeled_file)):
        key = dedupe_key(text)
        if not key:
            continue
        if key in public and public[key][1].is_vulgar != label.is_vulgar:
            conflicts.add(key)
        public[key] = (text, label, source)
    for key in conflicts:
        public.pop(key, None)
    seed: list[tuple[str, DetectionLabel, str]] = []
    if config.data.include_seed:
        for text, label in all_examples():
            if dedupe_key(text) not in public:
                seed.append((text, label, SEED_SOURCE))
    stats = {"label_conflicts_dropped": len(conflicts)}
    return list(public.values()), seed, stats


def prepare(config_path: str) -> dict:
    config = load_config(config_path)
    public, seed, stats = collect_examples(config_path)
    if not public and not seed:
        raise FileNotFoundError("No training examples. Run python -m vulgar_en.download_data or keep include_seed true.")
    holdout = config.data.val_ratio + config.data.test_ratio
    if public:
        strata = [f"{source}:{int(label.is_vulgar)}" for _, label, source in public]
        train_rows, rest = train_test_split(
            public, test_size=holdout, random_state=config.data.seed, stratify=strata
        )
        rest_strata = [f"{source}:{int(label.is_vulgar)}" for _, label, source in rest]
        eval_rows, test_rows = train_test_split(
            rest,
            test_size=config.data.test_ratio / holdout,
            random_state=config.data.seed,
            stratify=rest_strata,
        )
    else:
        # seed-only mode (offline smoke test): split the synthetic bank instead
        strata = [int(label.is_vulgar) for _, label, _ in seed]
        seed, rest = train_test_split(seed, test_size=holdout, random_state=config.data.seed, stratify=strata)
        rest_strata = [int(label.is_vulgar) for _, label, _ in rest]
        eval_rows, test_rows = train_test_split(
            rest,
            test_size=config.data.test_ratio / holdout,
            random_state=config.data.seed,
            stratify=rest_strata,
        )
        train_rows = []
    # Synthetic seed/template rows are train-only so templated sentences never reach eval/test.
    train_rows = list(train_rows) + list(seed)
    splits = {
        config.data.train_file: train_rows,
        config.data.eval_file: eval_rows,
        config.data.test_file: test_rows,
    }
    summary: dict = {"stats": stats, "splits": {}}
    for path, rows in splits.items():
        _write_jsonl(Path(path), [_record(text, label, source) for text, label, source in rows])
        summary["splits"][path] = {
            "rows": len(rows),
            "vulgar": sum(1 for _, label, _ in rows if label.is_vulgar),
            "by_source": dict(sorted(Counter(source for _, _, source in rows).items())),
            "by_severity": dict(sorted(Counter(label.severity for _, label, _ in rows).items())),
        }
    summary_path = Path(config.data.processed_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/eval/test JSONL splits")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2))


if __name__ == "__main__":
    main()
