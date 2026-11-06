"""Non-LLM reference baselines, fitted on the train split only.

* lexicon  - every vulgar term annotated >= MIN_COUNT times in train; severity of a
             match = the most frequent gold severity of that term in train.
* tfidf_lr - word 1-2gram + char 2-5gram TF-IDF with logistic regression for
             is_vulgar and severity; vulgar_terms come from the lexicon matcher.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from vulgar_en.config import load_config
from vulgar_en.evaluate import load_rows
from vulgar_en.metrics import full_report
from vulgar_en.schema import SEVERITIES, DetectionLabel, label_from_dict

MIN_COUNT = 3
SEVERITY_RANK = {name: index for index, name in enumerate(SEVERITIES)}


def build_lexicon(rows: list[dict]) -> dict[str, str]:
    severities: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        label = label_from_dict(row["label"])
        for term in label.vulgar_terms:
            severities[term][label.severity] += 1
    return {
        term: counts.most_common(1)[0][0]
        for term, counts in severities.items()
        if sum(counts.values()) >= MIN_COUNT
    }


def lexicon_match(text: str, lexicon: dict[str, str], patterns: dict[str, re.Pattern]) -> DetectionLabel:
    lowered = text.lower()
    found = [term for term, pattern in patterns.items() if pattern.search(lowered)]
    if not found:
        return DetectionLabel(False, "none", ())
    severity = max((lexicon[term] for term in found), key=lambda name: SEVERITY_RANK[name])
    return DetectionLabel(True, severity, tuple(sorted(found)))  # type: ignore[arg-type]


def compile_patterns(lexicon: dict[str, str]) -> dict[str, re.Pattern]:
    return {term: re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])") for term in lexicon}


def run(config_path: str, split_file: str | None) -> dict:
    config = load_config(config_path)
    train_rows = load_rows(Path(config.data.train_file))
    test_rows = load_rows(Path(split_file or config.data.test_file))
    gold = [label_from_dict(row["label"]) for row in test_rows]
    sources = [row.get("source", "unknown") for row in test_rows]
    texts = [row["input_text"] for row in test_rows]

    lexicon = build_lexicon(train_rows)
    patterns = compile_patterns(lexicon)
    lexicon_pred = [lexicon_match(text, lexicon, patterns) for text in texts]

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    train_texts = [row["input_text"] for row in train_rows]
    x_train = hstack([word.fit_transform(train_texts), char.fit_transform(train_texts)]).tocsr()
    x_test = hstack([word.transform(texts), char.transform(texts)]).tocsr()
    train_labels = [label_from_dict(row["label"]) for row in train_rows]
    flag_model = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    flag_model.fit(x_train, [int(item.is_vulgar) for item in train_labels])
    severity_model = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    severity_model.fit(x_train, [item.severity for item in train_labels])
    flags = flag_model.predict(x_test)
    severities = severity_model.predict(x_test)
    tfidf_pred = []
    for flag, severity, lex in zip(flags, severities, lexicon_pred):
        if not flag:
            tfidf_pred.append(DetectionLabel(False, "none", ()))
            continue
        severity = severity if severity != "none" else "strong"
        tfidf_pred.append(DetectionLabel(True, severity, lex.vulgar_terms))  # type: ignore[arg-type]

    return {
        "train_examples": len(train_rows),
        "lexicon_terms": len(lexicon),
        "lexicon": full_report(gold, lexicon_pred, sources),
        "tfidf_lr": full_report(gold, tfidf_pred, sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and score non-LLM baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--out", default="outputs/baselines.json")
    args = parser.parse_args()
    report = run(args.config, args.split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("lexicon", "tfidf_lr"):
        item = report[name]
        print(
            f"{name}: is_vulgar_f1={item['is_vulgar']['f1']:.4f} "
            f"severity_macro_f1={item['severity_macro_f1']:.4f} "
            f"terms_f1={item['vulgar_terms']['f1']:.4f}"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
