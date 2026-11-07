import pytest

from vulgar_en.baselines import build_lexicon, compile_patterns, lexicon_match
from vulgar_en.metrics import score
from vulgar_en.schema import DetectionLabel

CLEAN = DetectionLabel(False, "none", ())


def test_perfect_predictions() -> None:
    gold = [DetectionLabel(True, "explicit", ("fuck",)), CLEAN]
    report = score(gold, gold)
    assert report["is_vulgar"]["f1"] == 1.0
    assert report["vulgar_terms"]["f1"] == 1.0
    assert report["severity_accuracy"] == 1.0


def test_clean_agreement_earns_no_term_credit() -> None:
    gold = [CLEAN, CLEAN, DetectionLabel(True, "strong", ("shit",))]
    pred = [CLEAN, CLEAN, CLEAN]
    report = score(gold, pred)
    assert report["vulgar_terms"]["f1"] == 0.0
    assert report["is_vulgar"]["recall"] == 0.0


def test_term_micro_counts() -> None:
    gold = [DetectionLabel(True, "strong", ("shit", "ass"))]
    pred = [DetectionLabel(True, "strong", ("shit", "damn"))]
    terms = score(gold, pred)["vulgar_terms"]
    assert terms["precision"] == pytest.approx(0.5)
    assert terms["recall"] == pytest.approx(0.5)


def test_lexicon_baseline_word_boundaries() -> None:
    rows = [{"label": {"is_vulgar": True, "severity": "strong", "vulgar_terms": ["ass"]}}] * 3
    lexicon = build_lexicon(rows)
    patterns = compile_patterns(lexicon)
    assert lexicon_match("you ass", lexicon, patterns).is_vulgar
    assert not lexicon_match("an associate professor", lexicon, patterns).is_vulgar
