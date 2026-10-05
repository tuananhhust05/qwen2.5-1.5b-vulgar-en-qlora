from vulgar_en.download_data import from_hatexplain_row, from_toxic_spans_row, from_tweet_eval_row


def test_toxic_spans_extracts_terms() -> None:
    row = {
        "text_of_post": "This is fucking stupid.",
        "text": {"fucking": 1.0, "stupid": 0.7},
        "type": {"profane/obscene": 0.7, "insult": 0.7},
        "position": [],
    }
    converted = from_toxic_spans_row(row)
    assert converted is not None
    text, label = converted
    assert "fucking" in text
    assert label.is_vulgar is True
    assert label.severity == "explicit"
    assert "fucking" in label.vulgar_terms


def test_hatexplain_normal_is_clean() -> None:
    row = {
        "post_tokens": ["hello", "there", "friend"],
        "annotators": {"label": ["normal", "normal", "normal"]},
        "rationales": [],
    }
    converted = from_hatexplain_row(row)
    assert converted is not None
    _, label = converted
    assert label.is_vulgar is False
    assert label.vulgar_terms == ()


def test_hatexplain_list_annotators_and_terms() -> None:
    row = {
        "post_tokens": ["what", "the", "fuck"],
        "annotators": [
            {"label": "offensive"},
            {"label": "offensive"},
            {"label": "normal"},
        ],
        "rationales": [[0, 0, 1], [0, 0, 1]],
    }
    converted = from_hatexplain_row(row)
    assert converted is not None
    _, label = converted
    assert label.is_vulgar is True
    assert "fuck" in label.vulgar_terms
    assert "the" not in label.vulgar_terms


def test_tweet_eval_keeps_clean_only() -> None:
    clean = from_tweet_eval_row({"text": "Have a nice day", "label": 0})
    offensive = from_tweet_eval_row({"text": "some insult", "label": 1})
    assert clean is not None
    assert clean[1].is_vulgar is False
    assert offensive is None
