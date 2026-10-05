from vulgar_en.data_bank import all_examples


def test_dataset_has_both_classes() -> None:
    examples = all_examples()
    assert len(examples) >= 200
    vulgar = sum(1 for _, label in examples if label.is_vulgar)
    clean = sum(1 for _, label in examples if not label.is_vulgar)
    assert vulgar > 50
    assert clean > 50


def test_clean_examples_have_no_terms() -> None:
    for text, label in all_examples():
        if not label.is_vulgar:
            assert label.severity == "none"
            assert label.vulgar_terms == ()
            assert text.strip()


def test_vulgar_examples_list_terms() -> None:
    for _, label in all_examples():
        if label.is_vulgar:
            assert label.severity in {"mild", "strong", "explicit"}
            assert len(label.vulgar_terms) >= 1
