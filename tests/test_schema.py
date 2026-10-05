from vulgar_en.prompts import SYSTEM_PROMPT, build_messages
from vulgar_en.schema import DetectionLabel, parse_model_json


def test_parse_plain_json() -> None:
    label = parse_model_json('{"is_vulgar": true, "severity": "explicit", "vulgar_terms": ["Fuck", "fuck"]}')
    assert label.is_vulgar is True
    assert label.severity == "explicit"
    assert label.vulgar_terms == ("fuck",)


def test_parse_fenced_json() -> None:
    raw = """```json
{"is_vulgar": false, "severity": "none", "vulgar_terms": []}
```"""
    label = parse_model_json(raw)
    assert label == DetectionLabel(False, "none", ())


def test_clean_label_drops_terms() -> None:
    label = parse_model_json('{"is_vulgar": false, "severity": "mild", "vulgar_terms": ["damn"]}')
    assert label.is_vulgar is False
    assert label.severity == "none"
    assert label.vulgar_terms == ()


def test_build_messages_includes_assistant() -> None:
    messages = build_messages("hello", '{"is_vulgar": false, "severity": "none", "vulgar_terms": []}')
    assert messages[0]["role"] == "system"
    assert SYSTEM_PROMPT in messages[0]["content"]
    assert messages[1]["content"] == "Text: hello"
    assert messages[2]["role"] == "assistant"
