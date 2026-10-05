from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["none", "mild", "strong", "explicit"]
SEVERITIES: tuple[str, ...] = ("none", "mild", "strong", "explicit")


@dataclass(frozen=True)
class DetectionLabel:
    is_vulgar: bool
    severity: Severity
    vulgar_terms: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "is_vulgar": self.is_vulgar,
                "severity": self.severity,
                "vulgar_terms": list(self.vulgar_terms),
            },
            ensure_ascii=True,
        )


def normalize_terms(terms: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        cleaned = " ".join(str(term).strip().lower().split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return tuple(ordered)


def label_from_dict(payload: dict[str, Any]) -> DetectionLabel:
    severity = str(payload.get("severity", "none")).strip().lower()
    if severity not in SEVERITIES:
        severity = "explicit" if bool(payload.get("is_vulgar")) else "none"
    terms = payload.get("vulgar_terms", [])
    if not isinstance(terms, list):
        terms = []
    is_vulgar = bool(payload.get("is_vulgar"))
    normalized = normalize_terms(terms)
    if not is_vulgar:
        return DetectionLabel(False, "none", ())
    if not normalized:
        is_vulgar = False
        severity = "none"
    elif severity == "none":
        severity = "mild"
    return DetectionLabel(is_vulgar, severity, normalized)  # type: ignore[arg-type]


def parse_model_json(text: str) -> DetectionLabel:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output is not JSON")
    payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model JSON must be an object")
    return label_from_dict(payload)
