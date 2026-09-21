"""Pure, dependency-free feature helpers shared by retrieval scorers.

Kept separate from ``reasoning_bank`` so both the bank and the registered
scorers can import them without a circular dependency.
"""

from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

_AUTHORITY_RISK_PENALTIES = {
    "light": -0.35,
    "medium": -0.8,
    "heavy": -1.5,
}


def words(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _WORD_RE.finditer(text)}


def overlap_score(left: str, right: str) -> float:
    left_words = words(left)
    right_words = words(right)
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def normalize_authority_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def normalize_authority_risk(value: object) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in _AUTHORITY_RISK_PENALTIES:
        return normalized
    return ""


def authority_penalty(record: dict[str, Any]) -> float:
    if not normalize_authority_flag(record.get("authority_packaging_flag")):
        return 0.0
    risk = normalize_authority_risk(record.get("authority_packaging_risk"))
    return _AUTHORITY_RISK_PENALTIES.get(risk, _AUTHORITY_RISK_PENALTIES["light"])
