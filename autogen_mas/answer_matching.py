from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Any


_NUMBER_PATTERN_SOURCE = (
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[-+]?\d+(?:\.\d*)?)?%?"
)
_NUMBER_PATTERN = re.compile(_NUMBER_PATTERN_SOURCE)
_EXPLICIT_FINAL_ANSWER_PATTERN = re.compile(
    rf"(?:final\s+answer|answer)\s*(?:is|=|:)\s*(?P<answer>{_NUMBER_PATTERN_SOURCE})",
    re.IGNORECASE,
)
_CHESS_SQUARE_PATTERN = re.compile(r"[a-h][1-8]", re.IGNORECASE)
_CHESS_SQUARE_LINE_PATTERN = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)


def is_short_answer_correct(
    predicted_answer: str,
    acceptable_answers: list[str],
    *,
    tolerance: float = 1e-6,
    metadata: dict[str, Any] | None = None,
) -> bool:
    predicted = _normalize_text(
        canonicalize_short_answer(predicted_answer, metadata=metadata)
    )
    if not predicted:
        return False
    predicted_number = _parse_numeric_answer(predicted_answer)
    for acceptable_answer in acceptable_answers:
        acceptable = _normalize_text(
            canonicalize_short_answer(acceptable_answer, metadata=metadata)
        )
        if predicted == acceptable:
            return True
        acceptable_number = _parse_numeric_answer(acceptable_answer)
        if predicted_number is not None and acceptable_number is not None:
            if abs(float(predicted_number - acceptable_number)) <= tolerance:
                return True
    return False


def short_answer_tie_key(
    answer: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    answer = canonicalize_short_answer(answer, metadata=metadata)
    numeric = _parse_numeric_answer(answer)
    if numeric is not None:
        return ("number", str(numeric))
    return ("text", _normalize_text(answer))


def explicit_short_answer_from_reasoning(
    reasoning: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    if _short_answer_format(metadata) == "chess_square":
        return _extract_chess_square_answer(
            reasoning,
            source_square=_source_square(metadata),
        )
    matches = list(_EXPLICIT_FINAL_ANSWER_PATTERN.finditer(reasoning))
    if not matches:
        return None
    return re.sub(r"\s+", "", matches[-1].group("answer")).strip()


def canonicalize_short_answer(
    answer: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    stripped = answer.strip()
    if _short_answer_format(metadata) == "chess_square":
        extracted = _extract_chess_square_answer(
            stripped,
            source_square=_source_square(metadata),
        )
        if extracted is not None:
            return extracted
    return stripped


def _normalize_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[\s,]+", "", normalized)
    normalized = normalized.rstrip(".")
    return normalized


def _short_answer_format(metadata: dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get("answer_format")
    return value.strip().lower() if isinstance(value, str) else ""


def _source_square(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("source_square")
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _extract_chess_square_answer(
    content: str,
    *,
    source_square: str | None,
) -> str | None:
    normalized = content.lower()
    if source_square:
        normalized = re.sub(
            rf"\b{re.escape(source_square.lower())}\b",
            " ",
            normalized,
        )
    for line in normalized.splitlines():
        token = line.strip()
        if _CHESS_SQUARE_LINE_PATTERN.fullmatch(token):
            return token
    final_answer_index = normalized.rfind("final answer")
    if final_answer_index != -1:
        matches = _CHESS_SQUARE_PATTERN.findall(normalized[final_answer_index:])
        if matches:
            return matches[-1].lower()
    matches = _CHESS_SQUARE_PATTERN.findall(normalized)
    if matches:
        return matches[-1].lower()
    return None


def _parse_numeric_answer(value: str) -> Fraction | None:
    match = _NUMBER_PATTERN.search(value.strip())
    if not match:
        return None
    token = re.sub(r"\s+", "", match.group(0))
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            number = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        else:
            number = Fraction(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    if is_percent:
        number /= 100
    return number
