from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import QuestionRecord

_ANSWER_NEGATION_MARKERS = (
    "incorrect",
    "wrong",
    "not correct",
    "is false",
    "are false",
    "not preferable",
    "less appropriate",
    "not the best",
)


def _extract_choice_ids_from_fragment(text: str, valid_option_ids: list[str]) -> list[str]:
    matches: list[str] = []
    for option_id in valid_option_ids:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(option_id)}(?![A-Za-z0-9_])")
        if pattern.search(text):
            matches.append(option_id)
    return matches


def _explicit_option_endorsements(
    *,
    reasoning: str,
    valid_option_ids: list[str],
) -> list[tuple[int, list[str]]]:
    option_union = "|".join(re.escape(option_id) for option_id in valid_option_ids)
    if not option_union:
        return []
    option_group = rf"(?:{option_union})(?:\s*(?:,|/|and)\s*(?:{option_union}))*"
    patterns = (
        re.compile(rf"\boption[s]?\s+({option_group})\s+(?:is|are)\s+correct\b", re.IGNORECASE),
        re.compile(
            rf"\b(?:the\s+)?correct\s+answer(?:s)?\s+(?:is|are)\s+({option_group})\b",
            re.IGNORECASE,
        ),
        re.compile(rf"\banswer(?:s)?\s+(?:is|are)\s+({option_group})\b", re.IGNORECASE),
        re.compile(rf"\b(?:i|we)\s+(?:choose|select|picked|pick|chose)\s+({option_group})\b", re.IGNORECASE),
    )
    matches: list[tuple[int, list[str]]] = []
    for pattern in patterns:
        for match in pattern.finditer(reasoning):
            matches.append(
                (
                    match.start(),
                    _extract_choice_ids_from_fragment(match.group(1), valid_option_ids),
                )
            )
    return matches


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_reasoning_decision_option_ids(
    *,
    question_record: QuestionRecord,
    reasoning: str,
) -> list[str]:
    text = str(reasoning or "").strip()
    if not text:
        return []
    matches = _explicit_option_endorsements(
        reasoning=text,
        valid_option_ids=question_record.option_ids(),
    )
    if not matches:
        return []
    _, option_ids = max(matches, key=lambda item: item[0])
    return _dedupe_preserve_order(option_ids)


def answer_reasoning_consistency_check(
    *,
    question_record: QuestionRecord,
    assigned_option_ids: list[str],
    reasoning: str,
) -> bool:
    text = str(reasoning or "").strip()
    if not text:
        return False

    normalized = text.casefold()
    explicit_decision = extract_reasoning_decision_option_ids(
        question_record=question_record,
        reasoning=text,
    )
    if explicit_decision and sorted(explicit_decision) != sorted(assigned_option_ids):
        return False

    for option_id in assigned_option_ids:
        option_pattern = re.compile(
            rf"\boption\s+{re.escape(option_id.casefold())}\b", re.IGNORECASE
        )
        for match in option_pattern.finditer(text):
            window = normalized[
                max(0, match.start() - 40) : min(len(normalized), match.end() + 80)
            ]
            if any(marker in window for marker in _ANSWER_NEGATION_MARKERS):
                return False
    return True
