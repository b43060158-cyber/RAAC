from __future__ import annotations

import re


_SQUARE_PATTERN = re.compile(r"[a-h][1-8]", re.IGNORECASE)
_WHOLE_SQUARE_PATTERN = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)


def canonicalize_chess_final_answer(
    answer: str,
    *,
    source_square: str | None = None,
) -> str:
    extracted = extract_chess_square(answer, source_square=source_square)
    if extracted is not None:
        return extracted
    return answer.strip().lower()


def extract_chess_square(
    content: str,
    *,
    source_square: str | None = None,
) -> str | None:
    normalized = content.strip().lower()
    if not normalized:
        return None
    if source_square:
        normalized = re.sub(
            rf"\b{re.escape(source_square.strip().lower())}\b",
            " ",
            normalized,
        )
    for line in normalized.splitlines():
        candidate = line.strip()
        if _WHOLE_SQUARE_PATTERN.fullmatch(candidate):
            return candidate
    final_answer_index = normalized.rfind("final answer")
    if final_answer_index != -1:
        matches = _SQUARE_PATTERN.findall(normalized[final_answer_index:])
        if matches:
            return matches[-1].lower()
    matches = _SQUARE_PATTERN.findall(normalized)
    if matches:
        return matches[-1].lower()
    return None


def chess_answer_tie_key(
    answer: str,
    *,
    source_square: str | None = None,
) -> tuple[str, str]:
    return (
        "square",
        canonicalize_chess_final_answer(answer, source_square=source_square),
    )
