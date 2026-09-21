from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from autogen_mas.models import ChessQuestionRecord

from .base import DatasetAdapter, read_text_with_fallbacks


_SQUARE_PATTERN = re.compile(r"^[a-h][1-8]$", re.IGNORECASE)
_PROMPT_TEMPLATE = (
    'Given the chess game "{game}", give one valid destination square '
    'for the chess piece at "{piece}". Give a one line explanation of why '
    "your destination square is a valid move. State your final answer in a "
    "new line with a 2 letter response following the regex [a-h][1-8]."
)


class ChessAdapter(DatasetAdapter):
    dataset_name = "chess"

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def load(self) -> list[ChessQuestionRecord]:
        samples = json.loads(read_text_with_fallbacks(self.source_path))
        if not isinstance(samples, dict):
            raise ValueError(f"Chess dataset must be a JSON object: {self.source_path}")

        records: list[ChessQuestionRecord] = []
        for sample_index, (sample_id, sample) in enumerate(samples.items()):
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(
                    f"Chess sample at source index {sample_index} must have a non-empty string id."
                )
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Chess sample '{sample_id}' at source index {sample_index} must be an object."
                )
            records.append(
                self._record_from_sample(
                    sample_id=sample_id,
                    sample_index=sample_index,
                    sample=sample,
                )
            )
        return records

    def _record_from_sample(
        self,
        *,
        sample_id: str,
        sample_index: int,
        sample: dict[str, Any],
    ) -> ChessQuestionRecord:
        raw_input = _required_text(sample, "input", sample_id)
        try:
            game, source_square = raw_input.rsplit(" ", 1)
        except ValueError as exc:
            raise ValueError(
                f"Chess sample '{sample_id}' must end with a source square after the move list."
            ) from exc
        source_square = _normalize_square(source_square, sample_id, "input source square")
        legal_target_squares = _required_square_list(sample, "target", sample_id)
        rendered_question = _PROMPT_TEMPLATE.format(game=game, piece=source_square)

        return ChessQuestionRecord(
            question_id=sample_id,
            dataset_name=self.dataset_name,
            game=game,
            source_square=source_square,
            legal_target_squares=legal_target_squares,
            rendered_question=rendered_question,
            metadata={
                "source_path": str(self.source_path),
                "source_index": sample_index,
                "source_task_id": sample_id,
                "raw_input": raw_input,
                "game": game,
                "source_square": source_square,
                "answer_format": "chess_square",
                "legal_target_squares": list(legal_target_squares),
                "rendered_question": rendered_question,
            },
        )


def _required_text(sample: dict[str, Any], field_name: str, sample_id: str) -> str:
    value = sample.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Chess sample '{sample_id}' is missing non-empty '{field_name}'."
        )
    return value.strip()


def _required_square_list(
    sample: dict[str, Any],
    field_name: str,
    sample_id: str,
) -> list[str]:
    value = sample.get(field_name)
    if not isinstance(value, list):
        raise ValueError(
            f"Chess sample '{sample_id}' is missing list '{field_name}'."
        )
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Chess sample '{sample_id}' has invalid '{field_name}' item."
            )
        normalized = _normalize_square(item, sample_id, field_name)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    if not result:
        raise ValueError(f"Chess sample '{sample_id}' has empty '{field_name}'.")
    return result


def _normalize_square(value: str, sample_id: str, field_name: str) -> str:
    normalized = value.strip().lower()
    if not _SQUARE_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Chess sample '{sample_id}' has invalid {field_name}: {value!r}."
        )
    return normalized
