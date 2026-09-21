from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autogen_mas.models import ShortAnswerQuestionRecord

from .base import DatasetAdapter, read_text_with_fallbacks


class CIARAdapter(DatasetAdapter):
    dataset_name = "ciar"

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def load(self) -> list[ShortAnswerQuestionRecord]:
        samples = json.loads(read_text_with_fallbacks(self.source_path))
        if not isinstance(samples, list):
            raise ValueError(f"CIAR dataset must be a JSON array: {self.source_path}")

        records: list[ShortAnswerQuestionRecord] = []
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(
                    f"CIAR sample at source index {sample_index} must be an object."
                )
            records.append(self._record_from_sample(sample_index, sample))
        return records

    def _record_from_sample(
        self,
        sample_index: int,
        sample: dict[str, Any],
    ) -> ShortAnswerQuestionRecord:
        question = _required_text(sample, "question", sample_index)
        correct_aliases = _required_text_list(sample, "answer", sample_index)
        incorrect_aliases = _required_text_list(sample, "incorrect answer", sample_index)

        return ShortAnswerQuestionRecord(
            question_id=f"{sample_index:04d}",
            dataset_name=self.dataset_name,
            task_type="math_short_answer",
            question=question,
            acceptable_answers=correct_aliases,
            adversarial_target_answers=incorrect_aliases,
            metadata={
                "source_path": str(self.source_path),
                "source_index": sample_index,
                "explanation": sample.get("explanation"),
                "incorrect_explanation": sample.get("incorrect explanation"),
            },
        )


def _required_text(sample: dict[str, Any], field_name: str, sample_index: int) -> str:
    value = sample.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"CIAR sample at source index {sample_index} is missing non-empty "
            f"'{field_name}'."
        )
    return value


def _required_text_list(
    sample: dict[str, Any],
    field_name: str,
    sample_index: int,
) -> list[str]:
    value = sample.get(field_name)
    if not isinstance(value, list):
        raise ValueError(
            f"CIAR sample at source index {sample_index} is missing list "
            f"'{field_name}'."
        )
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"CIAR sample at source index {sample_index} has invalid "
                f"'{field_name}' item."
            )
        normalized = item.strip()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    if not result:
        raise ValueError(
            f"CIAR sample at source index {sample_index} has empty '{field_name}'."
        )
    return result
