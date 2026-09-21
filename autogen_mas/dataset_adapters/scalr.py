from __future__ import annotations

import json
import io
from pathlib import Path
import re
from typing import Any

from autogen_mas.models import OptionRecord, QuestionRecord

from .base import DatasetAdapter, read_text_with_fallbacks


_CHOICE_FIELD_PATTERN = re.compile(r"^choice_(\d+)$")


class SCALRAdapter(DatasetAdapter):
    dataset_name = "scalr"

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def load(self) -> list[QuestionRecord]:
        records: list[QuestionRecord] = []
        for sample_index, line in enumerate(io.StringIO(read_text_with_fallbacks(self.source_path))):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                raise ValueError(
                    f"SCALR sample at source index {sample_index} must be an object."
                )
            records.append(self._record_from_sample(sample_index, sample))
        return records

    def _record_from_sample(
        self,
        sample_index: int,
        sample: dict[str, Any],
    ) -> QuestionRecord:
        question = _required_text(sample, "question", sample_index)
        answer = _required_text(sample, "answer", sample_index)
        answer_index = _parse_non_negative_int(answer, "answer", sample_index)

        option_entries: list[tuple[int, str]] = []
        for field_name, value in sample.items():
            match = _CHOICE_FIELD_PATTERN.match(field_name)
            if match is None:
                continue
            option_entries.append(
                (_parse_non_negative_int(match.group(1), field_name, sample_index), str(value))
            )

        if not option_entries:
            raise ValueError(
                f"SCALR sample at source index {sample_index} does not contain any choice_* fields."
            )

        option_entries.sort(key=lambda item: item[0])
        expected_indexes = list(range(len(option_entries)))
        actual_indexes = [choice_index for choice_index, _ in option_entries]
        if actual_indexes != expected_indexes:
            raise ValueError(
                f"SCALR sample at source index {sample_index} must have contiguous choice_* "
                f"fields starting at choice_0; found {actual_indexes}."
            )

        if answer_index >= len(option_entries):
            raise ValueError(
                f"SCALR sample at source index {sample_index} has answer index {answer_index}, "
                f"but only {len(option_entries)} choices are available."
            )

        options = [
            OptionRecord(option_id=_option_id(choice_index), text=option_text)
            for choice_index, (_, option_text) in enumerate(option_entries)
        ]

        return QuestionRecord(
            question_id=f"{sample_index:04d}",
            dataset_name=self.dataset_name,
            task_type="single_choice",
            question=question,
            options=options,
            correct_option_ids=[options[answer_index].option_id],
            metadata={
                "source_path": str(self.source_path),
                "source_index": sample_index,
                "source_id": sample.get("index"),
                "answer_index": answer_index,
                "source_choice_count": len(option_entries),
            },
        )


def _option_id(index: int) -> str:
    chars: list[str] = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        chars.append(chr(ord("A") + remainder))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(chars))


def _required_text(sample: dict[str, Any], field_name: str, sample_index: int) -> str:
    value = sample.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"SCALR sample at source index {sample_index} is missing non-empty "
            f"'{field_name}'."
        )
    return value.strip()


def _parse_non_negative_int(value: str, field_name: str, sample_index: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"SCALR sample at source index {sample_index} has invalid integer value "
            f"for '{field_name}': {value!r}."
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"SCALR sample at source index {sample_index} has negative integer value "
            f"for '{field_name}': {parsed}."
        )
    return parsed
