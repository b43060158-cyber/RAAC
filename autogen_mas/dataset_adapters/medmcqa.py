from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Any

from autogen_mas.models import OptionRecord, QuestionRecord

from .base import DatasetAdapter, read_text_with_fallbacks


_OPTION_FIELDS = (
    ("A", "opa"),
    ("B", "opb"),
    ("C", "opc"),
    ("D", "opd"),
)


class MedMCQAAdapter(DatasetAdapter):
    dataset_name = "medmcqa"

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def load(self) -> list[QuestionRecord]:
        records: list[QuestionRecord] = []
        for sample_index, line in enumerate(io.StringIO(read_text_with_fallbacks(self.source_path))):
            if not line.strip():
                continue
            sample = json.loads(line)
            records.append(self._record_from_sample(sample_index, sample))
        return records

    def _record_from_sample(
        self,
        sample_index: int,
        sample: dict[str, Any],
    ) -> QuestionRecord:
        if "cop" not in sample:
            raise ValueError(
                "MedMCQA sample is missing the 'cop' answer field at "
                f"source index {sample_index} in {self.source_path}. "
                "This usually means the answer-hidden test split was used. "
                "Use a labeled split such as data/medmcqa/dev.json or "
                "data/medmcqa/train.json for run-dataset/evaluation."
            )
        correct_position = int(sample["cop"])
        if correct_position < 1 or correct_position > len(_OPTION_FIELDS):
            raise ValueError(
                f"Invalid MedMCQA cop value at source index {sample_index}: {correct_position}"
            )

        options = [
            OptionRecord(option_id=option_id, text=str(sample[field_name]))
            for option_id, field_name in _OPTION_FIELDS
        ]
        correct_option_id = options[correct_position - 1].option_id
        return QuestionRecord(
            question_id=f"{sample_index:04d}",
            dataset_name=self.dataset_name,
            task_type="single_choice",
            question=str(sample["question"]),
            options=options,
            correct_option_ids=[correct_option_id],
            metadata={
                "source_path": str(self.source_path),
                "source_index": sample_index,
                "source_id": sample.get("id"),
                "cop": correct_position,
                "choice_type": sample.get("choice_type"),
                "subject_name": sample.get("subject_name"),
                "topic_name": sample.get("topic_name"),
                "explanation": sample.get("exp"),
            },
        )
