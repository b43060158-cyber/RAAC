from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Any

from autogen_mas.models import CodeQuestionRecord

from .base import DatasetAdapter, read_text_with_fallbacks


class HumanEvalAdapter(DatasetAdapter):
    dataset_name = "humaneval"

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def load(self) -> list[CodeQuestionRecord]:
        records: list[CodeQuestionRecord] = []
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
    ) -> CodeQuestionRecord:
        source_task_id = str(sample["task_id"])
        question_id = source_task_id.split("/", 1)[-1]
        return CodeQuestionRecord(
            question_id=question_id,
            dataset_name=self.dataset_name,
            prompt=str(sample["prompt"]),
            entry_point=str(sample["entry_point"]),
            test=str(sample["test"]),
            source_path=str(self.source_path),
            source_task_id=source_task_id,
        )
