from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from autogen_mas.models import OptionRecord, QuestionRecord

from .base import DatasetAdapter, read_text_with_fallbacks


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


class TruthfulQAAdapter(DatasetAdapter):
    dataset_name = "truthfulqa"
    _TASK_TYPE = "single_choice"
    _TASK_KEY = "mc1_targets"

    def __init__(
        self,
        source_path: str | Path,
        task_types: Iterable[str] | None = None,
    ) -> None:
        self.source_path = Path(source_path)
        self.task_types = tuple(task_types or (self._TASK_TYPE,))
        unsupported = set(self.task_types) - {self._TASK_TYPE}
        if unsupported:
            requested = ", ".join(sorted(unsupported))
            raise ValueError(
                f"Unsupported TruthfulQA task type(s): {requested}. "
                f"TruthfulQA only supports {self._TASK_TYPE}."
            )

    def load(self) -> list[QuestionRecord]:
        data = json.loads(read_text_with_fallbacks(self.source_path))
        records: list[QuestionRecord] = []
        for sample_index, sample in enumerate(data):
            question_id = f"{sample_index:04d}"
            question = sample["question"]
            targets: dict[str, int] = sample[self._TASK_KEY]
            options: list[OptionRecord] = []
            correct_option_ids: list[str] = []
            for option_index, (option_text, label) in enumerate(targets.items()):
                option = OptionRecord(option_id=_option_id(option_index), text=option_text)
                options.append(option)
                if int(label) == 1:
                    correct_option_ids.append(option.option_id)
            if len(correct_option_ids) != 1:
                raise ValueError(
                    f"TruthfulQA single_choice sample {sample_index} must have exactly "
                    f"one correct option in {self._TASK_KEY}."
                )
            records.append(
                QuestionRecord(
                    question_id=question_id,
                    dataset_name=self.dataset_name,
                    task_type=self._TASK_TYPE,
                    question=question,
                    options=options,
                    correct_option_ids=correct_option_ids,
                    metadata={
                        "source_path": str(self.source_path),
                        "source_index": sample_index,
                        "source_task_key": self._TASK_KEY,
                    },
                )
            )
        return records
