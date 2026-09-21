from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autogen_mas.models import OptionRecord, QuestionRecord

from .base import DatasetAdapter, read_text_with_fallbacks


_LABEL_TO_OPTION_ID = {
    "CHATGPT": "A",
    "TIE": "B",
    "VICUNA13B": "C",
}

class FairEvalAdapter(DatasetAdapter):
    dataset_name = "faireval"

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)
        self.answer_key_path = self.source_path.with_name("review_gpt35_vicuna-13b_human.txt")

    def load(self) -> list[QuestionRecord]:
        samples = json.loads(read_text_with_fallbacks(self.source_path))
        if not isinstance(samples, list):
            raise ValueError(f"FairEval dataset must be a JSON array: {self.source_path}")

        answer_labels = self._load_answer_labels()
        if len(answer_labels) != len(samples):
            raise ValueError(
                "FairEval answer key line count does not match dataset size: "
                f"{len(answer_labels)} labels vs {len(samples)} samples."
            )

        records: list[QuestionRecord] = []
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(
                    f"FairEval sample at source index {sample_index} must be an object."
                )
            records.append(
                self._record_from_sample(
                    sample_index=sample_index,
                    sample=sample,
                    answer_label=answer_labels[sample_index],
                )
            )
        return records

    def _load_answer_labels(self) -> list[str]:
        labels: list[str] = []
        for line_index, line in enumerate(
            read_text_with_fallbacks(self.answer_key_path).splitlines(),
            start=1,
        ):
            label = line.strip()
            if not label:
                continue
            if label not in _LABEL_TO_OPTION_ID:
                supported = ", ".join(sorted(_LABEL_TO_OPTION_ID))
                raise ValueError(
                    f"FairEval answer key line {line_index} has unsupported label "
                    f"{label!r}. Supported labels: {supported}."
                )
            labels.append(label)
        return labels

    def _record_from_sample(
        self,
        *,
        sample_index: int,
        sample: dict[str, Any],
        answer_label: str,
    ) -> QuestionRecord:
        question = _required_text(sample, "question", sample_index)
        responses = sample.get("response")
        if not isinstance(responses, dict):
            raise ValueError(
                f"FairEval sample at source index {sample_index} is missing object 'response'."
            )
        response_1_text = _required_text(responses, "gpt35", sample_index)
        response_2_text = _required_text(responses, "vicuna", sample_index)

        source_question_id = sample.get("question_id")
        if source_question_id is None:
            raise ValueError(
                f"FairEval sample at source index {sample_index} is missing 'question_id'."
            )

        return QuestionRecord(
            question_id=f"{sample_index:04d}",
            dataset_name=self.dataset_name,
            task_type="single_choice",
            question=question,
            options=[
                OptionRecord(option_id="A", text=response_1_text),
                OptionRecord(option_id="B", text="A，C都对"),
                OptionRecord(option_id="C", text=response_2_text),
            ],
            correct_option_ids=[_LABEL_TO_OPTION_ID[answer_label]],
            metadata={
                "source_path": str(self.source_path),
                "answer_key_path": str(self.answer_key_path),
                "source_index": sample_index,
                "source_question_id": source_question_id,
                "category": sample.get("category"),
                "response_1_text": response_1_text,
                "response_2_text": response_2_text,
                "response_1_source": "gpt35",
                "response_2_source": "vicuna",
                "label_mapping": dict(_LABEL_TO_OPTION_ID),
            },
        )


def _required_text(sample: dict[str, Any], field_name: str, sample_index: int) -> str:
    value = sample.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"FairEval sample at source index {sample_index} is missing non-empty "
            f"'{field_name}'."
        )
    return value.strip()
