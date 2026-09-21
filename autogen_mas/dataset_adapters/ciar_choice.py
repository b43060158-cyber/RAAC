from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from autogen_mas.models import OptionRecord, QuestionRecord

from .base import DatasetAdapter, read_text_with_fallbacks


class CIARChoiceAdapter(DatasetAdapter):
    """CIAR served as a binary single-choice task.

    Each CIAR sample carries a correct-answer alias cluster (``answer``) and a
    single incorrect "intuitive trap" cluster (``incorrect answer``). This
    adapter turns them into a two-option ``single_choice`` question so the
    adversarial pipeline routes through the choice attack strategy (strategy9)
    instead of the math-short-answer strategy (strategy11).

    Option text joins the whole alias cluster with " / " to mirror the format
    the reasoning bank was originally built with (e.g. "0.75 / 75% / 3/4"), so
    retrieval against the existing CIAR corpus stays aligned. The correct
    option position is randomized deterministically per sample to avoid a fixed
    A/B label bias.
    """

    dataset_name = "ciar_choice"

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def load(self) -> list[QuestionRecord]:
        samples = json.loads(read_text_with_fallbacks(self.source_path))
        if not isinstance(samples, list):
            raise ValueError(f"CIAR dataset must be a JSON array: {self.source_path}")

        records: list[QuestionRecord] = []
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
    ) -> QuestionRecord:
        question = _required_text(sample, "question", sample_index)
        correct_aliases = _required_text_list(sample, "answer", sample_index)
        incorrect_aliases = _required_text_list(sample, "incorrect answer", sample_index)

        correct_text = " / ".join(correct_aliases)
        incorrect_text = " / ".join(incorrect_aliases)

        # Deterministic per-sample position so reruns are reproducible while
        # the correct answer is not always option A.
        rng = random.Random(f"{self.dataset_name}:{sample_index:04d}")
        correct_first = rng.random() < 0.5
        if correct_first:
            options = [
                OptionRecord(option_id="A", text=correct_text),
                OptionRecord(option_id="B", text=incorrect_text),
            ]
            correct_option_id = "A"
        else:
            options = [
                OptionRecord(option_id="A", text=incorrect_text),
                OptionRecord(option_id="B", text=correct_text),
            ]
            correct_option_id = "B"

        return QuestionRecord(
            question_id=f"{sample_index:04d}",
            dataset_name=self.dataset_name,
            task_type="single_choice",
            question=question,
            options=options,
            correct_option_ids=[correct_option_id],
            metadata={
                "source_path": str(self.source_path),
                "source_index": sample_index,
                "correct_answer_aliases": correct_aliases,
                "incorrect_answer_aliases": incorrect_aliases,
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
