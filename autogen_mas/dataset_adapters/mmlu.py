from __future__ import annotations

import csv
import io
from pathlib import Path

from autogen_mas.models import OptionRecord, QuestionRecord

from .base import DatasetAdapter, read_text_with_fallbacks


_OPTION_IDS = ("A", "B", "C", "D")
_HEADER_ROW = ("question", "a", "b", "c", "d", "answer")


class MMLUAdapter(DatasetAdapter):
    dataset_name = "mmlu"

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def load(self) -> list[QuestionRecord]:
        if self.source_path.is_dir():
            return self._load_directory(self.source_path)
        return self._load_file(self.source_path)

    def _load_directory(self, directory: Path) -> list[QuestionRecord]:
        """Aggregate every CSV under ``directory`` (recursively) into one pool.

        All questions across the subject files are combined and assigned
        globally-unique ``question_id`` values so they can be sampled together
        as a single dataset.
        """
        csv_paths = sorted(directory.rglob("*.csv"))
        if not csv_paths:
            raise ValueError(f"No CSV files found under MMLU directory {directory}.")

        records: list[QuestionRecord] = []
        for csv_path in csv_paths:
            records.extend(self._load_file(csv_path, start_index=len(records)))
        return records

    def _load_file(self, source_path: Path, start_index: int = 0) -> list[QuestionRecord]:
        records: list[QuestionRecord] = []
        reader = csv.reader(io.StringIO(read_text_with_fallbacks(source_path), newline=""))
        for sample_index, row in enumerate(reader):
            if not row or not any(cell.strip() for cell in row):
                continue
            if sample_index == 0 and self._is_header_row(row):
                continue
            records.append(
                self._record_from_row(
                    sample_index,
                    row,
                    source_path=source_path,
                    question_index=start_index + len(records),
                )
            )
        return records

    def _record_from_row(
        self,
        sample_index: int,
        row: list[str],
        *,
        source_path: Path,
        question_index: int,
    ) -> QuestionRecord:
        if len(row) != 6:
            raise ValueError(
                f"MMLU sample at source index {sample_index} in {source_path} must have "
                f"exactly 6 columns (question, A, B, C, D, answer); found {len(row)}."
            )

        question, *option_texts, answer = (cell.strip() for cell in row)
        if not question:
            raise ValueError(
                f"MMLU sample at source index {sample_index} in {source_path} is missing "
                f"a question."
            )

        normalized_answer = answer.upper()
        if normalized_answer not in _OPTION_IDS:
            supported = ", ".join(_OPTION_IDS)
            raise ValueError(
                f"MMLU sample at source index {sample_index} in {source_path} has invalid "
                f"answer {answer!r}. Supported labels: {supported}."
            )

        options = [
            OptionRecord(option_id=option_id, text=option_text)
            for option_id, option_text in zip(_OPTION_IDS, option_texts, strict=True)
        ]
        return QuestionRecord(
            question_id=f"{question_index:04d}",
            dataset_name=self.dataset_name,
            task_type="single_choice",
            question=question,
            options=options,
            correct_option_ids=[normalized_answer],
            metadata={
                "source_path": str(source_path),
                "source_index": sample_index,
                "subject": source_path.stem,
            },
        )

    def _is_header_row(self, row: list[str]) -> bool:
        normalized = tuple(cell.strip().lower() for cell in row)
        return normalized == _HEADER_ROW
