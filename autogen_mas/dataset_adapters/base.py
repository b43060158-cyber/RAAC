from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from autogen_mas.models import TaskRecord

_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030")


class DatasetAdapter(ABC):
    @abstractmethod
    def load(self) -> list[TaskRecord]:
        raise NotImplementedError


def read_text_with_fallbacks(path: str | Path) -> str:
    file_path = Path(path)
    last_error: UnicodeDecodeError | None = None
    for encoding in _TEXT_ENCODINGS:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    encodings = ", ".join(_TEXT_ENCODINGS)
    raise ValueError(
        f"Could not decode dataset file {file_path} with any supported encoding: {encodings}."
    ) from last_error
