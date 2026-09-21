from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autogen_mas.config import DashScopeSettings, load_llm_settings
from autogen_mas.runtime.clients import StructuredLLMClient, build_structured_client


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            text = line.strip()
            if not text:
                continue
            records.append(json.loads(text))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(record, ensure_ascii=False) + "\n")


def unique_by_sample_id(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            continue
        deduped[sample_id] = record
    return list(deduped.values())


def chunked(records: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    return [records[index : index + size] for index in range(0, len(records), size)]


def load_deepseek_client(env_path: str | Path = ".env") -> tuple[DashScopeSettings, StructuredLLMClient]:
    settings = load_llm_settings(env_path)
    settings.model = "deepseek-chat"
    settings.provider = "deepseek"
    client = build_structured_client(settings)
    return settings, client


def options_as_text(options: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for option in options:
        option_id = str(option.get("option_id", "")).strip()
        text = str(option.get("text", "")).strip()
        if option_id and text:
            lines.append(f"{option_id}. {text}")
        elif option_id:
            lines.append(option_id)
        elif text:
            lines.append(text)
    return "\n".join(lines)
