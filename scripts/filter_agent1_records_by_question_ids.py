from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize_question_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isdigit():
        return text.zfill(4)
    return text


def _question_id_from_question_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = text.split("__")
    if len(parts) < 3:
        return ""
    question_id = parts[1].strip()
    if not question_id.isdigit():
        return ""
    return question_id.zfill(4)


def _record_question_id(record: dict[str, Any]) -> str:
    for key in ("question_id",):
        question_id = _normalize_question_id(record.get(key))
        if question_id:
            return question_id
    for key in ("question_key", "source_question_key"):
        question_id = _question_id_from_question_key(record.get(key))
        if question_id:
            return question_id
    sample_id = str(record.get("sample_id", "")).strip()
    if sample_id:
        parts = sample_id.split("::")
        if len(parts) >= 2:
            question_id = _question_id_from_question_key(parts[1])
            if question_id:
                return question_id
    return ""


def _record_run_id(record: dict[str, Any]) -> str:
    run_id = str(record.get("run_id", "")).strip()
    if run_id:
        return run_id
    sample_id = str(record.get("sample_id", "")).strip()
    if sample_id and "::" in sample_id:
        return sample_id.split("::", 1)[0].strip()
    return ""


def _load_payload(path: Path) -> tuple[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                records.append(payload)
        return "jsonl", records
    return "json", json.loads(text)


def _filter_records(
    records: list[dict[str, Any]],
    allowed_question_ids: set[str],
    allowed_run_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if isinstance(record, dict)
        and _record_question_id(record) in allowed_question_ids
        and (not allowed_run_ids or _record_run_id(record) in allowed_run_ids)
    ]


def filter_payload(payload: Any, allowed_question_ids: set[str], allowed_run_ids: set[str] | None = None) -> Any:
    run_ids = allowed_run_ids or set()
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        output = dict(payload)
        output["records"] = _filter_records(payload["records"], allowed_question_ids, run_ids)
        if "record_count" in output:
            output["record_count"] = len(output["records"])
        return output
    if isinstance(payload, list):
        return _filter_records(payload, allowed_question_ids, run_ids)
    raise ValueError("Unsupported payload format. Expected JSONL records, a JSON list, or a JSON object with a records field.")


def _write_payload(path: Path, file_format: str, payload: Any) -> None:
    if file_format == "jsonl":
        records = payload if isinstance(payload, list) else payload.get("records", [])
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter agent1 pipeline artifacts by question id whitelist."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--question-ids",
        nargs="+",
        required=True,
        help="Question ids to keep, e.g. 551 868 1328 1577 1885 3799.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=[],
        help="Optional run ids to keep. If omitted, keep matching question ids from all runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed_question_ids = {
        _normalize_question_id(question_id)
        for question_id in args.question_ids
        if _normalize_question_id(question_id)
    }
    allowed_run_ids = {str(run_id).strip() for run_id in args.run_ids if str(run_id).strip()}
    if not allowed_question_ids:
        raise ValueError("No valid question ids provided.")

    file_format, payload = _load_payload(args.input)
    filtered = filter_payload(payload, allowed_question_ids, allowed_run_ids)
    _write_payload(args.output, file_format, filtered)

    if isinstance(filtered, dict):
        record_count = len(filtered.get("records", []))
    else:
        record_count = len(filtered)
    print(f"output={args.output}")
    print(f"record_count={record_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
