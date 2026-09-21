from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _normalize_option_ids(answer: dict[str, Any] | None) -> list[str]:
    if not isinstance(answer, dict):
        return []
    raw = answer.get("selected_option_ids", [])
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _same_option_ids(left: list[str], right: list[str]) -> bool:
    return sorted(left) == sorted(right)


def _parse_sample_id(sample_id: str) -> dict[str, object] | None:
    parts = sample_id.split("::")
    if len(parts) < 5:
        return None
    run_id = parts[0].strip()
    question_key = parts[1].strip()
    target_agent_id = parts[2].strip()
    round_pair = parts[3].strip()
    if "->" not in round_pair:
        return None
    left, right = round_pair.split("->", 1)
    try:
        from_round = int(left)
        to_round = int(right)
    except ValueError:
        return None
    return {
        "run_id": run_id,
        "question_key": question_key,
        "target_agent_id": target_agent_id,
        "from_round": from_round,
        "to_round": to_round,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _load_question_payload(run_path: Path, question_key: str) -> dict[str, Any] | None:
    question_path = run_path / "questions" / f"{question_key}.json"
    if not question_path.exists():
        return None
    return json.loads(question_path.read_text(encoding="utf-8"))


def _round_agent_map(round_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for agent_result in round_payload.get("agent_results", []):
        if not isinstance(agent_result, dict):
            continue
        agent_id = _normalize_text(agent_result.get("agent_id"))
        answer = agent_result.get("answer")
        if not agent_id and isinstance(answer, dict):
            agent_id = _normalize_text(answer.get("agent_id"))
        if agent_id:
            result[agent_id] = agent_result
    return result


def _find_round(rounds: list[dict[str, Any]], round_index: int) -> dict[str, Any] | None:
    for round_payload in rounds:
        try:
            current = int(round_payload.get("round_index", 0))
        except (TypeError, ValueError):
            continue
        if current == round_index:
            return round_payload
    return None


def _agent1_review_exists(previous_agent1_result: dict[str, Any], target_agent_id: str) -> bool:
    reviews = previous_agent1_result.get("reviews_given", [])
    if not isinstance(reviews, list):
        return False
    for review in reviews:
        if not isinstance(review, dict):
            continue
        if _normalize_text(review.get("target_agent_id")) == target_agent_id:
            return True
    return False


def validate_sample(record: dict[str, Any], run_roots: list[Path]) -> str | None:
    sample_id = _normalize_text(record.get("sample_id"))
    parsed = _parse_sample_id(sample_id)
    if parsed is None:
        return "invalid_sample_id"

    run_id = str(parsed["run_id"])
    question_key = str(parsed["question_key"])
    target_agent_id = str(parsed["target_agent_id"])
    from_round = int(parsed["from_round"])
    to_round = int(parsed["to_round"])

    run_path = next((root for root in run_roots if root.name == run_id), None)
    if run_path is None:
        return "run_not_found"

    payload = _load_question_payload(run_path, question_key)
    if payload is None:
        return "question_file_not_found"

    rounds = payload.get("rounds", [])
    if not isinstance(rounds, list):
        return "rounds_missing"

    previous_round = _find_round([item for item in rounds if isinstance(item, dict)], from_round)
    current_round = _find_round([item for item in rounds if isinstance(item, dict)], to_round)
    if previous_round is None or current_round is None:
        return "round_not_found"

    previous_map = _round_agent_map(previous_round)
    current_map = _round_agent_map(current_round)
    previous_agent1_result = previous_map.get("agent_1")
    previous_target_result = previous_map.get(target_agent_id)
    current_target_result = current_map.get(target_agent_id)
    if previous_agent1_result is None:
        return "agent1_missing_in_from_round"
    if previous_target_result is None or current_target_result is None:
        return "target_missing_in_rounds"
    if not _agent1_review_exists(previous_agent1_result, target_agent_id):
        return "agent1_review_missing"

    previous_answer = previous_target_result.get("answer")
    current_answer = current_target_result.get("answer")
    previous_selected_option_ids = _normalize_option_ids(previous_answer if isinstance(previous_answer, dict) else None)
    current_selected_option_ids = _normalize_option_ids(current_answer if isinstance(current_answer, dict) else None)
    if _same_option_ids(previous_selected_option_ids, current_selected_option_ids):
        return "no_option_change"

    if not isinstance(current_answer, dict):
        return "target_answer_missing"
    change_drivers = current_answer.get("change_drivers", [])
    if not isinstance(change_drivers, list):
        return "change_drivers_missing"
    normalized_change_drivers = [_normalize_text(item) for item in change_drivers if _normalize_text(item)]
    if "agent_1" not in normalized_change_drivers:
        return "agent1_not_in_change_drivers"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether each reasoning-bank sample is backed by an original run trace "
            "where agent_1 reviewed the target and the target changed options in the next round."
        )
    )
    parser.add_argument("--bank", type=Path, required=True, help="Reasoning-bank JSONL to validate.")
    parser.add_argument(
        "--run-paths",
        nargs="+",
        type=Path,
        required=True,
        help="One or more run directories containing the original questions/*.json files.",
    )
    parser.add_argument(
        "--print-reasons",
        action="store_true",
        help="Print the validation failure reason next to each invalid sample_id.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_roots = [path.expanduser().resolve() for path in args.run_paths]
    for run_root in run_roots:
        if not run_root.exists():
            raise FileNotFoundError(f"Run path not found: {run_root}")
    records = _load_jsonl(args.bank.expanduser().resolve())

    invalid_count = 0
    for record in records:
        failure_reason = validate_sample(record, run_roots)
        if failure_reason is None:
            continue
        invalid_count += 1
        sample_id = _normalize_text(record.get("sample_id"))
        if args.print_reasons:
            print(f"{sample_id}\t{failure_reason}")
        else:
            print(sample_id)

    print(f"checked_count={len(records)}")
    print(f"invalid_count={invalid_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
