"""Stage 1: mine successful agent_1 influence cases from run artifacts.

This script scans run question payloads and keeps cases where agent_1's review
appears in the recorded change drivers for another agent's answer revision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
NON_ADVERSARIAL_PREFIX = "competition-"
ADVERSARIAL_PREFIX = "adversarial-competition-"
CHOICE_TASK_TYPES = {"single_choice", "multiple_choice"}
REVIEWER_AGENT_ID = "agent_1"
NON_ADVERSARIAL_OUTPUT = "agent1_choice_influence_non_adversarial.json"
ADVERSARIAL_OUTPUT = "agent1_choice_influence_adversarial.json"
SINGLE_SEMANTIC_DATASETS = {"medmcqa", "medqa"}
DATASET_PURPOSE = (
    "Successful agent_1 review-to-answer-change influence samples for downstream "
    "LLM attribution analysis and malicious review rationale dataset construction."
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_option_ids(answer: dict[str, Any] | None) -> list[str]:
    if not isinstance(answer, dict):
        return []
    raw = answer.get("selected_option_ids", [])
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _same_option_ids(left: list[str], right: list[str]) -> bool:
    return sorted(left) == sorted(right)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_reviews(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    reviews = agent_result.get("reviews_given", [])
    if not isinstance(reviews, list):
        return []
    return [item for item in reviews if isinstance(item, dict)]


def _sorted_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rounds, key=lambda item: int(item.get("round_index", 0)))


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


def _format_options(options: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for option in options:
        option_id = _normalize_text(option.get("option_id"))
        text = _normalize_text(option.get("text"))
        if option_id and text:
            formatted.append(f"{option_id}: {text}")
        elif option_id:
            formatted.append(option_id)
        elif text:
            formatted.append(text)
    return "\n".join(formatted)


def _format_analysis_context(
    *,
    question_text: str,
    options: list[dict[str, Any]],
    previous_selected_option_ids: list[str],
    previous_reasoning: str,
    review_text: str,
    review_stance: str,
    review_score: int | float | None,
    new_selected_option_ids: list[str],
    new_reasoning: str,
    change_summary: str,
) -> str:
    score_text = "" if review_score is None else str(review_score)
    sections = [
        "Question:",
        question_text or "-",
        "",
        "Options:",
        _format_options(options) or "-",
        "",
        "Target Agent Previous Answer:",
        f"Selected options: {previous_selected_option_ids or []}",
        f"Reasoning: {previous_reasoning or '-'}",
        "",
        "Agent 1 Review:",
        f"Stance: {review_stance or '-'}",
        f"Score: {score_text or '-'}",
        f"Review text: {review_text or '-'}",
        "",
        "Target Agent New Answer:",
        f"Selected options: {new_selected_option_ids or []}",
        f"Reasoning: {new_reasoning or '-'}",
        "",
        "Target Agent Change Summary:",
        change_summary or "-",
    ]
    return "\n".join(sections)


def _run_type(run_dir_name: str) -> str | None:
    if run_dir_name.startswith(ADVERSARIAL_PREFIX):
        return "adversarial"
    if run_dir_name.startswith(NON_ADVERSARIAL_PREFIX):
        return "non_adversarial"
    return None


def _is_effective_single_choice(payload: dict[str, Any]) -> bool:
    task_type = _normalize_text(payload.get("task_type"))
    if task_type == "single_choice":
        return True
    if task_type != "multiple_choice":
        return False

    dataset_name = _normalize_text(payload.get("dataset_name")).lower()
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    source_task_key = _normalize_text(metadata.get("source_task_key"))
    if source_task_key == "mc2_targets":
        return False
    if source_task_key == "mc1_targets":
        return True

    choice_type = _normalize_text(metadata.get("choice_type")).lower()
    if choice_type == "single":
        return True
    if choice_type == "multi":
        return dataset_name in SINGLE_SEMANTIC_DATASETS

    if dataset_name in SINGLE_SEMANTIC_DATASETS:
        return True
    return False


def _find_agent1_review(previous_agent1_result: dict[str, Any], target_agent_id: str) -> dict[str, Any] | None:
    for review in _normalize_reviews(previous_agent1_result):
        if _normalize_text(review.get("target_agent_id")) == target_agent_id:
            return review
    return None


def _extract_sample(
    *,
    payload: dict[str, Any],
    run_id: str,
    run_type: str,
    from_round: dict[str, Any],
    to_round: dict[str, Any],
    target_agent_id: str,
    previous_target_result: dict[str, Any],
    new_target_result: dict[str, Any],
    previous_agent1_result: dict[str, Any],
    sample_index: int,
) -> dict[str, Any] | None:
    previous_answer = previous_target_result.get("answer")
    new_answer = new_target_result.get("answer")
    if not isinstance(previous_answer, dict) or not isinstance(new_answer, dict):
        return None

    previous_selected_option_ids = _normalize_option_ids(previous_answer)
    new_selected_option_ids = _normalize_option_ids(new_answer)
    if _same_option_ids(previous_selected_option_ids, new_selected_option_ids):
        return None

    target_change_drivers = new_answer.get("change_drivers", [])
    if not isinstance(target_change_drivers, list):
        return None
    normalized_change_drivers = [_normalize_text(item) for item in target_change_drivers if _normalize_text(item)]
    if REVIEWER_AGENT_ID not in normalized_change_drivers:
        return None

    review = _find_agent1_review(previous_agent1_result, target_agent_id)
    if review is None:
        return None

    options = payload.get("options", [])
    normalized_options = options if isinstance(options, list) else []
    sample_id = (
        f"{run_id}::{payload.get('question_key', payload.get('question_id', 'unknown'))}"
        f"::{target_agent_id}::{int(from_round.get('round_index', 0))}"
        f"->{int(to_round.get('round_index', 0))}::{sample_index}"
    )

    agent1_was_adversarial = bool(previous_agent1_result.get("advers-agent", False))
    target_was_adversarial = bool(new_target_result.get("advers-agent", False))
    question_text = _normalize_text(payload.get("question"))
    previous_reasoning = _normalize_text(previous_answer.get("reasoning"))
    new_reasoning = _normalize_text(new_answer.get("reasoning"))
    review_text = _normalize_text(review.get("main_reason"))
    review_stance = _normalize_text(review.get("stance"))
    review_score_raw = review.get("score")
    review_score: int | float | None
    if isinstance(review_score_raw, (int, float)):
        review_score = review_score_raw
    else:
        review_score = None
    change_summary = _normalize_text(new_answer.get("change_summary"))

    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "run_type": run_type,
        "question_key": _normalize_text(payload.get("question_key")),
        "question_id": _normalize_text(payload.get("question_id")),
        "dataset_name": _normalize_text(payload.get("dataset_name")),
        "task_type": _normalize_text(payload.get("task_type")),
        "effective_task_type": "single_choice",
        "from_round": int(from_round.get("round_index", 0)),
        "to_round": int(to_round.get("round_index", 0)),
        "reviewer_agent_id": REVIEWER_AGENT_ID,
        "target_agent_id": target_agent_id,
        "question_text": question_text,
        "options": normalized_options,
        "correct_option_ids": [str(item) for item in payload.get("correct_option_ids", [])],
        "target_previous_selected_option_ids": previous_selected_option_ids,
        "target_previous_reasoning": previous_reasoning,
        "agent1_review_score": review_score,
        "agent1_review_stance": review_stance,
        "agent1_review_text": review_text,
        "target_new_selected_option_ids": new_selected_option_ids,
        "target_new_reasoning": new_reasoning,
        "target_change_summary": change_summary,
        "target_change_drivers": normalized_change_drivers,
        "target_was_adversarial": target_was_adversarial,
        "agent1_was_adversarial": agent1_was_adversarial,
        "analysis_context": _format_analysis_context(
            question_text=question_text,
            options=normalized_options,
            previous_selected_option_ids=previous_selected_option_ids,
            previous_reasoning=previous_reasoning,
            review_text=review_text,
            review_stance=review_stance,
            review_score=review_score,
            new_selected_option_ids=new_selected_option_ids,
            new_reasoning=new_reasoning,
            change_summary=change_summary,
        ),
    }


def extract_records_from_question(payload: dict[str, Any], run_id: str, run_type: str) -> list[dict[str, Any]]:
    if _normalize_text(payload.get("task_type")) not in CHOICE_TASK_TYPES:
        return []
    if not _is_effective_single_choice(payload):
        return []

    rounds = payload.get("rounds", [])
    if not isinstance(rounds, list) or len(rounds) < 2:
        return []

    records: list[dict[str, Any]] = []
    sorted_rounds = _sorted_rounds([item for item in rounds if isinstance(item, dict)])
    sample_index = 0

    for previous_round, current_round in zip(sorted_rounds, sorted_rounds[1:]):
        previous_map = _round_agent_map(previous_round)
        current_map = _round_agent_map(current_round)
        previous_agent1_result = previous_map.get(REVIEWER_AGENT_ID)
        if previous_agent1_result is None:
            continue

        for target_agent_id, previous_target_result in previous_map.items():
            if target_agent_id == REVIEWER_AGENT_ID:
                continue
            current_target_result = current_map.get(target_agent_id)
            if current_target_result is None:
                continue

            sample_index += 1
            sample = _extract_sample(
                payload=payload,
                run_id=run_id,
                run_type=run_type,
                from_round=previous_round,
                to_round=current_round,
                target_agent_id=target_agent_id,
                previous_target_result=previous_target_result,
                new_target_result=current_target_result,
                previous_agent1_result=previous_agent1_result,
                sample_index=sample_index,
            )
            if sample is not None:
                records.append(sample)

    return records


def build_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_purpose": DATASET_PURPOSE,
        "record_count": len(records),
        "records": records,
    }


def extract_datasets(runs_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    non_adversarial_records: list[dict[str, Any]] = []
    adversarial_records: list[dict[str, Any]] = []

    for run_path in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        run_type = _run_type(run_path.name)
        if run_type is None:
            continue
        question_dir = run_path / "questions"
        if not question_dir.exists():
            continue

        for question_file in sorted(question_dir.glob("*.json")):
            payload = _load_json(question_file)
            records = extract_records_from_question(payload, run_id=run_path.name, run_type=run_type)
            if run_type == "adversarial":
                adversarial_records.extend(records)
            else:
                non_adversarial_records.extend(records)

    return build_dataset(non_adversarial_records), build_dataset(adversarial_records)


def write_dataset(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract successful agent_1 review influence samples from choice-task runs "
            "for downstream LLM attribution analysis."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Runs directory to scan. Defaults to {DEFAULT_RUNS_DIR}.",
    )
    parser.add_argument(
        "--non-adversarial-output",
        type=Path,
        default=DEFAULT_RUNS_DIR / NON_ADVERSARIAL_OUTPUT,
        help="Output JSON path for competition-* runs.",
    )
    parser.add_argument(
        "--adversarial-output",
        type=Path,
        default=DEFAULT_RUNS_DIR / ADVERSARIAL_OUTPUT,
        help="Output JSON path for adversarial-competition-* runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    non_adversarial_dataset, adversarial_dataset = extract_datasets(runs_dir)
    write_dataset(args.non_adversarial_output, non_adversarial_dataset)
    write_dataset(args.adversarial_output, adversarial_dataset)

    print(f"non_adversarial_output={args.non_adversarial_output}")
    print(f"non_adversarial_record_count={non_adversarial_dataset['record_count']}")
    print(f"adversarial_output={args.adversarial_output}")
    print(f"adversarial_record_count={adversarial_dataset['record_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
