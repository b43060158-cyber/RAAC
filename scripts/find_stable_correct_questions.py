from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_AGENT_IDS = ("agent_2", "agent_3", "agent_4")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_option_ids(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _question_key_to_path(run_dir: Path, question_key: str) -> Path:
    return run_dir / "questions" / f"{question_key}.json"


def _agent_answer_by_round(question_payload: dict[str, Any], agent_id: str) -> list[tuple[str, ...]]:
    rounds = question_payload.get("rounds", [])
    if not isinstance(rounds, list):
        return []

    answers: list[tuple[str, ...]] = []
    for round_payload in rounds:
        if not isinstance(round_payload, dict):
            continue
        agent_results = round_payload.get("agent_results", [])
        if not isinstance(agent_results, list):
            continue
        for agent_result in agent_results:
            if not isinstance(agent_result, dict):
                continue
            current_agent_id = str(agent_result.get("agent_id", "")).strip()
            answer = agent_result.get("answer")
            if not current_agent_id and isinstance(answer, dict):
                current_agent_id = str(answer.get("agent_id", "")).strip()
            if current_agent_id != agent_id or not isinstance(answer, dict):
                continue
            answers.append(_normalize_option_ids(answer.get("selected_option_ids", [])))
            break
    return answers


def _is_consistently_identical_across_agents(
    question_payload: dict[str, Any],
    agent_ids: tuple[str, ...],
) -> bool:
    answer_sequences = {
        agent_id: _agent_answer_by_round(question_payload, agent_id) for agent_id in agent_ids
    }
    if any(not sequence for sequence in answer_sequences.values()):
        return False

    round_count = len(next(iter(answer_sequences.values())))
    if any(len(sequence) != round_count for sequence in answer_sequences.values()):
        return False

    baseline = answer_sequences[agent_ids[0]][0]
    if not baseline:
        return False

    for round_index in range(round_count):
        round_answers = [answer_sequences[agent_id][round_index] for agent_id in agent_ids]
        if any(answer != baseline for answer in round_answers):
            return False
    return True


def collect_successful_question_ids(run_dir: Path) -> list[str]:
    question_results = _load_json(run_dir / "evaluation" / "question_results.json")
    if not isinstance(question_results, list):
        raise ValueError("evaluation/question_results.json must be a JSON list")

    successful_ids: list[str] = []
    for result in question_results:
        if not isinstance(result, dict):
            continue
        if result.get("is_correct") is not True:
            continue
        question_id = str(result.get("question_id", "")).strip()
        if question_id:
            successful_ids.append(question_id)
    return sorted(set(successful_ids))


def collect_incorrect_question_ids(run_dir: Path) -> list[str]:
    question_results = _load_json(run_dir / "evaluation" / "question_results.json")
    if not isinstance(question_results, list):
        raise ValueError("evaluation/question_results.json must be a JSON list")

    incorrect_ids: list[str] = []
    for result in question_results:
        if not isinstance(result, dict):
            continue
        if result.get("is_correct") is not False:
            continue
        question_id = str(result.get("question_id", "")).strip()
        if question_id:
            incorrect_ids.append(question_id)
    return sorted(set(incorrect_ids))


def collect_stable_correct_question_ids(
    run_dir: Path,
    agent_ids: tuple[str, ...] = DEFAULT_AGENT_IDS,
) -> list[str]:
    question_results = _load_json(run_dir / "evaluation" / "question_results.json")
    if not isinstance(question_results, list):
        raise ValueError("evaluation/question_results.json must be a JSON list")

    collected_ids: list[str] = []
    for result in question_results:
        if not isinstance(result, dict):
            continue
        if result.get("is_correct") is not True:
            continue

        question_key = str(result.get("question_key", "")).strip()
        question_id = str(result.get("question_id", "")).strip()
        if not question_key or not question_id:
            continue

        question_path = _question_key_to_path(run_dir, question_key)
        if not question_path.exists():
            continue
        question_payload = _load_json(question_path)
        if not isinstance(question_payload, dict):
            continue

        if _is_consistently_identical_across_agents(question_payload, agent_ids):
            collected_ids.append(question_id)
    return sorted(set(collected_ids))


def build_report(run_dir: Path, agent_ids: tuple[str, ...] = DEFAULT_AGENT_IDS) -> dict[str, Any]:
    successful_ids = collect_successful_question_ids(run_dir)
    incorrect_ids = collect_incorrect_question_ids(run_dir)
    collected_ids = collect_stable_correct_question_ids(run_dir, agent_ids=agent_ids)
    initial_difference_ids = sorted(set(successful_ids) - set(collected_ids))
    difference_ids = sorted(set(initial_difference_ids) - set(incorrect_ids))
    return {
        "run_dir": str(run_dir),
        "agent_ids": list(agent_ids),
        "successful_question_ids": successful_ids,
        "incorrect_question_ids": incorrect_ids,
        "collected_question_ids": collected_ids,
        "initial_difference_question_ids": initial_difference_ids,
        "difference_question_ids": difference_ids,
        "successful_count": len(successful_ids),
        "incorrect_count": len(incorrect_ids),
        "collected_count": len(collected_ids),
        "initial_difference_count": len(initial_difference_ids),
        "difference_count": len(difference_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect question IDs whose final evaluation is correct and whose agent_2/3/4 "
            "answers remain identical across all rounds, then subtract those IDs from all "
            "successful question IDs, and finally subtract the evaluated-incorrect question set."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Run directory, e.g. runs/adversarial-competition-20260526-104637")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path. Defaults to <run_dir>/stable_correct_questions.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "stable_correct_questions.json"
    )

    report = build_report(run_dir)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(f"report_path={output_path}")
    print(f"successful_count={report['successful_count']}")
    print(f"incorrect_count={report['incorrect_count']}")
    print(f"collected_count={report['collected_count']}")
    print(f"initial_difference_count={report['initial_difference_count']}")
    print(f"difference_count={report['difference_count']}")
    print("difference_question_ids=" + json.dumps(report["difference_question_ids"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
