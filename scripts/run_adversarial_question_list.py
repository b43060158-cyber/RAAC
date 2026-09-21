from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autogen_mas.cli import (
    _add_adversarial_overrides,
    _build_question_progress,
    _emit_adversarial_model_plan,
    _load_questions,
    _persist_feedback_summary_enabled,
    _resolve_dataset_path,
    _with_adversarial_overrides,
    _with_max_workers,
    _with_question_selection,
)
from autogen_mas.config import QuestionSelectionConfig, load_settings
from autogen_mas.evaluation import Evaluator
from autogen_mas.runtime import AdversarialCompetitionRunner


def _split_tokens(values: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            token = part.strip()
            if token:
                tokens.append(token)
    return tokens


def parse_question_indices(raw_values: Iterable[str]) -> list[int]:
    tokens = _split_tokens(raw_values)
    if not tokens:
        raise ValueError("Please provide at least one question index.")

    indices: list[int] = []
    for token in tokens:
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid question index '{token}'. Expected integers.") from exc
        if index < 0:
            raise ValueError(f"Question index must be non-negative: {index}")
        indices.append(index)

    duplicates = sorted({index for index in indices if indices.count(index) > 1})
    if duplicates:
        raise ValueError(f"Duplicate question indices are not allowed: {duplicates}")
    return indices


def load_question_indices(args: argparse.Namespace) -> list[int]:
    raw_values = list(args.question_indices or [])
    if args.question_index_file:
        raw_values.append(Path(args.question_index_file).read_text(encoding="utf-8"))
    return parse_question_indices(raw_values)


def select_questions_by_index(questions: list[object], indices: list[int]) -> list[object]:
    if not questions:
        raise ValueError("Dataset is empty; no questions are available to run.")
    max_index = len(questions) - 1
    invalid = [index for index in indices if index > max_index]
    if invalid:
        raise ValueError(
            f"Question indices out of range for dataset of size {len(questions)}: {invalid}"
        )
    return [questions[index] for index in indices]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a specific list of questions with the adversarial LLM-MAS runner "
            "and automatically evaluate the resulting run."
        )
    )
    parser.add_argument("--config", default="config/agents.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dataset", default="medmcqa")
    parser.add_argument("--dataset-path")
    parser.add_argument(
        "--question-indices",
        nargs="*",
        default=[],
        help="Question indices to run. Accepts space-separated and/or comma-separated values.",
    )
    parser.add_argument(
        "--question-index-file",
        help="Optional text file containing question indices separated by commas or whitespace.",
    )
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--resume-run-path")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--persist-feedback-summary", action="store_true")
    parser.add_argument(
        "--selection-rule",
        choices=["top-agent"],
        default="top-agent",
    )
    _add_adversarial_overrides(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    persist_feedback_summary = _persist_feedback_summary_enabled(args, parser)

    try:
        question_indices = load_question_indices(args)
        experiment_config, llm_settings = load_settings(args.config, args.env_file)
        experiment_config = _with_adversarial_overrides(experiment_config, args)
        dataset_path = _resolve_dataset_path(
            experiment_config=experiment_config,
            dataset_name=args.dataset,
            explicit_path=args.dataset_path,
        )
        questions = _load_questions(args.dataset, dataset_path)
        selected_questions = select_questions_by_index(questions, question_indices)
    except ValueError as exc:
        parser.error(str(exc))

    selection = QuestionSelectionConfig(
        limit=len(selected_questions),
        seed=None,
        strategy=f"explicit_indices:{','.join(str(index) for index in question_indices)}",
    )
    experiment_config = _with_question_selection(experiment_config, selection)
    experiment_config = _with_max_workers(experiment_config, args.max_workers)

    _emit_adversarial_model_plan(experiment_config, llm_settings)
    runner = AdversarialCompetitionRunner(
        experiment_config=experiment_config,
        llm_settings=llm_settings,
        debug_mode=args.debug,
        persist_feedback_summary=persist_feedback_summary,
    )
    progress_callback, close_progress = _build_question_progress(
        "Adversarial LLM-MAS",
        len(selected_questions),
    )
    try:
        artifacts = runner.run_dataset(
            selected_questions,
            progress_callback=progress_callback,
            resume_run_path=args.resume_run_path,
        )
    finally:
        close_progress()

    report = Evaluator().evaluate(
        artifacts.run_path,
        selection_rule=args.selection_rule,
    )
    summary = {
        "run_path": artifacts.run_path,
        "dataset": args.dataset,
        "dataset_path": dataset_path,
        "question_indices": question_indices,
        "question_keys": [question.question_key for question in selected_questions],
        "debug": bool(args.debug),
        "attack_strategy": experiment_config.adversarial_mas.adversarial_agent.choice_attack_strategy,
        **report.summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
