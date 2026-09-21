from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_accuracy_comparison import (
    DEFAULT_MODEL_ID,
    SummaryLoadError,
    default_multi_output_path,
    default_output_path,
    load_run_accuracy,
    main,
    plot_accuracy_comparison,
)


def _write_summary(run_path: Path, summary: dict) -> None:
    evaluation_dir = run_path / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def _write_question(run_path: Path, *, dataset_name: str = "ciar") -> None:
    question_dir = run_path / "questions"
    question_dir.mkdir(parents=True)
    (question_dir / "question.json").write_text(
        json.dumps({"dataset_name": dataset_name}),
        encoding="utf-8",
    )


class PlotAccuracyComparisonTest(unittest.TestCase):
    def test_loads_summaries_and_computes_accuracy_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            normal_path = root / "normal-run"
            adversarial_path = root / "adversarial-run"
            _write_summary(
                normal_path,
                {
                    "total_questions": 50,
                    "accuracy": 0.9,
                    "correct_questions": 45,
                    "evaluated_questions": 50,
                },
            )
            _write_question(normal_path, dataset_name="ciar")
            _write_summary(
                adversarial_path,
                {
                    "total_questions": 50,
                    "accuracy": 0.7,
                    "correct_questions": 35,
                    "evaluated_questions": 50,
                },
            )
            _write_question(adversarial_path, dataset_name="ciar")

            normal = load_run_accuracy(normal_path, label="Normal LLM-MAS")
            adversarial = load_run_accuracy(adversarial_path, label="Adversarial LLM-MAS")
            result = plot_accuracy_comparison(
                normal,
                adversarial,
                output=root / "comparison.png",
                title="Demo",
            )

            self.assertEqual(normal.correct_questions, 45)
            self.assertEqual(normal.dataset_names, ("ciar",))
            self.assertEqual(normal.total_questions, 50)
            self.assertEqual(adversarial.evaluated_questions, 50)
            self.assertEqual(result["dataset"], "ciar")
            self.assertEqual(result["model_id"], DEFAULT_MODEL_ID)
            self.assertEqual(result["total_questions"], "50")
            self.assertAlmostEqual(result["accuracy_drop_points"], 20.0)
            self.assertTrue((root / "comparison.png").exists())

    def test_missing_summary_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(SummaryLoadError, "Evaluation summary not found"):
                load_run_accuracy(Path(tmp_dir) / "missing-run", label="Normal LLM-MAS")

    def test_missing_accuracy_field_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_path = Path(tmp_dir) / "run"
            _write_summary(
                run_path,
                {
                    "total_questions": 2,
                    "correct_questions": 1,
                    "evaluated_questions": 2,
                },
            )
            _write_question(run_path)

            with self.assertRaisesRegex(SummaryLoadError, "Missing required field 'accuracy'"):
                load_run_accuracy(run_path, label="Normal LLM-MAS")

    def test_missing_evaluated_questions_falls_back_to_total_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_path = Path(tmp_dir) / "legacy-run"
            _write_summary(
                run_path,
                {
                    "total_questions": 100,
                    "accuracy": 0.66,
                    "correct_questions": 66,
                },
            )
            _write_question(run_path, dataset_name="medmcqa")

            accuracy = load_run_accuracy(run_path, label="Adversarial LLM-MAS")

            self.assertEqual(accuracy.evaluated_questions, 100)

    def test_output_argument_is_used_by_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            normal_path = root / "normal-run"
            adversarial_path = root / "adversarial-run"
            output_path = root / "custom-output.png"
            _write_summary(
                normal_path,
                {
                    "total_questions": 10,
                    "accuracy": 0.9,
                    "correct_questions": 9,
                    "evaluated_questions": 10,
                },
            )
            _write_question(normal_path, dataset_name="ciar")
            _write_summary(
                adversarial_path,
                {
                    "total_questions": 10,
                    "accuracy": 0.8,
                    "correct_questions": 8,
                    "evaluated_questions": 10,
                },
            )
            _write_question(adversarial_path, dataset_name="ciar")

            exit_code = main(
                [
                    str(normal_path),
                    str(adversarial_path),
                    "--output",
                    str(output_path),
                    "--title",
                    "Custom",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

    def test_cli_accepts_multiple_named_adversarial_strategies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            normal_path = root / "normal-run"
            strategy_a_path = root / "strategy-a-run"
            strategy_b_path = root / "strategy-b-run"
            output_path = root / "strategies.png"
            _write_summary(
                normal_path,
                {
                    "total_questions": 100,
                    "accuracy": 0.76,
                    "correct_questions": 76,
                    "evaluated_questions": 100,
                },
            )
            _write_question(normal_path, dataset_name="truthfulqa")
            _write_summary(
                strategy_a_path,
                {
                    "total_questions": 100,
                    "accuracy": 0.66,
                    "correct_questions": 66,
                    "evaluated_questions": 100,
                },
            )
            _write_question(strategy_a_path, dataset_name="truthfulqa")
            _write_summary(
                strategy_b_path,
                {
                    "total_questions": 100,
                    "accuracy": 0.61,
                    "correct_questions": 61,
                    "evaluated_questions": 100,
                },
            )
            _write_question(strategy_b_path, dataset_name="truthfulqa")

            exit_code = main(
                [
                    str(normal_path),
                    "--adversarial",
                    f"comment_lines={strategy_a_path}",
                    "--adversarial",
                    f"feedback_poisoning={strategy_b_path}",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

    def test_cli_accepts_custom_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            normal_path = root / "normal-run"
            adversarial_path = root / "adversarial-run"
            output_path = root / "custom-model.png"
            _run_with_summary(
                normal_path,
                accuracy=0.9,
                correct_questions=90,
                dataset_name="truthfulqa",
            )
            _run_with_summary(
                adversarial_path,
                accuracy=0.72,
                correct_questions=72,
                dataset_name="truthfulqa",
            )

            exit_code = main(
                [
                    str(normal_path),
                    "--adversarial",
                    f"strategy={adversarial_path}",
                    "--model-id",
                    "qwen-max",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

    def test_plot_returns_drops_for_multiple_strategies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            normal = load_run_accuracy(
                _run_with_summary(
                    root / "normal-run",
                    accuracy=0.8,
                    correct_questions=80,
                    dataset_name="medmcqa",
                ),
                label="Normal LLM-MAS",
            )
            strategy_a = load_run_accuracy(
                _run_with_summary(
                    root / "strategy-a",
                    accuracy=0.7,
                    correct_questions=70,
                    dataset_name="medmcqa",
                ),
                label="strategy_a",
            )
            strategy_b = load_run_accuracy(
                _run_with_summary(
                    root / "strategy-b",
                    accuracy=0.65,
                    correct_questions=65,
                    dataset_name="medmcqa",
                ),
                label="strategy_b",
            )

            result = plot_accuracy_comparison(
                normal,
                [strategy_a, strategy_b],
                output=root / "multi.png",
            )

            self.assertEqual(result["adversarial_accuracies"]["strategy_a"], 0.7)
            self.assertAlmostEqual(result["accuracy_drop_points"]["strategy_b"], 15.0)
            self.assertTrue((root / "multi.png").exists())

    def test_cli_rejects_unnamed_adversarial_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_path = _run_with_summary(
                Path(tmp_dir) / "normal-run",
                accuracy=0.8,
                correct_questions=80,
                dataset_name="medmcqa",
            )

            exit_code = main([str(run_path), "--adversarial", str(run_path)])

            self.assertEqual(exit_code, 2)

    def test_default_output_path_uses_run_names(self) -> None:
        output = default_output_path("runs/normal-run", "runs/adversarial-run")

        self.assertEqual(
            output.name,
            "accuracy_comparison_normal-run_vs_adversarial-run.png",
        )

    def test_default_multi_output_path_uses_all_run_names(self) -> None:
        output = default_multi_output_path(
            "runs/competition-20260505-210012",
            [
                "runs/adversarial-competition-20260430-091925",
                "runs/adversarial-competition-20260507-111314",
            ],
        )

        self.assertEqual(
            output.name,
            (
                "accuracy_comparison_competition-20260505-210012_vs_"
                "adversarial-competition-20260430-091925_"
                "adversarial-competition-20260507-111314.png"
            ),
        )


def _run_with_summary(
    run_path: Path,
    *,
    accuracy: float,
    correct_questions: int,
    dataset_name: str,
) -> Path:
    _write_summary(
        run_path,
        {
            "total_questions": 100,
            "accuracy": accuracy,
            "correct_questions": correct_questions,
            "evaluated_questions": 100,
        },
    )
    _write_question(run_path, dataset_name=dataset_name)
    return run_path
