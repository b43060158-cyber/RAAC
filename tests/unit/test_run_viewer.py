from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_viewer import build_question_response, build_run_response, build_runs_response


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class RunViewerTest(unittest.TestCase):
    def test_builds_run_and_question_payload_from_result_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_path = Path(tmp_dir) / "competition-test"
            _write_json(
                run_path / "manifest.json",
                {
                    "run_id": "competition-test",
                    "experiment_config": {
                        "agents": [
                            {"agent_id": "agent_1"},
                            {"agent_id": "agent_2"},
                        ]
                    },
                },
            )
            _write_json(
                run_path / "evaluation" / "summary.json",
                {"total_questions": 1, "correct_questions": 1, "accuracy": 1.0},
            )
            _write_json(
                run_path / "evaluation" / "question_results.json",
                [
                    {
                        "question_key": "demo__0001__single_choice",
                        "selected_agent_id": "agent_1",
                        "predicted_option_ids": ["A"],
                        "correct_option_ids": ["A"],
                        "is_correct": True,
                    }
                ],
            )
            _write_json(
                run_path / "questions" / "demo__0001__single_choice.json",
                {
                    "run_id": "competition-test",
                    "question_id": "0001",
                    "dataset_name": "demo",
                    "task_type": "single_choice",
                    "question_key": "demo__0001__single_choice",
                    "question": "Pick A.",
                    "options": [
                        {"option_id": "A", "text": "Alpha"},
                        {"option_id": "B", "text": "Beta"},
                    ],
                    "correct_option_ids": ["A"],
                    "rounds": [
                        {
                            "round_index": 1,
                            "agent_results": [
                                {
                                    "agent_id": "agent_1",
                                    "answer": {
                                        "agent_id": "agent_1",
                                        "selected_option_ids": ["A"],
                                        "reasoning": "A is correct.",
                                        "confidence": 0.9,
                                        "changed_answer": False,
                                    },
                                    "reviews_given": [
                                        {
                                            "reviewer_agent_id": "agent_1",
                                            "target_agent_id": "agent_2",
                                            "score": 9,
                                            "stance": "support",
                                            "main_reason": "Good answer.",
                                        }
                                    ],
                                    "received_reviews": [],
                                    "total_score": 9.0,
                                    "average_score": 9.0,
                                    "advers-agent": False,
                                },
                                {
                                    "agent_id": "agent_2",
                                    "answer": {
                                        "agent_id": "agent_2",
                                        "selected_option_ids": ["B"],
                                        "reasoning": "B seems plausible.",
                                        "changed_answer": True,
                                        "change_drivers": ["agent_1"],
                                        "change_summary": "Changed after review.",
                                    },
                                    "reviews_given": [],
                                    "received_reviews": [],
                                    "total_score": 0.0,
                                    "average_score": 0.0,
                                    "advers-agent": False,
                                },
                            ],
                        }
                    ],
                },
            )

            run = build_run_response(run_path)
            self.assertEqual(run["run_id"], "competition-test")
            self.assertEqual(run["question_count"], 1)
            self.assertEqual(run["agents"], ["agent_1", "agent_2"])
            self.assertEqual(run["round_indexes"], [1])
            self.assertEqual(run["questions"][0]["selected_answer"], "A. Alpha")
            self.assertEqual(run["questions"][0]["selected_confidence"], 0.9)
            self.assertEqual(run["questions"][0]["changed_agents"], ["agent_2"])

            question = build_question_response(run_path, "demo__0001__single_choice")
            self.assertEqual(question["evaluation"]["selected_agent_id"], "agent_1")
            self.assertEqual(
                question["question"]["rounds"][0]["agent_results"][0]["reviews_given"][0]["main_reason"],
                "Good answer.",
            )

    def test_lists_valid_runs_for_dropdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "runs"
            valid_run = runs_root / "competition-new"
            invalid_run = runs_root / "not-a-run"
            _write_json(
                valid_run / "manifest.json",
                {"run_id": "competition-new", "created_at": "2026-05-09T10:00:00"},
            )
            _write_json(
                valid_run / "evaluation" / "summary.json",
                {"accuracy": 0.5},
            )
            _write_json(
                valid_run / "evaluation" / "question_results.json",
                [{"question_key": "demo__0001__single_choice", "is_correct": True}],
            )
            _write_json(
                valid_run / "questions" / "demo__0001__single_choice.json",
                {"question_key": "demo__0001__single_choice"},
            )
            _write_json(
                valid_run / "questions" / "demo__0002__single_choice.json",
                {"question_key": "demo__0002__single_choice"},
            )
            invalid_run.mkdir(parents=True)

            payload = build_runs_response(runs_root)

            self.assertEqual(payload["runs_root"], str(runs_root.resolve()))
            self.assertEqual(len(payload["runs"]), 1)
            self.assertEqual(payload["runs"][0]["name"], "competition-new")
            self.assertEqual(payload["runs"][0]["run_id"], "competition-new")
            self.assertEqual(payload["runs"][0]["question_count"], 2)
            self.assertEqual(payload["runs"][0]["evaluated_question_count"], 1)
            self.assertEqual(payload["runs"][0]["missing_evaluation_count"], 1)
            self.assertEqual(payload["runs"][0]["accuracy"], 0.5)

    def test_run_dropdown_excludes_runs_without_matching_evaluation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "runs"
            no_evaluation = runs_root / "competition-no-evaluation"
            mismatched = runs_root / "competition-mismatched"
            _write_json(
                no_evaluation / "questions" / "demo__0001__single_choice.json",
                {"question_key": "demo__0001__single_choice"},
            )
            _write_json(
                mismatched / "questions" / "demo__0001__single_choice.json",
                {"question_key": "demo__0001__single_choice"},
            )
            _write_json(
                mismatched / "evaluation" / "question_results.json",
                [{"question_key": "other__0001__single_choice"}],
            )

            payload = build_runs_response(runs_root)

            self.assertEqual(payload["runs"], [])


if __name__ == "__main__":
    unittest.main()
