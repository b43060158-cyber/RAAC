from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reasoning_bank_pipeline.extract_agent1_choice_influence_dataset import (
    ADVERSARIAL_OUTPUT,
    NON_ADVERSARIAL_OUTPUT,
    extract_datasets,
    main,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _review(reviewer: str, target: str, reason: str, *, score=7, stance="mixed") -> dict:
    return {
        "reviewer_agent_id": reviewer,
        "target_agent_id": target,
        "score": score,
        "stance": stance,
        "main_reason": reason,
    }


def _answer(agent_id: str, option_ids: list[str], reasoning: str, *, drivers=None, summary="", changed=False) -> dict:
    return {
        "agent_id": agent_id,
        "selected_option_ids": option_ids,
        "reasoning": reasoning,
        "changed_answer": changed,
        "change_drivers": drivers or [],
        "change_summary": summary,
    }


def _agent_result(
    agent_id: str,
    option_ids: list[str],
    reasoning: str,
    *,
    reviews_given=None,
    drivers=None,
    summary="",
    changed=False,
    adversarial=False,
) -> dict:
    return {
        "agent_id": agent_id,
        "answer": _answer(agent_id, option_ids, reasoning, drivers=drivers, summary=summary, changed=changed),
        "reviews_given": reviews_given or [],
        "received_reviews": [],
        "total_score": 0.0,
        "average_score": 0.0,
        "advers-agent": adversarial,
    }


def _round(index: int, *agent_results: dict) -> dict:
    return {"round_index": index, "agent_results": list(agent_results)}


class ExtractAgent1ChoiceInfluenceDatasetTest(unittest.TestCase):
    def test_extracts_successful_choice_influence_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs_dir = Path(tmp_dir) / "runs"
            competition_run = runs_dir / "competition-demo"
            adversarial_run = runs_dir / "adversarial-competition-demo"

            competition_payload = {
                "run_id": "competition-demo",
                "question_id": "0001",
                "dataset_name": "demo",
                "task_type": "single_choice",
                "question_key": "demo__0001__single_choice",
                "question": "Pick the best answer.",
                "options": [
                    {"option_id": "A", "text": "Alpha"},
                    {"option_id": "B", "text": "Beta"},
                    {"option_id": "C", "text": "Gamma"},
                ],
                "correct_option_ids": ["B"],
                "rounds": [
                    _round(
                        1,
                        _agent_result(
                            "agent_1",
                            ["B"],
                            "B is best because it matches the clue.",
                            reviews_given=[
                                _review(
                                    "agent_1",
                                    "agent_2",
                                    "Your answer ignores the clue; option B better fits the question.",
                                    score=4,
                                    stance="oppose",
                                )
                            ],
                        ),
                        _agent_result("agent_2", ["A"], "A seems plausible from the first sentence."),
                    ),
                    _round(
                        2,
                        _agent_result("agent_1", ["B"], "B is still best."),
                        _agent_result(
                            "agent_2",
                            ["B"],
                            "After reconsidering the clue, B fits better than A.",
                            drivers=["agent_1"],
                            summary="Switched from A to B after review highlighted the ignored clue.",
                            changed=True,
                        ),
                    ),
                ],
            }
            _write_json(competition_run / "questions" / "demo__0001__single_choice.json", competition_payload)

            adversarial_payload = {
                "run_id": "adversarial-competition-demo",
                "question_id": "0002",
                "dataset_name": "demo",
                "task_type": "multiple_choice",
                "question_key": "demo__0002__multiple_choice",
                "question": "Select all correct statements.",
                "metadata": {"source_task_key": "mc2_targets"},
                "options": [
                    {"option_id": "A", "text": "First"},
                    {"option_id": "B", "text": "Second"},
                    {"option_id": "C", "text": "Third"},
                ],
                "correct_option_ids": ["A", "C"],
                "rounds": [
                    _round(
                        1,
                        _agent_result(
                            "agent_1",
                            ["A", "C"],
                            "A and C are supported by the evidence.",
                            reviews_given=[
                                _review(
                                    "agent_1",
                                    "agent_3",
                                    "You missed option C; both A and C are directly supported.",
                                    score=3,
                                    stance="oppose",
                                )
                            ],
                            adversarial=True,
                        ),
                        _agent_result("agent_3", ["A"], "Only A seems safe.", adversarial=False),
                    ),
                    _round(
                        2,
                        _agent_result("agent_1", ["A", "C"], "A and C remain correct.", adversarial=True),
                        _agent_result(
                            "agent_3",
                            ["A", "C"],
                            "Adding C resolves the missing evidence noted in review.",
                            drivers=["agent_4", "agent_1"],
                            summary="Expanded answer to include C after peer feedback.",
                            changed=True,
                            adversarial=False,
                        ),
                    ),
                ],
            }
            _write_json(adversarial_run / "questions" / "demo__0002__multiple_choice.json", adversarial_payload)

            non_adversarial_dataset, adversarial_dataset = extract_datasets(runs_dir)

            self.assertEqual(non_adversarial_dataset["record_count"], 1)
            self.assertEqual(adversarial_dataset["record_count"], 0)

            non_adversarial_record = non_adversarial_dataset["records"][0]
            self.assertEqual(non_adversarial_record["run_type"], "non_adversarial")
            self.assertEqual(non_adversarial_record["effective_task_type"], "single_choice")
            self.assertEqual(non_adversarial_record["target_previous_selected_option_ids"], ["A"])
            self.assertEqual(non_adversarial_record["target_new_selected_option_ids"], ["B"])
            self.assertEqual(non_adversarial_record["target_change_drivers"], ["agent_1"])
            self.assertEqual(
                non_adversarial_record["agent1_review_text"],
                "Your answer ignores the clue; option B better fits the question.",
            )
            self.assertIn("Question:", non_adversarial_record["analysis_context"])
            self.assertIn("Agent 1 Review:", non_adversarial_record["analysis_context"])
            self.assertIn("Target Agent New Answer:", non_adversarial_record["analysis_context"])

    def test_keeps_effectively_single_choice_medmcqa_even_if_task_type_is_multiple_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs_dir = Path(tmp_dir) / "runs"
            run_path = runs_dir / "competition-demo"

            payload = {
                "run_id": "competition-demo",
                "question_id": "0003",
                "dataset_name": "medmcqa",
                "task_type": "multiple_choice",
                "question_key": "medmcqa__0003__multiple_choice",
                "question": "Pick the best diagnosis.",
                "metadata": {"choice_type": "multi"},
                "options": [
                    {"option_id": "A", "text": "Alpha"},
                    {"option_id": "B", "text": "Beta"},
                    {"option_id": "C", "text": "Gamma"},
                    {"option_id": "D", "text": "Delta"},
                ],
                "correct_option_ids": ["C"],
                "rounds": [
                    _round(
                        1,
                        _agent_result(
                            "agent_1",
                            ["C"],
                            "C fits the stem best.",
                            reviews_given=[
                                _review(
                                    "agent_1",
                                    "agent_2",
                                    "Option C is better supported by the stem than option A.",
                                    score=3,
                                    stance="oppose",
                                )
                            ],
                        ),
                        _agent_result("agent_2", ["A"], "I prefer A."),
                    ),
                    _round(
                        2,
                        _agent_result("agent_1", ["C"], "C remains best."),
                        _agent_result(
                            "agent_2",
                            ["C"],
                            "C is better supported after reviewing the stem.",
                            drivers=["agent_1"],
                            summary="Changed from A to C after review.",
                            changed=True,
                        ),
                    ),
                ],
            }
            _write_json(run_path / "questions" / "medmcqa__0003__multiple_choice.json", payload)

            non_adversarial_dataset, adversarial_dataset = extract_datasets(runs_dir)
            self.assertEqual(adversarial_dataset["record_count"], 0)
            self.assertEqual(non_adversarial_dataset["record_count"], 1)
            record = non_adversarial_dataset["records"][0]
            self.assertEqual(record["dataset_name"], "medmcqa")
            self.assertEqual(record["task_type"], "multiple_choice")
            self.assertEqual(record["effective_task_type"], "single_choice")

    def test_skips_non_choice_and_missing_review_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs_dir = Path(tmp_dir) / "runs"
            run_path = runs_dir / "competition-demo"

            skipped_no_review = {
                "run_id": "competition-demo",
                "question_id": "0001",
                "dataset_name": "demo",
                "task_type": "single_choice",
                "question_key": "demo__0001__single_choice",
                "question": "Pick one.",
                "options": [{"option_id": "A", "text": "Alpha"}, {"option_id": "B", "text": "Beta"}],
                "correct_option_ids": ["B"],
                "rounds": [
                    _round(
                        1,
                        _agent_result("agent_1", ["B"], "B is correct.", reviews_given=[]),
                        _agent_result("agent_2", ["A"], "A seems right."),
                    ),
                    _round(
                        2,
                        _agent_result("agent_1", ["B"], "B is still correct."),
                        _agent_result("agent_2", ["B"], "B now seems better.", drivers=["agent_1"], changed=True),
                    ),
                ],
            }
            non_choice = {
                "run_id": "competition-demo",
                "question_id": "0002",
                "dataset_name": "demo",
                "task_type": "math_short_answer",
                "question_key": "demo__0002__math_short_answer",
                "question": "Compute 1+1.",
                "rounds": [],
            }

            _write_json(run_path / "questions" / "demo__0001__single_choice.json", skipped_no_review)
            _write_json(run_path / "questions" / "demo__0002__math_short_answer.json", non_choice)

            non_adversarial_dataset, adversarial_dataset = extract_datasets(runs_dir)
            self.assertEqual(non_adversarial_dataset["record_count"], 0)
            self.assertEqual(adversarial_dataset["record_count"], 0)

    def test_main_writes_default_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs_dir = Path(tmp_dir) / "runs"
            run_path = runs_dir / "competition-demo"
            payload = {
                "run_id": "competition-demo",
                "question_id": "0001",
                "dataset_name": "demo",
                "task_type": "single_choice",
                "question_key": "demo__0001__single_choice",
                "question": "Pick one.",
                "options": [{"option_id": "A", "text": "Alpha"}, {"option_id": "B", "text": "Beta"}],
                "correct_option_ids": ["B"],
                "rounds": [
                    _round(
                        1,
                        _agent_result(
                            "agent_1",
                            ["B"],
                            "B is correct.",
                            reviews_given=[_review("agent_1", "agent_2", "B matches the clue.")],
                        ),
                        _agent_result("agent_2", ["A"], "A seems right."),
                    ),
                    _round(
                        2,
                        _agent_result("agent_1", ["B"], "B is still correct."),
                        _agent_result(
                            "agent_2",
                            ["B"],
                            "B now seems better.",
                            drivers=["agent_1"],
                            summary="Moved to B after review.",
                            changed=True,
                        ),
                    ),
                ],
            }
            _write_json(run_path / "questions" / "demo__0001__single_choice.json", payload)

            previous_cwd = Path.cwd()
            try:
                import os
                import sys

                os.chdir(tmp_dir)
                argv_backup = sys.argv
                sys.argv = [
                    "scripts/reasoning_bank_pipeline/extract_agent1_choice_influence_dataset.py",
                    "--runs-dir",
                    str(runs_dir),
                    "--non-adversarial-output",
                    str(runs_dir / NON_ADVERSARIAL_OUTPUT),
                    "--adversarial-output",
                    str(runs_dir / ADVERSARIAL_OUTPUT),
                ]
                exit_code = main()
            finally:
                os.chdir(previous_cwd)
                sys.argv = argv_backup

            self.assertEqual(exit_code, 0)
            self.assertTrue((runs_dir / NON_ADVERSARIAL_OUTPUT).exists())
            self.assertTrue((runs_dir / ADVERSARIAL_OUTPUT).exists())


if __name__ == "__main__":
    unittest.main()
