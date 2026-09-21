from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reasoning_bank_pipeline.build_incremental_reasoning_bank_candidate import (
    _scan_selected_records,
)


class IncrementalReasoningBankCandidateTest(unittest.TestCase):
    def test_scan_selected_records_respects_run_dataset_and_question_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_path = root / "adversarial-competition-20260526-000000"
            question_dir = run_path / "questions"
            question_dir.mkdir(parents=True)

            matching_payload = {
                "dataset_name": "medmcqa",
                "task_type": "single_choice",
                "question_key": "medmcqa__0551__single_choice",
                "question_id": "551",
                "question": "Question?",
                "options": [{"option_id": "A", "text": "Alpha"}, {"option_id": "B", "text": "Beta"}],
                "correct_option_ids": ["A"],
                "rounds": [
                    {
                        "round_index": 1,
                        "agent_results": [
                            {
                                "agent_id": "agent_1",
                                "answer": {"selected_option_ids": ["B"], "reasoning": "r1"},
                                "reviews_given": [
                                    {
                                        "target_agent_id": "agent_2",
                                        "main_reason": "Option A better fits the stem.",
                                        "stance": "oppose",
                                        "score": 2,
                                    }
                                ],
                                "advers-agent": True,
                            },
                            {
                                "agent_id": "agent_2",
                                "answer": {"selected_option_ids": ["B"], "reasoning": "old"},
                            },
                        ],
                    },
                    {
                        "round_index": 2,
                        "agent_results": [
                            {
                                "agent_id": "agent_1",
                                "answer": {"selected_option_ids": ["B"], "reasoning": "r2"},
                            },
                            {
                                "agent_id": "agent_2",
                                "answer": {
                                    "selected_option_ids": ["A"],
                                    "reasoning": "new",
                                    "change_drivers": ["agent_1"],
                                    "change_summary": "Switched to A.",
                                },
                            },
                        ],
                    },
                ],
            }
            non_matching_payload = {
                **matching_payload,
                "dataset_name": "truthfulqa",
                "question_key": "truthfulqa__0001__single_choice",
                "question_id": "1",
            }
            (question_dir / "medmcqa__0551__single_choice.json").write_text(
                json.dumps(matching_payload),
                encoding="utf-8",
            )
            (question_dir / "truthfulqa__0001__single_choice.json").write_text(
                json.dumps(non_matching_payload),
                encoding="utf-8",
            )

            non_payload, adv_payload = _scan_selected_records(
                run_paths=[run_path],
                dataset_names={"medmcqa"},
                question_ids={"0551"},
            )

            self.assertEqual(non_payload["record_count"], 0)
            self.assertEqual(adv_payload["record_count"], 1)
            self.assertEqual(adv_payload["records"][0]["question_id"], "551")
            self.assertEqual(adv_payload["records"][0]["dataset_name"], "medmcqa")

    def test_scan_selected_records_can_include_all_questions_for_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_path = root / "adversarial-competition-20260526-000000"
            question_dir = run_path / "questions"
            question_dir.mkdir(parents=True)

            def payload(question_id: str) -> dict:
                return {
                    "dataset_name": "truthfulqa",
                    "task_type": "single_choice",
                    "question_key": f"truthfulqa__{question_id}__single_choice",
                    "question_id": question_id,
                    "question": "Question?",
                    "options": [{"option_id": "A", "text": "Alpha"}, {"option_id": "B", "text": "Beta"}],
                    "correct_option_ids": ["A"],
                    "rounds": [
                        {
                            "round_index": 1,
                            "agent_results": [
                                {
                                    "agent_id": "agent_1",
                                    "answer": {"selected_option_ids": ["B"], "reasoning": "r1"},
                                    "reviews_given": [
                                        {
                                            "target_agent_id": "agent_2",
                                            "main_reason": "Option B is better.",
                                            "stance": "oppose",
                                            "score": 2,
                                        }
                                    ],
                                    "advers-agent": True,
                                },
                                {
                                    "agent_id": "agent_2",
                                    "answer": {"selected_option_ids": ["A"], "reasoning": "old"},
                                },
                            ],
                        },
                        {
                            "round_index": 2,
                            "agent_results": [
                                {"agent_id": "agent_1", "answer": {"selected_option_ids": ["B"], "reasoning": "r2"}},
                                {
                                    "agent_id": "agent_2",
                                    "answer": {
                                        "selected_option_ids": ["B"],
                                        "reasoning": "new",
                                        "change_drivers": ["agent_1"],
                                        "change_summary": "Switched to B.",
                                    },
                                },
                            ],
                        },
                    ],
                }

            (question_dir / "truthfulqa__0001__single_choice.json").write_text(
                json.dumps(payload("0001")),
                encoding="utf-8",
            )
            (question_dir / "truthfulqa__0002__single_choice.json").write_text(
                json.dumps(payload("0002")),
                encoding="utf-8",
            )

            non_payload, adv_payload = _scan_selected_records(
                run_paths=[run_path],
                dataset_names={"truthfulqa"},
                question_ids=None,
            )

            self.assertEqual(non_payload["record_count"], 0)
            self.assertEqual(adv_payload["record_count"], 2)
            self.assertEqual(
                sorted(record["question_id"] for record in adv_payload["records"]),
                ["0001", "0002"],
            )


if __name__ == "__main__":
    unittest.main()
