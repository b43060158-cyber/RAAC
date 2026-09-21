from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_reasoning_bank_agent1_influence import validate_sample


class CheckReasoningBankAgent1InfluenceTest(unittest.TestCase):
    def test_validate_sample_accepts_true_agent1_influence_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_path = root / "adversarial-competition-20260525-181247"
            question_dir = run_path / "questions"
            question_dir.mkdir(parents=True)

            payload = {
                "rounds": [
                    {
                        "round_index": 1,
                        "agent_results": [
                            {
                                "agent_id": "agent_1",
                                "answer": {"selected_option_ids": ["B"], "reasoning": "r1"},
                                "reviews_given": [
                                    {
                                        "target_agent_id": "agent_3",
                                        "main_reason": "Option B is better.",
                                    }
                                ],
                            },
                            {
                                "agent_id": "agent_3",
                                "answer": {"selected_option_ids": ["A"], "reasoning": "old"},
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
                                "agent_id": "agent_3",
                                "answer": {
                                    "selected_option_ids": ["B"],
                                    "reasoning": "new",
                                    "change_drivers": ["agent_1"],
                                    "change_summary": "Changed after review.",
                                },
                            },
                        ],
                    },
                ]
            }
            (question_dir / "truthfulqa__0342__single_choice.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            failure = validate_sample(
                {
                    "sample_id": "adversarial-competition-20260525-181247::truthfulqa__0342__single_choice::agent_3::1->2::2"
                },
                [run_path],
            )
            self.assertIsNone(failure)

    def test_validate_sample_rejects_missing_agent1_change_driver(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_path = root / "adversarial-competition-20260525-181247"
            question_dir = run_path / "questions"
            question_dir.mkdir(parents=True)

            payload = {
                "rounds": [
                    {
                        "round_index": 1,
                        "agent_results": [
                            {
                                "agent_id": "agent_1",
                                "answer": {"selected_option_ids": ["B"], "reasoning": "r1"},
                                "reviews_given": [
                                    {
                                        "target_agent_id": "agent_3",
                                        "main_reason": "Option B is better.",
                                    }
                                ],
                            },
                            {
                                "agent_id": "agent_3",
                                "answer": {"selected_option_ids": ["A"], "reasoning": "old"},
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
                                "agent_id": "agent_3",
                                "answer": {
                                    "selected_option_ids": ["B"],
                                    "reasoning": "new",
                                    "change_drivers": ["agent_2"],
                                    "change_summary": "Changed after review.",
                                },
                            },
                        ],
                    },
                ]
            }
            (question_dir / "truthfulqa__0342__single_choice.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            failure = validate_sample(
                {
                    "sample_id": "adversarial-competition-20260525-181247::truthfulqa__0342__single_choice::agent_3::1->2::2"
                },
                [run_path],
            )
            self.assertEqual(failure, "agent1_not_in_change_drivers")


if __name__ == "__main__":
    unittest.main()
