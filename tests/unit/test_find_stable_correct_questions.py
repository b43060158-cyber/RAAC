from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.find_stable_correct_questions import (
    build_report,
    collect_incorrect_question_ids,
    collect_stable_correct_question_ids,
    collect_successful_question_ids,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _agent_result(agent_id: str, option_ids: list[str]) -> dict:
    return {
        "agent_id": agent_id,
        "answer": {
            "agent_id": agent_id,
            "selected_option_ids": option_ids,
        },
    }


class FindStableCorrectQuestionsTest(unittest.TestCase):
    def test_collects_only_correct_questions_with_identical_agent2_3_4_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "demo-run"
            _write_json(
                run_dir / "evaluation" / "question_results.json",
                [
                    {
                        "question_id": "0001",
                        "question_key": "demo__0001__single_choice",
                        "is_correct": True,
                    },
                    {
                        "question_id": "0002",
                        "question_key": "demo__0002__single_choice",
                        "is_correct": True,
                    },
                    {
                        "question_id": "0003",
                        "question_key": "demo__0003__single_choice",
                        "is_correct": False,
                    },
                ],
            )
            _write_json(
                run_dir / "questions" / "demo__0001__single_choice.json",
                {
                    "rounds": [
                        {
                            "round_index": 1,
                            "agent_results": [
                                _agent_result("agent_2", ["A"]),
                                _agent_result("agent_3", ["A"]),
                                _agent_result("agent_4", ["A"]),
                            ],
                        },
                        {
                            "round_index": 2,
                            "agent_results": [
                                _agent_result("agent_2", ["A"]),
                                _agent_result("agent_3", ["A"]),
                                _agent_result("agent_4", ["A"]),
                            ],
                        },
                    ]
                },
            )
            _write_json(
                run_dir / "questions" / "demo__0002__single_choice.json",
                {
                    "rounds": [
                        {
                            "round_index": 1,
                            "agent_results": [
                                _agent_result("agent_2", ["A"]),
                                _agent_result("agent_3", ["A"]),
                                _agent_result("agent_4", ["A"]),
                            ],
                        },
                        {
                            "round_index": 2,
                            "agent_results": [
                                _agent_result("agent_2", ["A"]),
                                _agent_result("agent_3", ["B"]),
                                _agent_result("agent_4", ["A"]),
                            ],
                        },
                    ]
                },
            )
            _write_json(
                run_dir / "questions" / "demo__0003__single_choice.json",
                {
                    "rounds": [
                        {
                            "round_index": 1,
                            "agent_results": [
                                _agent_result("agent_2", ["C"]),
                                _agent_result("agent_3", ["C"]),
                                _agent_result("agent_4", ["C"]),
                            ],
                        }
                    ]
                },
            )

            self.assertEqual(collect_successful_question_ids(run_dir), ["0001", "0002"])
            self.assertEqual(collect_incorrect_question_ids(run_dir), ["0003"])
            self.assertEqual(collect_stable_correct_question_ids(run_dir), ["0001"])

            report = build_report(run_dir)
            self.assertEqual(report["initial_difference_question_ids"], ["0002"])
            self.assertEqual(report["difference_question_ids"], ["0002"])

    def test_excludes_questions_missing_any_target_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "demo-run"
            _write_json(
                run_dir / "evaluation" / "question_results.json",
                [
                    {
                        "question_id": "0005",
                        "question_key": "demo__0005__single_choice",
                        "is_correct": True,
                    }
                ],
            )
            _write_json(
                run_dir / "questions" / "demo__0005__single_choice.json",
                {
                    "rounds": [
                        {
                            "round_index": 1,
                            "agent_results": [
                                _agent_result("agent_2", ["A"]),
                                _agent_result("agent_3", ["A"]),
                            ],
                        }
                    ]
                },
            )

            self.assertEqual(collect_stable_correct_question_ids(run_dir), [])


if __name__ == "__main__":
    unittest.main()
