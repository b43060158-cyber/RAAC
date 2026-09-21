from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from autogen_mas.config import AgentConfig, DashScopeSettings, DriverAttributionConfig
from scripts.evaluate_changed_answers import (
    EXTERNAL_DRIVER_FOUND,
    NOT_EVALUATED,
    SELF_DECISION_CHANGE,
    build_change_chain_report,
    evaluate_changed_answers,
)


def _answer(agent_id: str, option_ids: list[str], *, changed=False, drivers=None) -> dict:
    return {
        "agent_id": agent_id,
        "selected_option_ids": option_ids,
        "reasoning": f"{agent_id} chose {','.join(option_ids)}",
        "changed_answer": changed,
        "change_drivers": drivers or [],
        "change_summary": "changed" if changed else "",
    }


def _code_answer(agent_id: str, code: str, *, changed=False, drivers=None) -> dict:
    return {
        "agent_id": agent_id,
        "code": code,
        "reasoning": f"{agent_id} wrote code",
        "changed_answer": changed,
        "change_drivers": drivers or [],
        "change_summary": "changed" if changed else "",
    }


def _review(reviewer: str, target: str, reason: str, *, score=6, stance="mixed") -> dict:
    return {
        "reviewer_agent_id": reviewer,
        "target_agent_id": target,
        "score": score,
        "stance": stance,
        "main_reason": reason,
    }


def _agent_result(
    agent_id: str,
    option_ids: list[str],
    *,
    received_reviews=None,
    changed=False,
    drivers=None,
) -> dict:
    return {
        "agent_id": agent_id,
        "answer": _answer(agent_id, option_ids, changed=changed, drivers=drivers),
        "reviews_given": [],
        "received_reviews": received_reviews or [],
        "total_score": 0.0,
        "average_score": 0.0,
    }


def _code_agent_result(
    agent_id: str,
    code: str,
    *,
    received_reviews=None,
    changed=False,
    drivers=None,
) -> dict:
    return {
        "agent_id": agent_id,
        "answer": _code_answer(agent_id, code, changed=changed, drivers=drivers),
        "reviews_given": [],
        "received_reviews": received_reviews or [],
        "total_score": 0.0,
        "average_score": 0.0,
    }


def _payload(question_id: str, rounds: list[dict]) -> dict:
    return {
        "run_id": "run",
        "question_id": question_id,
        "dataset_name": "demo",
        "task_type": "single_choice",
        "question": "Choose one.",
        "options": [{"option_id": "A", "text": "Alpha"}, {"option_id": "B", "text": "Beta"}, {"option_id": "C", "text": "Gamma"}],
        "correct_option_ids": ["A"],
        "rounds": rounds,
        "metadata": {},
        "question_key": f"demo__{question_id}__single_choice",
    }


def _code_payload(question_id: str, rounds: list[dict]) -> dict:
    return {
        "run_id": "run",
        "question_id": question_id,
        "dataset_name": "HumanEval",
        "task_type": "code_generation",
        "question": "Write f.",
        "prompt": "def f(x):",
        "entry_point": "f",
        "test": "assert f(1) == 2",
        "rounds": rounds,
        "metadata": {},
        "question_key": f"HumanEval__{question_id}__code_generation",
    }


def _round(index: int, *agent_results: dict) -> dict:
    return {"round_index": index, "agent_results": list(agent_results)}


def _write_run(payloads: list[dict]) -> Path:
    path = Path(tempfile.mkdtemp())
    questions = path / "questions"
    questions.mkdir()
    (path / "evaluation").mkdir()
    for payload in payloads:
        (questions / f"{payload['question_key']}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    return path


class FirstReviewDriverClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> dict[str, Any]:
        payload = json.loads(user_prompt)
        review = payload["previous_round_reviews"][0]
        return {
            "driver_scores": [
                {
                    "agent_id": review["reviewer_agent_id"],
                    "score": 0.9,
                    "label": "strong_driver",
                    "evidence_review_ids": [review["review_id"]],
                    "evidence_type": ["recommended_new_answer"],
                    "evidence": "this generated text must be ignored",
                }
            ],
            "insufficient_evidence": False,
            "reason": "based on review id",
        }


class EmptyDriverClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> dict[str, Any]:
        return {"driver_scores": [], "insufficient_evidence": True, "reason": "none"}


class MultiDriverClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> dict[str, Any]:
        payload = json.loads(user_prompt)
        return {
            "driver_scores": [
                {
                    "agent_id": review["reviewer_agent_id"],
                    "score": 0.9,
                    "label": "strong_driver",
                    "evidence_review_ids": [review["review_id"]],
                    "evidence_type": ["recommended_new_answer"],
                }
                for review in payload["previous_round_reviews"]
            ],
            "insufficient_evidence": False,
            "reason": "all cited",
        }


class FailingDriverClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> dict[str, Any]:
        raise RuntimeError("403 forbidden")


class InvalidEvidenceClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> dict[str, Any]:
        payload = json.loads(user_prompt)
        valid_review = payload["previous_round_reviews"][0]
        return {
            "driver_scores": [
                {
                    "agent_id": valid_review["reviewer_agent_id"],
                    "score": 0.9,
                    "label": "strong_driver",
                    "evidence_review_ids": [valid_review["review_id"]],
                    "evidence_type": ["recommended_new_answer"],
                    "evidence": "ignored",
                },
                {
                    "agent_id": "unknown_agent",
                    "score": 1.0,
                    "label": "strong_driver",
                    "evidence_review_ids": [valid_review["review_id"]],
                    "evidence_type": ["recommended_new_answer"],
                },
                {
                    "agent_id": "agent_3",
                    "score": 1.0,
                    "label": "strong_driver",
                    "evidence_review_ids": ["missing-review-id"],
                    "evidence_type": ["recommended_new_answer"],
                },
            ],
            "insufficient_evidence": False,
            "reason": "mixed",
        }


def _driver_config() -> DriverAttributionConfig:
    return DriverAttributionConfig(
        enabled=True,
        judges=[
            AgentConfig("driver_judge_1", temperature=0.0),
            AgentConfig("driver_judge_2", temperature=0.1),
            AgentConfig("driver_judge_3", temperature=0.2),
        ],
        max_workers=1,
    )


class ChangedAnswerEvaluationTest(unittest.TestCase):
    def tearDown(self) -> None:
        for path in getattr(self, "_run_paths", []):
            shutil.rmtree(path, ignore_errors=True)

    def _run_path(self, payloads: list[dict]) -> Path:
        path = _write_run(payloads)
        self._run_paths = [*getattr(self, "_run_paths", []), path]
        return path

    def test_builds_stepwise_and_initial_to_final_changes_without_llm(self) -> None:
        run_path = self._run_path(
            [
                _payload(
                    "q1",
                    [
                        _round(
                            1,
                            _agent_result(
                                "agent_1",
                                ["A"],
                                received_reviews=[
                                    _review("agent_2", "agent_1", "A is weak; B is better.")
                                ],
                            ),
                        ),
                        _round(2, _agent_result("agent_1", ["B"], changed=True, drivers=["agent_2"])),
                        _round(3, _agent_result("agent_1", ["B"])),
                    ],
                )
            ]
        )

        summary = evaluate_changed_answers(run_path)

        self.assertEqual(summary["stepwise_change_count"], 1)
        self.assertEqual(summary["initial_to_final_change_count"], 1)
        step = summary["stepwise_changes"][0]
        self.assertEqual(step["from_option_ids"], ["A"])
        self.assertEqual(step["to_option_ids"], ["B"])
        self.assertEqual(step["previous_round_reviews"][0]["main_reason_quote"], "A is weak; B is better.")
        self.assertEqual(step["attribution_result"], NOT_EVALUATED)
        initial = summary["initial_to_final_changes"][0]
        self.assertTrue(initial["changed"])
        self.assertEqual(initial["from_option_ids"], ["A"])
        self.assertEqual(initial["to_option_ids"], ["B"])
        self.assertEqual(summary["initial_to_final_total_comparisons"], 1)
        self.assertEqual(summary["initial_to_final_unchanged_count"], 0)

    def test_multiple_choice_order_does_not_count_as_change(self) -> None:
        payload = _payload(
            "q1",
            [
                _round(1, _agent_result("agent_1", ["A", "B"])),
                _round(2, _agent_result("agent_1", ["B", "A"])),
            ],
        )
        payload["task_type"] = "multiple_choice"
        payload["question_key"] = "demo__q1__multiple_choice"
        run_path = self._run_path([payload])

        summary = evaluate_changed_answers(run_path)

        self.assertEqual(summary["stepwise_change_count"], 0)
        self.assertEqual(summary["initial_to_final_changes"], [])
        self.assertEqual(summary["initial_to_final_changed_details"], [])
        self.assertEqual(summary["initial_to_final_total_comparisons"], 1)
        self.assertEqual(summary["initial_to_final_unchanged_count"], 1)

        full_summary = evaluate_changed_answers(
            run_path,
            include_unchanged_initial_to_final=True,
        )
        self.assertFalse(full_summary["initial_to_final_changes"][0]["changed"])

    def test_tracks_code_changes_with_hashes_and_diff_without_llm(self) -> None:
        previous_code = "def f(x):\n    return x + 1\n"
        current_code = "def f(x):\n    return x + 2\n"
        run_path = self._run_path(
            [
                _code_payload(
                    "q1",
                    [
                        _round(1, _code_agent_result("agent_1", previous_code)),
                        _round(2, _code_agent_result("agent_1", current_code)),
                    ],
                )
            ]
        )

        summary = evaluate_changed_answers(run_path)

        self.assertEqual(summary["actual_option_changed_count"], 0)
        self.assertEqual(summary["actual_code_changed_count"], 1)
        self.assertEqual(summary["actual_answer_changed_count"], 1)
        self.assertEqual(summary["questions_with_actual_option_changes"], 0)
        self.assertEqual(summary["questions_with_actual_code_changes"], 1)
        self.assertEqual(summary["questions_with_actual_answer_changes"], 1)
        self.assertEqual(summary["stepwise_change_count"], 1)

        step = summary["stepwise_changes"][0]
        self.assertFalse(step["actual_option_changed"])
        self.assertTrue(step["actual_code_changed"])
        self.assertTrue(step["actual_answer_changed"])
        self.assertNotEqual(step["previous_code_hash"], step["current_code_hash"])
        self.assertIn("-    return x + 1", step["code_diff"])
        self.assertIn("+    return x + 2", step["code_diff"])

        code_change = summary["actual_code_changes"][0]
        self.assertEqual(code_change, summary["actual_answer_changes"][0])
        self.assertFalse(code_change["self_reported_changed_answer"])
        self.assertTrue(code_change["actual_answer_changed"])
        self.assertNotEqual(code_change["previous_code_hash"], code_change["current_code_hash"])

        mismatch = summary["change_consistency_mismatches"][0]
        self.assertFalse(mismatch["self_reported_changed_answer"])
        self.assertTrue(mismatch["actual_code_changed"])
        self.assertTrue(mismatch["actual_answer_changed"])

        initial = summary["initial_to_final_changes"][0]
        self.assertTrue(initial["changed"])
        self.assertTrue(initial["actual_code_changed"])
        self.assertTrue(initial["actual_answer_changed"])
        self.assertIn("-    return x + 1", initial["code_diff"])
        self.assertIn("+    return x + 2", initial["code_diff"])

    def test_third_party_attribution_uses_only_review_ids_and_quotes(self) -> None:
        run_path = self._run_path(
            [
                _payload(
                    "q1",
                    [
                        _round(
                            1,
                            _agent_result(
                                "agent_1",
                                ["A"],
                                received_reviews=[
                                    _review("agent_2", "agent_1", "A is weak; B is better."),
                                    _review("agent_3", "agent_1", "This review is invalidly cited."),
                                ],
                            ),
                        ),
                        _round(2, _agent_result("agent_1", ["B"], changed=True, drivers=["agent_2"])),
                    ],
                )
            ]
        )

        summary = evaluate_changed_answers(
            run_path,
            driver_attribution_config=_driver_config(),
            llm_settings=DashScopeSettings(api_key="test", model="fallback-model"),
            client=InvalidEvidenceClient(),
        )

        step = summary["stepwise_changes"][0]
        self.assertEqual(step["attribution_result"], EXTERNAL_DRIVER_FOUND)
        self.assertFalse(step["self_decision_change"])
        self.assertEqual(len(step["third_party_inferred_drivers"]), 1)
        driver = step["third_party_inferred_drivers"][0]
        self.assertEqual(driver["agent_id"], "agent_2")
        self.assertEqual(driver["evidence_quotes"], ["A is weak; B is better."])
        self.assertNotIn("ignored", json.dumps(driver))
        self.assertTrue(any("unknown_agent" in warning for warning in step["driver_attribution_warnings"]))
        self.assertTrue(any("missing-review-id" in warning for warning in step["driver_attribution_warnings"]))

    def test_self_decision_change_when_no_external_driver_has_evidence(self) -> None:
        run_path = self._run_path(
            [
                _payload(
                    "q1",
                    [
                        _round(
                            1,
                            _agent_result(
                                "agent_1",
                                ["A"],
                                received_reviews=[_review("agent_2", "agent_1", "Ambiguous feedback.")],
                            ),
                        ),
                        _round(2, _agent_result("agent_1", ["B"], changed=True)),
                    ],
                )
            ]
        )

        summary = evaluate_changed_answers(
            run_path,
            driver_attribution_config=_driver_config(),
            llm_settings=DashScopeSettings(api_key="test", model="fallback-model"),
            client=EmptyDriverClient(),
        )

        step = summary["stepwise_changes"][0]
        self.assertEqual(step["attribution_result"], SELF_DECISION_CHANGE)
        self.assertTrue(step["self_decision_change"])
        self.assertEqual(step["third_party_inferred_drivers"], [])
        initial = summary["initial_to_final_changes"][0]
        self.assertEqual(initial["self_decision_change_steps"], ["1->2"])
        self.assertEqual(initial["net_change_drivers"], [])

    def test_initial_to_final_aggregates_from_stepwise_changes(self) -> None:
        run_path = self._run_path(
            [
                _payload(
                    "q1",
                    [
                        _round(
                            1,
                            _agent_result(
                                "agent_1",
                                ["A"],
                                received_reviews=[_review("agent_2", "agent_1", "Move from A to B.")],
                            ),
                        ),
                        _round(
                            2,
                            _agent_result(
                                "agent_1",
                                ["B"],
                                received_reviews=[_review("agent_3", "agent_1", "Move from B to C.")],
                                changed=True,
                            ),
                        ),
                        _round(
                            3,
                            _agent_result(
                                "agent_1",
                                ["C"],
                                received_reviews=[_review("agent_4", "agent_1", "Return to B.")],
                                changed=True,
                            ),
                        ),
                        _round(4, _agent_result("agent_1", ["B"], changed=True)),
                    ],
                )
            ]
        )

        summary = evaluate_changed_answers(
            run_path,
            driver_attribution_config=_driver_config(),
            llm_settings=DashScopeSettings(api_key="test", model="fallback-model"),
            client=FirstReviewDriverClient(),
        )

        self.assertEqual(summary["stepwise_change_count"], 3)
        initial = summary["initial_to_final_changes"][0]
        self.assertTrue(initial["changed"])
        self.assertEqual(initial["from_option_ids"], ["A"])
        self.assertEqual(initial["to_option_ids"], ["B"])
        self.assertEqual(
            {driver["agent_id"] for driver in initial["cumulative_path_drivers"]},
            {"agent_2", "agent_3", "agent_4"},
        )
        self.assertEqual(
            [driver["agent_id"] for driver in initial["direct_final_state_drivers"]],
            ["agent_4"],
        )
        self.assertEqual(
            {driver["agent_id"] for driver in initial["net_change_drivers"]},
            {"agent_2", "agent_4"},
        )

    def test_simple_chain_report_keeps_only_changed_trajectories(self) -> None:
        run_path = self._run_path(
            [
                _payload(
                    "q1",
                    [
                        _round(
                            1,
                            _agent_result(
                                "agent_1",
                                ["A"],
                                received_reviews=[
                                    _review("agent_2", "agent_1", "A is weak; B is better."),
                                    _review("agent_3", "agent_1", "B is more accurate."),
                                ],
                            ),
                            _agent_result("agent_2", ["A"]),
                        ),
                        _round(
                            2,
                            _agent_result("agent_1", ["B"], changed=True),
                            _agent_result("agent_2", ["A"]),
                        ),
                        _round(
                            3,
                            _agent_result("agent_1", ["B"]),
                            _agent_result("agent_2", ["A"]),
                        ),
                    ],
                )
            ]
        )

        full_summary = evaluate_changed_answers(
            run_path,
            driver_attribution_config=_driver_config(),
            llm_settings=DashScopeSettings(api_key="test", model="fallback-model"),
            client=MultiDriverClient(),
        )
        report = build_change_chain_report(full_summary)

        self.assertEqual(report["summary"]["changed_chain_count"], 1)
        self.assertEqual(report["summary"]["stepwise_change_count"], 1)
        self.assertEqual(report["summary"]["final_answer_changed_count"], 1)
        self.assertEqual(report["summary"]["external_agent_step_count"], 1)
        self.assertEqual(len(report["changes"]), 1)
        change = report["changes"][0]
        self.assertEqual(change["agent_id"], "agent_1")
        self.assertEqual(
            change["answer_chain"],
            [
                {"round": 1, "answer": ["A"]},
                {"round": 2, "answer": ["B"]},
                {"round": 3, "answer": ["B"]},
            ],
        )
        self.assertEqual(change["step_changes"][0]["cause"], "external_agent")
        self.assertEqual(
            {driver["agent_id"] for driver in change["final_answer_drivers"]},
            {"agent_2", "agent_3"},
        )
        self.assertEqual(change["final_driver_method"], "last_step_into_final_answer")
        self.assertIn("quote", change["final_answer_drivers"][0]["evidence"][0])

    def test_simple_chain_report_handles_no_net_change(self) -> None:
        run_path = self._run_path(
            [
                _payload(
                    "q1",
                    [
                        _round(1, _agent_result("agent_1", ["A"])),
                        _round(2, _agent_result("agent_1", ["B"], changed=True)),
                        _round(3, _agent_result("agent_1", ["A"], changed=True)),
                    ],
                )
            ]
        )

        report = build_change_chain_report(evaluate_changed_answers(run_path))

        self.assertEqual(report["summary"]["changed_chain_count"], 1)
        change = report["changes"][0]
        self.assertFalse(change["final_answer_changed"])
        self.assertEqual(change["final_answer_drivers"], [])
        self.assertEqual(change["final_driver_method"], "no_net_final_change")
        self.assertEqual([step["cause"] for step in change["step_changes"]], ["unknown", "unknown"])

    def test_simple_chain_report_marks_self_decision_and_unknown(self) -> None:
        run_path = self._run_path(
            [
                _payload(
                    "q1",
                    [
                        _round(
                            1,
                            _agent_result(
                                "agent_1",
                                ["A"],
                                received_reviews=[_review("agent_2", "agent_1", "Ambiguous.")],
                            ),
                        ),
                        _round(2, _agent_result("agent_1", ["B"], changed=True)),
                    ],
                )
            ]
        )

        self_report = build_change_chain_report(evaluate_changed_answers(run_path))
        self.assertEqual(self_report["changes"][0]["step_changes"][0]["cause"], "unknown")
        self.assertEqual(self_report["changes"][0]["final_driver_method"], "driver_attribution_not_run")

        full_summary = evaluate_changed_answers(
            run_path,
            driver_attribution_config=_driver_config(),
            llm_settings=DashScopeSettings(api_key="test", model="fallback-model"),
            client=EmptyDriverClient(),
        )
        report = build_change_chain_report(full_summary)
        self.assertEqual(report["changes"][0]["step_changes"][0]["cause"], "self_decision")
        self.assertEqual(
            report["changes"][0]["final_driver_method"],
            "last_step_into_final_answer_self_decision",
        )

    def test_judge_failure_does_not_crash_and_keeps_unknown_cause(self) -> None:
        run_path = self._run_path(
            [
                _payload(
                    "q1",
                    [
                        _round(
                            1,
                            _agent_result(
                                "agent_1",
                                ["A"],
                                received_reviews=[_review("agent_2", "agent_1", "B maybe.")],
                            ),
                        ),
                        _round(2, _agent_result("agent_1", ["B"], changed=True)),
                    ],
                )
            ]
        )

        full_summary = evaluate_changed_answers(
            run_path,
            driver_attribution_config=_driver_config(),
            llm_settings=DashScopeSettings(api_key="test", model="fallback-model"),
            client=FailingDriverClient(),
        )
        report = build_change_chain_report(full_summary)

        self.assertEqual(report["changes"][0]["step_changes"][0]["cause"], "unknown")
        self.assertEqual(report["changes"][0]["final_driver_method"], "driver_attribution_not_run")
        self.assertTrue(full_summary["stepwise_changes"][0]["driver_attribution_warnings"])


if __name__ == "__main__":
    unittest.main()
