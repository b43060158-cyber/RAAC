import json
import tempfile
import unittest
from pathlib import Path

from autogen_mas.evaluation import Evaluator
from autogen_mas.models import (
    AnswerSubmission,
    ChessQuestionRecord,
    CodeQuestionRecord,
    OptionRecord,
    QuestionRoundResult,
    QuestionRunResult,
    ReviewSubmission,
    RoundAgentResult,
    ShortAnswerQuestionRecord,
)
from autogen_mas.persistence import JsonRunStore


def _agent_result(agent_id, selected_option_ids, total_score, average_score):
    answer = AnswerSubmission(agent_id=agent_id, selected_option_ids=selected_option_ids, reasoning="Reasoned")
    received_reviews = [
        ReviewSubmission("judge_1", agent_id, int(total_score / 2), "support", "Strong."),
        ReviewSubmission("judge_2", agent_id, int(total_score / 2), "support", "Strong."),
    ]
    return RoundAgentResult(
        agent_id=agent_id,
        answer=answer,
        reviews_given=[],
        received_reviews=received_reviews,
        total_score=float(total_score),
        average_score=float(average_score),
    )


def _single_agent_result(agent_id, selected_option_ids):
    return RoundAgentResult(
        agent_id=agent_id,
        answer=AnswerSubmission(agent_id=agent_id, selected_option_ids=selected_option_ids, reasoning="Reasoned"),
        reviews_given=[],
        received_reviews=[],
        total_score=0.0,
        average_score=0.0,
    )


def _code_agent_result(agent_id, code, total_score, average_score):
    answer = AnswerSubmission(
        agent_id=agent_id,
        selected_option_ids=[],
        reasoning="Implemented the function.",
        code=code,
    )
    received_reviews = [
        ReviewSubmission("judge_1", agent_id, int(total_score / 2), "support", "Strong."),
        ReviewSubmission("judge_2", agent_id, int(total_score / 2), "support", "Strong."),
    ]
    return RoundAgentResult(
        agent_id=agent_id,
        answer=answer,
        reviews_given=[],
        received_reviews=received_reviews,
        total_score=float(total_score),
        average_score=float(average_score),
    )


def _short_answer_agent_result(agent_id, final_answer, total_score, average_score):
    answer = AnswerSubmission(
        agent_id=agent_id,
        selected_option_ids=[],
        final_answer=final_answer,
        reasoning="Reasoned short answer.",
    )
    received_reviews = [
        ReviewSubmission("judge_1", agent_id, int(total_score / 2), "support", "Strong."),
        ReviewSubmission("judge_2", agent_id, int(total_score / 2), "support", "Strong."),
    ]
    return RoundAgentResult(
        agent_id=agent_id,
        answer=answer,
        reviews_given=[],
        received_reviews=received_reviews,
        total_score=float(total_score),
        average_score=float(average_score),
    )


class EvaluatorTest(unittest.TestCase):
    def test_evaluator_selects_top_agent_and_computes_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="eval-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult(
                    run_id="eval-run",
                    question_id="q1",
                    dataset_name="demo",
                    task_type="single_choice",
                    question="Choose one.",
                    options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                    correct_option_ids=["A"],
                    rounds=[
                        QuestionRoundResult(
                            round_index=2,
                            agent_results=[
                                _agent_result("agent_1", ["B"], 12, 6),
                                _agent_result("agent_2", ["A"], 12, 6),
                                _agent_result("agent_3", ["B"], 10, 5),
                            ],
                        )
                    ],
                ),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult(
                    run_id="eval-run",
                    question_id="q2",
                    dataset_name="demo",
                    task_type="multiple_choice",
                    question="Choose many.",
                    options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                    correct_option_ids=["A", "B"],
                    rounds=[
                        QuestionRoundResult(
                            round_index=2,
                            agent_results=[
                                _agent_result("agent_1", ["A"], 14, 7),
                                _agent_result("agent_2", ["A", "B"], 13, 6.5),
                            ],
                        )
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertEqual(len(report.question_results), 2)
            self.assertEqual(report.question_results[0].selected_agent_id, "agent_1")
            self.assertEqual(report.question_results[0].predicted_option_ids, ["B"])
            self.assertFalse(report.question_results[0].is_correct)
            self.assertTrue(report.question_results[0].is_tie)
            self.assertTrue(report.question_results[0].excluded_from_accuracy)
            self.assertEqual(
                report.question_results[0].tie_candidate_ids,
                ["agent_1", "agent_2"],
            )
            self.assertEqual(report.question_results[1].selected_agent_id, "agent_1")
            self.assertFalse(report.question_results[1].is_correct)
            self.assertFalse(report.question_results[1].is_tie)
            self.assertEqual(report.summary["total_questions"], 2)
            self.assertEqual(report.summary["evaluated_questions"], 1)
            self.assertEqual(report.summary["tie_question_count"], 1)
            self.assertEqual(
                report.summary["tie_question_keys"],
                ["demo__q1__single_choice"],
            )
            self.assertEqual(report.summary["correct_questions"], 0)
            self.assertEqual(report.summary["accuracy"], 0.0)
            self.assertEqual(report.summary["selection_rule_id"], "top_agent")
            self.assertIn("single_choice", report.summary["by_task_type"])
            self.assertTrue((Path(run_path) / "evaluation" / "summary.json").exists())
            self.assertTrue(
                (Path(run_path) / "evaluation" / "top_agent" / "summary.json").exists()
            )

    def test_top_agent_score_tie_with_same_answer_is_not_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="same-answer-tie-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult(
                    run_id="same-answer-tie-run",
                    question_id="q1",
                    dataset_name="demo",
                    task_type="single_choice",
                    question="Choose one.",
                    options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                    correct_option_ids=["A"],
                    rounds=[
                        QuestionRoundResult(
                            round_index=2,
                            agent_results=[
                                _agent_result("agent_1", ["A"], 12, 6),
                                _agent_result("agent_2", ["A"], 12, 6),
                                _agent_result("agent_3", ["B"], 10, 5),
                            ],
                        )
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertEqual(report.question_results[0].selected_agent_id, "agent_1")
            self.assertTrue(report.question_results[0].is_correct)
            self.assertFalse(report.question_results[0].is_tie)
            self.assertFalse(report.question_results[0].excluded_from_accuracy)
            self.assertEqual(report.question_results[0].tie_candidate_ids, [])
            self.assertEqual(report.summary["total_questions"], 1)
            self.assertEqual(report.summary["evaluated_questions"], 1)
            self.assertEqual(report.summary["tie_question_count"], 0)
            self.assertEqual(report.summary["correct_questions"], 1)
            self.assertEqual(report.summary["accuracy"], 1.0)

    def test_evaluator_uses_choice_consensus_round_before_later_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="choice-consensus-eval-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult(
                    run_id="choice-consensus-eval-run",
                    question_id="q1",
                    dataset_name="demo",
                    task_type="single_choice",
                    question="Choose one.",
                    options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                    correct_option_ids=["A"],
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[
                                _agent_result("agent_1", ["B"], 8, 4),
                                _agent_result("agent_2", ["B"], 6, 3),
                            ],
                            reached_consensus=True,
                        ),
                        QuestionRoundResult(
                            round_index=2,
                            agent_results=[
                                _agent_result("agent_1", ["A"], 20, 10),
                                _agent_result("agent_2", ["A"], 18, 9),
                            ],
                        ),
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertEqual(report.question_results[0].selected_agent_id, "agent_1")
            self.assertEqual(report.question_results[0].predicted_option_ids, ["B"])
            self.assertFalse(report.question_results[0].is_correct)

    def test_evaluator_uses_math_consensus_round_with_equivalent_answers(self) -> None:
        question = ShortAnswerQuestionRecord(
            question_id="0000",
            dataset_name="ciar",
            task_type="math_short_answer",
            question="What is the average speed?",
            acceptable_answers=["1.5", "3/2"],
            adversarial_target_answers=["2"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="math-consensus-eval-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult.from_task(
                    run_id="math-consensus-eval-run",
                    task=question,
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[
                                _short_answer_agent_result("agent_1", "1.5", 8, 4),
                                _short_answer_agent_result("agent_2", "3/2", 6, 3),
                            ],
                            reached_consensus=True,
                        ),
                        QuestionRoundResult(
                            round_index=2,
                            agent_results=[
                                _short_answer_agent_result("agent_1", "2", 20, 10),
                                _short_answer_agent_result("agent_2", "2", 18, 9),
                            ],
                        ),
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertEqual(report.question_results[0].selected_agent_id, "agent_1")
            self.assertEqual(report.question_results[0].predicted_final_answer, "1.5")
            self.assertTrue(report.question_results[0].is_correct)

    def test_normal_run_summary_includes_answer_change_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="normal-change-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult(
                    run_id="normal-change-run",
                    question_id="q1",
                    dataset_name="demo",
                    task_type="single_choice",
                    question="Choose one.",
                    options=[
                        OptionRecord("A", "Alpha"),
                        OptionRecord("B", "Beta"),
                        OptionRecord("C", "Gamma"),
                    ],
                    correct_option_ids=["A"],
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[
                                _agent_result("agent_1", ["A"], 10, 5),
                                _agent_result("agent_2", ["B"], 8, 4),
                                _agent_result("agent_3", ["B"], 6, 3),
                            ],
                        ),
                        QuestionRoundResult(
                            round_index=2,
                            agent_results=[
                                _agent_result("agent_1", ["B"], 8, 4),
                                _agent_result("agent_2", ["A"], 10, 5),
                                _agent_result("agent_3", ["C"], 6, 3),
                            ],
                        ),
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertEqual(
                report.summary["answer_change_metrics"],
                {
                    "honest_agents_changed_answer_count": 3,
                    "changed_to_wrong_count": 2,
                    "changed_to_correct_count": 1,
                },
            )
            self.assertNotIn("adversarial_metrics", report.summary)

    def test_top_agent_tie_breaks_by_all_round_average_score_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="all-round-tie-break-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult(
                    run_id="all-round-tie-break-run",
                    question_id="q1",
                    dataset_name="demo",
                    task_type="single_choice",
                    question="Choose one.",
                    options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                    correct_option_ids=["A"],
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[
                                _agent_result("agent_1", ["B"], 6, 3),
                                _agent_result("agent_2", ["A"], 10, 5),
                            ],
                        ),
                        QuestionRoundResult(
                            round_index=2,
                            agent_results=[
                                _agent_result("agent_1", ["B"], 12, 6),
                                _agent_result("agent_2", ["A"], 12, 6),
                            ],
                        ),
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertEqual(report.question_results[0].selected_agent_id, "agent_2")
            self.assertTrue(report.question_results[0].is_correct)
            self.assertFalse(report.question_results[0].is_tie)
            self.assertEqual(report.question_results[0].tie_candidate_ids, [])

    def test_evaluate_round_sweep_scores_each_round_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="round-sweep-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult(
                    run_id="round-sweep-run",
                    question_id="q1",
                    dataset_name="demo",
                    task_type="single_choice",
                    question="Choose one.",
                    options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                    correct_option_ids=["A"],
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[
                                _agent_result("agent_1", ["B"], 12, 6),
                                _agent_result("agent_2", ["A"], 10, 5),
                            ],
                        ),
                        QuestionRoundResult(
                            round_index=2,
                            agent_results=[
                                _agent_result("agent_1", ["B"], 8, 4),
                                _agent_result("agent_2", ["A"], 14, 7),
                            ],
                        ),
                    ],
                ),
            )

            summary = Evaluator().evaluate_round_sweep(run_path)

            self.assertEqual(summary["experiment"], "round_sweep")
            self.assertEqual(summary["max_round"], 2)
            self.assertEqual(len(summary["rounds"]), 2)
            self.assertEqual(summary["rounds"][0]["round"], 1)
            self.assertEqual(summary["rounds"][0]["accuracy"], 0.0)
            self.assertEqual(summary["rounds"][0]["correct_questions"], 0)
            self.assertEqual(summary["rounds"][1]["round"], 2)
            self.assertEqual(summary["rounds"][1]["accuracy"], 1.0)
            self.assertEqual(summary["rounds"][1]["correct_questions"], 1)
            persisted = Path(run_path) / "evaluation" / "round_sweep" / "summary.json"
            self.assertTrue(persisted.exists())
            self.assertEqual(json.loads(persisted.read_text())["max_round"], 2)

    def test_evaluator_scores_math_short_answers_with_equivalent_forms(self) -> None:
        question = ShortAnswerQuestionRecord(
            question_id="0000",
            dataset_name="ciar",
            task_type="math_short_answer",
            question="What is the average speed?",
            acceptable_answers=["1.5", "3/2"],
            adversarial_target_answers=["2"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="short-answer-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult.from_task(
                    run_id="short-answer-run",
                    task=question,
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[
                                _short_answer_agent_result("agent_1", "3/2 m/s", 12, 6),
                                _short_answer_agent_result("agent_2", "2", 10, 5),
                            ],
                        )
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertEqual(report.question_results[0].selected_agent_id, "agent_1")
            self.assertEqual(report.question_results[0].predicted_final_answer, "3/2 m/s")
            self.assertEqual(report.question_results[0].acceptable_answers, ["1.5", "3/2"])
            self.assertTrue(report.question_results[0].is_correct)
            self.assertEqual(
                report.summary["by_task_type"]["math_short_answer"]["accuracy"],
                1.0,
            )

    def test_evaluator_rejects_ciar_incorrect_answer(self) -> None:
        question = ShortAnswerQuestionRecord(
            question_id="0000",
            dataset_name="ciar",
            task_type="math_short_answer",
            question="What is the average speed?",
            acceptable_answers=["1.5", "3/2"],
            adversarial_target_answers=["2"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="short-answer-wrong-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult.from_task(
                    run_id="short-answer-wrong-run",
                    task=question,
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[
                                _short_answer_agent_result("agent_1", "2", 12, 6),
                            ],
                        )
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertFalse(report.question_results[0].is_correct)

    def test_evaluator_scores_chess_square_extracted_from_explanation(self) -> None:
        question = ChessQuestionRecord(
            question_id="Chess_0",
            dataset_name="chess",
            game="a2a4 a7a5",
            source_square="d2",
            legal_target_squares=["e3", "c3", "d1"],
            rendered_question="Chess question.",
            metadata={
                "answer_format": "chess_square",
                "source_square": "d2",
            },
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="chess-short-answer-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult.from_task(
                    run_id="chess-short-answer-run",
                    task=question,
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[
                                _short_answer_agent_result(
                                    "agent_1",
                                    "The piece on d2 can move to e3.",
                                    12,
                                    6,
                                ),
                            ],
                        )
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertEqual(report.question_results[0].predicted_final_answer, "e3")
            self.assertTrue(report.question_results[0].is_correct)
            self.assertEqual(report.question_results[0].task_type, "chess_move")

    def test_evaluator_scores_percent_equivalence_for_short_answers(self) -> None:
        question = ShortAnswerQuestionRecord(
            question_id="0001",
            dataset_name="ciar",
            task_type="math_short_answer",
            question="What is the probability?",
            acceptable_answers=["0.75", "3/4"],
            adversarial_target_answers=["0.9"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="short-answer-percent-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult.from_task(
                    run_id="short-answer-percent-run",
                    task=question,
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[
                                _short_answer_agent_result("agent_1", "75%", 12, 6),
                            ],
                        )
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

            self.assertTrue(report.question_results[0].is_correct)

    def test_evaluate_single_agent_writes_baseline_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="single-agent-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult(
                    run_id="single-agent-run",
                    question_id="q1",
                    dataset_name="demo",
                    task_type="single_choice",
                    question="Choose one.",
                    options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                    correct_option_ids=["A"],
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[_single_agent_result("single_agent", ["A"])],
                        )
                    ],
                ),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult(
                    run_id="single-agent-run",
                    question_id="q2",
                    dataset_name="demo",
                    task_type="single_choice",
                    question="Choose one.",
                    options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                    correct_option_ids=["A"],
                    rounds=[
                        QuestionRoundResult(
                            round_index=1,
                            agent_results=[_single_agent_result("single_agent", ["B"])],
                        )
                    ],
                ),
            )

            summary = Evaluator().evaluate_single_agent(
                run_path,
                agent_id="single_agent",
                model="qwen-plus",
                temperature=0.2,
            )

            summary_path = Path(run_path) / "evaluation" / "single_agent_summary.json"
            self.assertTrue(summary_path.exists())
            self.assertEqual(summary["agent_id"], "single_agent")
            self.assertEqual(summary["model"], "qwen-plus")
            self.assertEqual(summary["temperature"], 0.2)
            self.assertEqual(summary["selection_rule"], "single_agent_direct_answer")
            self.assertEqual(summary["total_questions"], 2)
            self.assertEqual(summary["correct_questions"], 1)
            self.assertEqual(summary["accuracy"], 0.5)
            self.assertEqual(summary["accuracy"], Evaluator().evaluate(run_path).summary["accuracy"])

    def test_evaluator_executes_top_agent_humaneval_code(self) -> None:
        question = CodeQuestionRecord(
            question_id="0",
            dataset_name="humaneval",
            prompt="def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
            entry_point="add_one",
            test="def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-1) == 0\n",
            source_path="data/HumanEval/HumanEval.jsonl",
            source_task_id="HumanEval/0",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="humaneval-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult.from_task(
                    run_id="humaneval-run",
                    task=question,
                    rounds=[
                        QuestionRoundResult(
                            round_index=3,
                            agent_results=[
                                _code_agent_result("agent_1", "    return x\n", 8, 4),
                                _code_agent_result("agent_2", "    return x + 1\n", 10, 5),
                            ],
                        )
                    ],
                ),
            )

            report = Evaluator().evaluate(run_path)

        self.assertEqual(report.question_results[0].selected_agent_id, "agent_2")
        self.assertEqual(report.question_results[0].predicted_option_ids, [])
        self.assertEqual(report.question_results[0].correct_option_ids, [])
        self.assertTrue(report.question_results[0].is_correct)
        self.assertEqual(report.summary["by_task_type"]["code_generation"]["accuracy"], 1.0)

    def test_weighted_vote_selection_rule_is_rejected(self) -> None:
        question = CodeQuestionRecord(
            question_id="0",
            dataset_name="humaneval",
            prompt="def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
            entry_point="add_one",
            test="def check(candidate):\n    assert candidate(1) == 2\n",
            source_path="data/HumanEval/HumanEval.jsonl",
            source_task_id="HumanEval/0",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="humaneval-run",
                experiment_config=type("Config", (), {"snapshot": lambda self: {}})(),
                llm_settings=type("Settings", (), {"snapshot": lambda self: {}})(),
            )
            store.persist_question_result(
                run_path,
                QuestionRunResult.from_task(
                    run_id="humaneval-run",
                    task=question,
                    rounds=[
                        QuestionRoundResult(
                            round_index=3,
                            agent_results=[
                                _code_agent_result("agent_1", "    return x + 1\n", 10, 5),
                            ],
                        )
                    ],
                ),
            )

            with self.assertRaisesRegex(ValueError, "Unsupported selection rule"):
                Evaluator().evaluate(run_path, selection_rule="weighted-vote")


if __name__ == "__main__":
    unittest.main()
