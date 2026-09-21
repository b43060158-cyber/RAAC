import unittest

from autogen_mas.models import (
    AnswerSubmission,
    ChessQuestionRecord,
    CodeQuestionRecord,
    OptionRecord,
    QuestionRecord,
    ReviewSubmission,
    ShortAnswerQuestionRecord,
    ValidationError,
)


class SubmissionValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.single_choice_question = QuestionRecord(
            question_id="q1",
            dataset_name="test",
            task_type="single_choice",
            question="Pick one.",
            options=[
                OptionRecord("A", "Alpha"),
                OptionRecord("B", "Beta"),
            ],
            correct_option_ids=["A"],
        )
        self.multiple_choice_question = QuestionRecord(
            question_id="q2",
            dataset_name="test",
            task_type="multiple_choice",
            question="Pick many.",
            options=[
                OptionRecord("A", "Alpha"),
                OptionRecord("B", "Beta"),
                OptionRecord("C", "Gamma"),
            ],
            correct_option_ids=["A", "C"],
        )
        self.code_question = CodeQuestionRecord(
            question_id="0",
            dataset_name="humaneval",
            prompt="def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
            entry_point="add_one",
            test="def check(candidate):\n    assert candidate(1) == 2\n",
            source_path="data/HumanEval/HumanEval.jsonl",
            source_task_id="HumanEval/0",
        )
        self.short_answer_question = ShortAnswerQuestionRecord(
            question_id="0000",
            dataset_name="ciar",
            task_type="math_short_answer",
            question="What is 3/2 as a decimal?",
            acceptable_answers=["1.5", "3/2"],
            adversarial_target_answers=["2"],
        )

    def test_single_choice_requires_exactly_one_option(self) -> None:
        with self.assertRaises(ValidationError):
            AnswerSubmission(
                agent_id="agent_1",
                selected_option_ids=["A", "B"],
                reasoning="Two answers.",
            ).validate(self.single_choice_question)

    def test_multiple_choice_accepts_multiple_options(self) -> None:
        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=["C", "A", "A"],
            reasoning="Both seem correct.",
        ).validate(self.multiple_choice_question)

        self.assertEqual(submission.selected_option_ids, ["A", "C"])

    def test_changed_answer_requires_summary_and_external_drivers(self) -> None:
        with self.assertRaises(ValidationError):
            AnswerSubmission(
                agent_id="agent_1",
                selected_option_ids=["A"],
                reasoning="Updated after feedback.",
                changed_answer=True,
                change_drivers=["agent_1"],
                change_summary="",
            ).validate(self.single_choice_question)

        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=["A"],
            reasoning="Updated after feedback.",
            changed_answer=True,
            change_drivers=["agent_2", "agent_2", "agent_3"],
            change_summary="Peer feedback highlighted the missing constraint.",
        ).validate(self.single_choice_question)

        self.assertEqual(submission.change_drivers, ["agent_2", "agent_3"])

    def test_invalid_option_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AnswerSubmission(
                agent_id="agent_1",
                selected_option_ids=["Z"],
                reasoning="Unknown option.",
            ).validate(self.single_choice_question)

    def test_single_choice_uses_reasoning_decision_when_it_conflicts_with_selected_option(self) -> None:
        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=["A"],
            reasoning="My calculation gives option B, so the correct answer is B.",
        ).validate(self.single_choice_question)

        self.assertEqual(submission.selected_option_ids, ["B"])

    def test_multiple_choice_uses_reasoning_decision_when_it_conflicts_with_selected_option_set(self) -> None:
        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=["A", "C"],
            reasoning="The correct answers are A and B.",
        ).validate(self.multiple_choice_question)

        self.assertEqual(submission.selected_option_ids, ["A", "B"])

    def test_choice_reasoning_accepts_explicit_self_consistent_conclusion(self) -> None:
        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=["A"],
            reasoning="Option A is correct because it satisfies the stated condition.",
        ).validate(self.single_choice_question)

        self.assertEqual(submission.selected_option_ids, ["A"])

    def test_choice_reasoning_still_rejects_selected_option_that_it_calls_incorrect(self) -> None:
        with self.assertRaisesRegex(ValidationError, "reasoning must defend"):
            AnswerSubmission(
                agent_id="agent_1",
                selected_option_ids=["A"],
                reasoning="Option A is incorrect because it violates the condition.",
            ).validate(self.single_choice_question)

    def test_code_generation_requires_code_and_no_options(self) -> None:
        with self.assertRaises(ValidationError):
            AnswerSubmission(
                agent_id="agent_1",
                selected_option_ids=[],
                reasoning="No code.",
            ).validate(self.code_question)

        with self.assertRaises(ValidationError):
            AnswerSubmission(
                agent_id="agent_1",
                selected_option_ids=["A"],
                reasoning="Code with an option.",
                code="def add_one(x):\n    return x + 1\n",
            ).validate(self.code_question)

        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=[],
            reasoning="Implements the requested function.",
            code="  def add_one(x):\n      return x + 1\n  ",
        ).validate(self.code_question)

        self.assertEqual(submission.selected_option_ids, [])
        self.assertIn("def add_one", submission.code)

    def test_short_answer_requires_final_answer_and_reasoning(self) -> None:
        with self.assertRaises(ValidationError):
            AnswerSubmission(
                agent_id="agent_1",
                selected_option_ids=[],
                reasoning="No final answer.",
            ).validate(self.short_answer_question)

        with self.assertRaises(ValidationError):
            AnswerSubmission(
                agent_id="agent_1",
                selected_option_ids=["A"],
                final_answer="1.5",
                reasoning="Short answer with option ids.",
            ).validate(self.short_answer_question)

        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=[],
            final_answer="  1.5  ",
            reasoning="  Three divided by two is 1.5.  ",
        ).validate(self.short_answer_question)

        self.assertEqual(submission.selected_option_ids, [])
        self.assertEqual(submission.final_answer, "1.5")
        self.assertEqual(submission.reasoning, "Three divided by two is 1.5.")

    def test_short_answer_final_answer_follows_explicit_reasoning_conclusion(self) -> None:
        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=[],
            final_answer="0.9",
            reasoning=(
                "P(third | not first two) = 0.3 / 0.4 = 0.75. "
                "So final answer is 0.75."
            ),
        ).validate(self.short_answer_question)

        self.assertEqual(submission.final_answer, "0.75")

    def test_short_answer_keeps_field_when_reasoning_has_only_intermediate_numbers(self) -> None:
        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=[],
            final_answer="0.9",
            reasoning="The calculation uses 0.3 and 0.4 as intermediate values.",
        ).validate(self.short_answer_question)

        self.assertEqual(submission.final_answer, "0.9")

    def test_chess_short_answer_normalizes_square_from_explanation(self) -> None:
        chess_question = ChessQuestionRecord(
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

        submission = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=[],
            final_answer="The piece on d2 can legally move to e3.",
            reasoning="From d2 the piece can move to e3, so final answer: e3.",
        ).validate(chess_question)

        self.assertEqual(submission.final_answer, "e3")

    def test_review_validation_checks_score_fields_and_stance(self) -> None:
        with self.assertRaises(ValidationError):
            ReviewSubmission(
                reviewer_agent_id="agent_1",
                target_agent_id="agent_2",
                score=11,
                stance="support",
                main_reason="Clear.",
            ).validate()

        with self.assertRaises(ValidationError):
            ReviewSubmission(
                reviewer_agent_id="agent_1",
                target_agent_id="agent_2",
                score=8,
                stance="agree",
                main_reason="Clear.",
            ).validate()

        with self.assertRaises(ValidationError):
            ReviewSubmission(
                reviewer_agent_id="agent_1",
                target_agent_id="agent_2",
                score=8,
                stance="mixed",
                main_reason="  ",
            ).validate()

    def test_chain_of_thought_serializes_only_when_present(self) -> None:
        answer = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=["A"],
            reasoning="A is best.",
        ).validate(self.single_choice_question)
        review = ReviewSubmission(
            reviewer_agent_id="agent_2",
            target_agent_id="agent_1",
            score=8,
            stance="support",
            main_reason="Clear.",
        ).validate()

        self.assertNotIn("chain_of_thought", answer.to_dict())
        self.assertNotIn("chain_of_thought", review.to_dict())

        debug_answer = AnswerSubmission(
            agent_id="agent_1",
            selected_option_ids=["A"],
            reasoning="A is best.",
            confidence=0.82,
            chain_of_thought="  Step one, then step two.  ",
        ).validate(self.single_choice_question)
        debug_review = ReviewSubmission(
            reviewer_agent_id="agent_2",
            target_agent_id="agent_1",
            score=8,
            stance="support",
            main_reason="Clear.",
            chain_of_thought="  Compare answer and evidence.  ",
        ).validate()

        self.assertEqual(
            debug_answer.to_dict()["confidence"],
            0.82,
        )
        self.assertEqual(
            debug_answer.to_dict()["chain_of_thought"],
            "Step one, then step two.",
        )
        self.assertEqual(
            debug_review.to_dict()["chain_of_thought"],
            "Compare answer and evidence.",
        )
        self.assertNotIn(
            "chain_of_thought",
            debug_answer.to_dict(include_chain_of_thought=False),
        )
        self.assertNotIn(
            "chain_of_thought",
            debug_review.to_dict(include_chain_of_thought=False),
        )

    def test_confidence_must_be_within_unit_interval(self) -> None:
        with self.assertRaisesRegex(ValidationError, "confidence must be between 0.0 and 1.0"):
            AnswerSubmission(
                agent_id="agent_1",
                selected_option_ids=["A"],
                reasoning="A is best.",
                confidence=1.2,
            ).validate(self.single_choice_question)


if __name__ == "__main__":
    unittest.main()
