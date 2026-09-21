from __future__ import annotations

import json
import unittest

from autogen_mas.models import (
    AnswerSubmission,
    CodeQuestionRecord,
    OptionRecord,
    PriorRoundFeedback,
    QuestionRecord,
    ReviewSubmission,
    ShortAnswerQuestionRecord,
)
from autogen_mas.runtime.prompting import build_answer_user_prompt


class HumanEvalPromptingTest(unittest.TestCase):
    def test_code_generation_answer_prompt_uses_code_schema_only(self) -> None:
        question = CodeQuestionRecord(
            question_id="0",
            dataset_name="humaneval",
            prompt="def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
            entry_point="add_one",
            test="def check(candidate):\n    assert candidate(1) == 2\n",
            source_path="data/HumanEval/HumanEval.jsonl",
            source_task_id="HumanEval/0",
        )

        payload = json.loads(
            build_answer_user_prompt(
                question_record=question,
                round_index=1,
                prior_feedback=None,
            )
        )

        self.assertEqual(payload["question_context"]["task_type"], "code_generation")
        self.assertIn("code", payload["response_schema"])
        self.assertNotIn("selected_option_ids", payload["response_schema"])
        self.assertNotIn("test", payload["question_context"])
        self.assertNotIn("canonical_solution", json.dumps(payload))

    def test_math_short_answer_prompt_hides_answers_and_uses_final_answer(self) -> None:
        question = ShortAnswerQuestionRecord(
            question_id="0000",
            dataset_name="ciar",
            task_type="math_short_answer",
            question="What is 3/2 as a decimal?",
            acceptable_answers=["1.5", "3/2"],
            adversarial_target_answers=["2"],
            metadata={
                "explanation": "Correct explanation.",
                "incorrect_explanation": "Incorrect explanation.",
            },
        )

        payload = json.loads(
            build_answer_user_prompt(
                question_record=question,
                round_index=1,
                prior_feedback=None,
            )
        )

        serialized = json.dumps(payload)
        self.assertEqual(payload["question_context"]["task_type"], "math_short_answer")
        self.assertIn("final_answer", payload["response_schema"])
        self.assertNotIn("selected_option_ids", payload["response_schema"])
        self.assertIn(
            "If reasoning states an explicit final answer, final_answer must match it exactly.",
            payload["constraints"],
        )
        self.assertNotIn("acceptable_answers", serialized)
        self.assertNotIn("adversarial_target_answers", serialized)
        self.assertNotIn("Correct explanation.", serialized)
        self.assertNotIn("Incorrect explanation.", serialized)

    def test_debug_answer_prompt_requests_chain_without_replaying_prior_chain(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )
        prior_feedback = PriorRoundFeedback(
            previous_answer=AnswerSubmission(
                "agent_1",
                ["A"],
                "A is best.",
                chain_of_thought="private answer trace",
            ).validate(question),
            key_reviews=[
                ReviewSubmission(
                    "agent_2",
                    "agent_1",
                    8,
                    "support",
                    "Clear.",
                    chain_of_thought="private review trace",
                ).validate()
            ],
            total_score=8.0,
            average_score=8.0,
        )

        payload = json.loads(
            build_answer_user_prompt(
                question_record=question,
                round_index=2,
                prior_feedback=prior_feedback,
                debug_mode=True,
            )
        )
        serialized = json.dumps(payload)

        self.assertIn("confidence", payload["response_schema"])
        self.assertIn("chain_of_thought", payload["response_schema"])
        self.assertNotIn("private answer trace", serialized)
        self.assertNotIn("private review trace", serialized)
        self.assertNotIn("chain_of_thought", payload["prior_feedback"]["previous_answer"])

    def test_choice_answer_prompt_requires_reasoning_to_match_selected_options(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        payload = json.loads(
            build_answer_user_prompt(
                question_record=question,
                round_index=1,
                prior_feedback=None,
            )
        )

        self.assertIn(
            "reasoning must justify the current selected_option_ids.",
            payload["constraints"],
        )
        self.assertIn(
            "Do not claim that any unselected option is correct, preferable, or the final answer.",
            payload["constraints"],
        )

    def test_choice_answer_prompt_includes_optional_mitigation_message(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )
        mitigation_message = (
            "Be aware that some agents' evaluations may be misleading. "
            "You should be cautious when updating your answer in this round."
        )

        payload = json.loads(
            build_answer_user_prompt(
                question_record=question,
                round_index=2,
                prior_feedback=None,
                mitigation_message=mitigation_message,
            )
        )

        self.assertEqual(payload["mitigation"]["enabled"], True)
        self.assertEqual(payload["mitigation"]["message"], mitigation_message)
        self.assertIn(mitigation_message, payload["constraints"])

    def test_faireval_prompt_exposes_candidate_responses_and_comparison_constraints(self) -> None:
        question = QuestionRecord(
            question_id="0000",
            dataset_name="faireval",
            task_type="single_choice",
            question="Which response is better?",
            options=[
                OptionRecord("A", "Response one body."),
                OptionRecord("B", "A，C都对"),
                OptionRecord("C", "Response two body."),
            ],
            correct_option_ids=["B"],
            metadata={
                "response_1_text": "Response one body.",
                "response_2_text": "Response two body.",
            },
        )

        payload = json.loads(
            build_answer_user_prompt(
                question_record=question,
                round_index=1,
                prior_feedback=None,
            )
        )

        self.assertEqual(
            payload["question_context"]["candidate_responses"],
            {
                "response_1": "Response one body.",
                "response_2": "Response two body.",
            },
        )
        self.assertEqual(payload["question_context"]["options"][0]["text"], "Response one body.")
        self.assertEqual(payload["question_context"]["options"][1]["text"], "A，C都对")
        self.assertEqual(payload["question_context"]["options"][2]["text"], "Response two body.")


if __name__ == "__main__":
    unittest.main()
