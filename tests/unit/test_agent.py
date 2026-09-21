from __future__ import annotations

import json
import unittest

from autogen_mas.config import AgentConfig, AlignmentJudgeConfig, ExperimentConfig
from autogen_mas.models import AnswerSubmission, OptionRecord, QuestionRecord, ValidationError
from autogen_mas.runtime.agent import ModelBackedMasAgent


class FakeStructuredClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_json(self, *, system_prompt, user_prompt, model, temperature):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model": model,
                "temperature": temperature,
            }
        )
        payload = json.loads(user_prompt)
        return {
            "reviews": [
                {
                    "target_agent_id": submission["agent_id"],
                    "score": 8,
                    "stance": "support",
                    "main_reason": "The answer is well supported.",
                }
                for submission in payload["peer_submissions"]
            ]
        }


class DebugStructuredClient(FakeStructuredClient):
    def __init__(self, *, include_chain_of_thought: bool) -> None:
        super().__init__()
        self.include_chain_of_thought = include_chain_of_thought

    def _maybe_debug(self) -> dict[str, object]:
        payload: dict[str, object] = {"confidence": 0.73}
        if self.include_chain_of_thought:
            payload["chain_of_thought"] = "Inspect the evidence, then choose A."
        return payload

    def generate_json(self, *, system_prompt, user_prompt, model, temperature):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model": model,
                "temperature": temperature,
            }
        )
        payload = json.loads(user_prompt)
        if payload["stage"] == "answer":
            return {
                "selected_option_ids": ["A"],
                "reasoning": "A is best.",
                **self._maybe_debug(),
            }
        if payload["stage"] == "batch_review":
            return {
                "reviews": [
                    {
                        "target_agent_id": submission["agent_id"],
                        "score": 8,
                        "stance": "support",
                        "main_reason": "The answer is well supported.",
                        **self._maybe_debug(),
                    }
                    for submission in payload["peer_submissions"]
                ]
            }
        return super().generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )


class ModelBackedMasAgentTest(unittest.TestCase):
    def test_answer_coerces_single_string_option_id(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        class StringOptionClient(FakeStructuredClient):
            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                self.calls.append(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "model": model,
                        "temperature": temperature,
                    }
                )
                return {
                    "selected_option_ids": "A",
                    "reasoning": "The best answer is A.",
                }

        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=StringOptionClient(),
            default_model="fake-model",
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )

        self.assertEqual(answer.selected_option_ids, ["A"])

    def test_answer_prefers_reasoning_decision_over_conflicting_structured_option(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        class ConflictingAnswerClient(FakeStructuredClient):
            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                self.calls.append(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "model": model,
                        "temperature": temperature,
                    }
                )
                return {
                    "selected_option_ids": ["A"],
                    "reasoning": "After checking the conditions, the correct answer is B.",
                }

        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=ConflictingAnswerClient(),
            default_model="fake-model",
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )

        self.assertEqual(answer.selected_option_ids, ["B"])

    def test_answer_repairs_multi_option_single_choice_from_reasoning_decision(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        class MultiOptionClient(FakeStructuredClient):
            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                self.calls.append(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "model": model,
                        "temperature": temperature,
                    }
                )
                return {
                    "selected_option_ids": ["A", "B"],
                    "reasoning": "After comparing the choices, the correct answer is B.",
                }

        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=MultiOptionClient(),
            default_model="fake-model",
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )

        self.assertEqual(answer.selected_option_ids, ["B"])

    def test_answer_repairs_empty_single_choice_from_final_answer(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        class EmptyOptionClient(FakeStructuredClient):
            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                self.calls.append(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "model": model,
                        "temperature": temperature,
                    }
                )
                return {
                    "selected_option_ids": [],
                    "final_answer": "A",
                    "reasoning": "Alpha is the best supported option.",
                }

        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=EmptyOptionClient(),
            default_model="fake-model",
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )

        self.assertEqual(answer.selected_option_ids, ["A"])

    def test_answer_repair_pass_fixes_invalid_clean_single_choice_answer(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        class RepairingClient(FakeStructuredClient):
            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                self.calls.append(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "model": model,
                        "temperature": temperature,
                    }
                )
                payload = json.loads(user_prompt)
                if payload["stage"] == "answer":
                    return {
                        "selected_option_ids": [],
                        "reasoning": "Both options have some appeal.",
                    }
                if payload["stage"] == "answer_repair":
                    return {
                        "selected_option_ids": ["B"],
                        "reasoning": "B is the single best matching option.",
                    }
                return super().generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                )

        client = RepairingClient()
        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=client,
            default_model="fake-model",
            enable_answer_repair=True,
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )

        self.assertEqual(answer.selected_option_ids, ["B"])
        self.assertEqual(len(client.calls), 2)
        repair_prompt = json.loads(client.calls[1]["user_prompt"])
        self.assertEqual(repair_prompt["stage"], "answer_repair")
        self.assertEqual(repair_prompt["invalid_response"]["selected_option_ids"], [])
        self.assertEqual(client.calls[1]["temperature"], 0.0)

    def test_answer_repair_pass_is_disabled_by_default(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        class InvalidAnswerClient(FakeStructuredClient):
            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                self.calls.append(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "model": model,
                        "temperature": temperature,
                    }
                )
                return {
                    "selected_option_ids": [],
                    "reasoning": "Both options have some appeal.",
                }

        client = InvalidAnswerClient()
        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=client,
            default_model="fake-model",
        )

        with self.assertRaisesRegex(ValidationError, "must choose exactly one option"):
            agent.answer(
                question_context=question,
                prior_feedback=None,
                round_index=1,
            )
        self.assertEqual(len(client.calls), 1)

    def test_answer_alignment_judge_can_keep_structured_option(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        class JudgeClient(FakeStructuredClient):
            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                self.calls.append(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "model": model,
                        "temperature": temperature,
                    }
                )
                payload = json.loads(user_prompt)
                if payload["stage"] == "answer":
                    return {
                        "selected_option_ids": ["A"],
                        "reasoning": "An intermediate note says the correct answer is B, but the defended conclusion remains A.",
                    }
                if payload["stage"] == "alignment_judge":
                    return {
                        "selected_candidate": "structured",
                        "resolved_option_ids": ["A"],
                        "main_reason": "The reasoning text ultimately still defends the structured answer.",
                    }
                return super().generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                )

        config = ExperimentConfig(
            alignment_judge=AlignmentJudgeConfig(
                enabled=True,
                judges=[AgentConfig(agent_id="judge_1", model="judge-model", temperature=0.0)],
                fallback_policy="reasoning_override",
            )
        )
        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=config,
            client=JudgeClient(),
            default_model="fake-model",
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )

        self.assertEqual(answer.selected_option_ids, ["A"])
        self.assertEqual(answer.raw_selected_option_ids, ["A"])
        self.assertEqual(answer.alignment_resolution, "llm_judge_structured")
        self.assertEqual(answer.alignment_judge_votes, ["judge_1:structured"])

    def test_answer_alignment_judge_repairs_inconsistent_reasoning_without_explicit_option(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        class JudgeRepairClient(FakeStructuredClient):
            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                self.calls.append(
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "model": model,
                        "temperature": temperature,
                    }
                )
                payload = json.loads(user_prompt)
                if payload["stage"] == "answer":
                    return {
                        "selected_option_ids": ["A"],
                        "reasoning": "Option A is incorrect because it contradicts the stated condition.",
                    }
                if payload["stage"] == "alignment_judge_repair":
                    return {
                        "resolved_option_ids": ["B"],
                        "main_reason": "The reasoning rejects A, so B is the defended alternative.",
                    }
                return super().generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                )

        config = ExperimentConfig(
            alignment_judge=AlignmentJudgeConfig(
                enabled=True,
                judges=[AgentConfig(agent_id="judge_1", model="judge-model", temperature=0.0)],
                fallback_policy="reasoning_override",
            )
        )
        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=config,
            client=JudgeRepairClient(),
            default_model="fake-model",
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )

        self.assertEqual(answer.selected_option_ids, ["B"])
        self.assertEqual(answer.raw_selected_option_ids, ["A"])
        self.assertEqual(answer.alignment_resolution, "llm_judge_inferred")
        self.assertEqual(answer.alignment_judge_votes, ["judge_1:inferred"])

    def test_review_many_uses_one_batch_llm_call(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )
        reviewer_answer = AnswerSubmission("agent_1", ["A"], "A is correct.").validate(question)
        peer_submissions = [
            AnswerSubmission("agent_2", ["A"], "Same answer.").validate(question),
            AnswerSubmission("agent_3", ["B"], "Different answer.").validate(question),
        ]
        client = FakeStructuredClient()
        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1", temperature=0.3),
            experiment_config=ExperimentConfig(),
            client=client,
            default_model="fake-model",
        )

        reviews = agent.review_many(
            question_context=question,
            peer_submissions=peer_submissions,
            reviewer_answer=reviewer_answer,
            round_index=1,
        )

        self.assertEqual(len(client.calls), 1)
        self.assertEqual([review.target_agent_id for review in reviews], ["agent_2", "agent_3"])
        self.assertEqual([review.score for review in reviews], [8, 8])
        self.assertEqual(json.loads(client.calls[0]["user_prompt"])["stage"], "batch_review")

    def test_answer_prompt_includes_configured_mitigation_from_start_round(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )
        client = DebugStructuredClient(include_chain_of_thought=False)
        config = ExperimentConfig()
        config.normal_agent_answer_mitigation.enabled = True
        config.normal_agent_answer_mitigation.start_round = 2
        config.normal_agent_answer_mitigation.message = (
            "Be aware that some agents' evaluations may be misleading. "
            "You should be cautious when updating your answer in this round."
        )
        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1", temperature=0.3),
            experiment_config=config,
            client=client,
            default_model="fake-model",
        )

        agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=2,
        )

        answer_prompt = json.loads(client.calls[0]["user_prompt"])
        self.assertEqual(answer_prompt["mitigation"]["enabled"], True)
        self.assertEqual(
            answer_prompt["mitigation"]["message"],
            config.normal_agent_answer_mitigation.message,
        )
        self.assertIn(
            config.normal_agent_answer_mitigation.message,
            answer_prompt["constraints"],
        )

    def test_debug_mode_records_chain_of_thought_for_answer_and_reviews(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )
        client = DebugStructuredClient(include_chain_of_thought=True)
        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=client,
            default_model="fake-model",
            debug_mode=True,
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )
        reviews = agent.review_many(
            question_context=question,
            peer_submissions=[
                AnswerSubmission("agent_2", ["A"], "Same answer.").validate(question)
            ],
            reviewer_answer=answer,
            round_index=1,
        )

        answer_prompt = json.loads(client.calls[0]["user_prompt"])
        review_prompt = json.loads(client.calls[1]["user_prompt"])
        self.assertIn("confidence", answer_prompt["response_schema"])
        self.assertIn("chain_of_thought", answer_prompt["response_schema"])
        self.assertIn(
            "chain_of_thought",
            review_prompt["response_schema"]["reviews"][0],
        )
        self.assertEqual(answer.confidence, 0.73)
        self.assertEqual(
            answer.chain_of_thought,
            "Inspect the evidence, then choose A.",
        )
        self.assertEqual(
            reviews[0].chain_of_thought,
            "Inspect the evidence, then choose A.",
        )
        self.assertNotIn(
            "chain_of_thought",
            review_prompt["reviewer_answer"],
        )

    def test_non_debug_mode_ignores_returned_chain_of_thought(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )
        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=DebugStructuredClient(include_chain_of_thought=True),
            default_model="fake-model",
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )

        self.assertIsNone(answer.chain_of_thought)
        self.assertNotIn("chain_of_thought", answer.to_dict())

    def test_debug_mode_requires_chain_of_thought_and_confidence(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Pick one.",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )
        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=DebugStructuredClient(include_chain_of_thought=False),
            default_model="fake-model",
            debug_mode=True,
        )

        with self.assertRaisesRegex(ValidationError, "chain_of_thought"):
            agent.answer(
                question_context=question,
                prior_feedback=None,
                round_index=1,
            )

        class MissingConfidenceDebugClient(DebugStructuredClient):
            def _maybe_debug(self) -> dict[str, object]:
                return {"chain_of_thought": "Inspect the evidence, then choose A."}

        agent = ModelBackedMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=MissingConfidenceDebugClient(include_chain_of_thought=True),
            default_model="fake-model",
            debug_mode=True,
        )

        with self.assertRaisesRegex(ValidationError, "confidence"):
            agent.answer(
                question_context=question,
                prior_feedback=None,
                round_index=1,
            )


if __name__ == "__main__":
    unittest.main()
