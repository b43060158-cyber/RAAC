from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from autogen_mas.config import (
    AgentConfig,
    AlignmentJudgeConfig,
    DashScopeSettings,
    ExperimentConfig,
    RuntimeConfig,
    SingleAgentConfig,
)
from autogen_mas.models import (
    AnswerSubmission,
    OptionRecord,
    QuestionRecord,
    ReviewSubmission,
    ShortAnswerQuestionRecord,
    ValidationError,
)
from autogen_mas.persistence import JsonRunStore
from autogen_mas.runtime.agent import MasAgent
from autogen_mas.runtime.batch_runner import BatchRuntimeOptions
from autogen_mas.runtime.single_agent_runner import BatchSingleAgentRunner, SingleAgentRunner
from autogen_mas.runtime.single_agent_runner import SingleAgentModelBackedAgent


class ScriptedSelfReviewAgent(MasAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self.answer_calls = 0
        self.review_calls = 0
        self.prior_feedback_payloads: list[dict | None] = []

    def answer(self, question_context, prior_feedback, round_index):
        self.answer_calls += 1
        self.prior_feedback_payloads.append(
            None if prior_feedback is None else prior_feedback.to_dict()
        )
        return AnswerSubmission(
            agent_id=self.agent_id,
            selected_option_ids=["A"],
            reasoning=f"{self.agent_id} round {round_index} answer",
            changed_answer=round_index > 1,
            change_summary="Updated after self-evaluation." if round_index > 1 else "",
        ).validate(question_context)

    def review(self, question_context, peer_submission, reviewer_answer, round_index):
        self.review_calls += 1
        return ReviewSubmission(
            reviewer_agent_id=self.agent_id,
            target_agent_id=peer_submission.agent_id,
            score=round_index + 5,
            stance="mixed",
            main_reason=f"Self-review for round {round_index}.",
        )


class ScriptedBatchSelfReviewAgent(ScriptedSelfReviewAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self.answer_many_calls = 0
        self.review_many_questions_calls = 0

    def answer_many(self, question_contexts, prior_feedback_by_question, round_index):
        self.answer_many_calls += 1
        return {
            question.question_key: self.answer(
                question_context=question,
                prior_feedback=prior_feedback_by_question.get(question.question_key),
                round_index=round_index,
            )
            for question in question_contexts
        }

    def review_many_questions(
        self,
        question_contexts,
        reviewer_answers_by_question,
        peer_submissions_by_question,
        round_index,
    ):
        self.review_many_questions_calls += 1
        return {
            question.question_key: [
                self.review(
                    question_context=question,
                    peer_submission=reviewer_answers_by_question[question.question_key],
                    reviewer_answer=reviewer_answers_by_question[question.question_key],
                    round_index=round_index,
                )
            ]
            for question in question_contexts
        }


class FailingSingleAgent(ScriptedSelfReviewAgent):
    def __init__(self, agent_config: AgentConfig, *, fail_question_keys: set[str]) -> None:
        super().__init__(agent_config)
        self.fail_question_keys = fail_question_keys

    def answer(self, question_context, prior_feedback, round_index):
        if question_context.question_key in self.fail_question_keys:
            raise ValidationError("synthetic invalid single-agent answer")
        return super().answer(question_context, prior_feedback, round_index)


class InspectingJsonRunStore(JsonRunStore):
    def __init__(self, output_dir: str | Path, *, marker: dict[str, object]) -> None:
        super().__init__(output_dir)
        self.marker = marker

    def persist_question_result(self, run_path, result):
        persisted = super().persist_question_result(run_path, result)
        if result.question_id == "q1":
            self.marker["q1_persisted_before_q2_round2_review_completed"] = not self.marker.get(
                "q2_round2_review_completed",
                False,
            )
            event = self.marker.get("q1_persisted_event")
            if isinstance(event, threading.Event):
                event.set()
        return persisted


class InspectingBatchSelfReviewAgent(ScriptedBatchSelfReviewAgent):
    def __init__(self, agent_config: AgentConfig, *, marker: dict[str, object]) -> None:
        super().__init__(agent_config)
        self.marker = marker

    def review(self, question_context, peer_submission, reviewer_answer, round_index):
        if question_context.question_id == "q2" and round_index == 2:
            self.marker["q2_round2_review_started"] = True
            event = self.marker.get("q1_persisted_event")
            if isinstance(event, threading.Event):
                event.wait(timeout=1.0)
            review = super().review(
                question_context=question_context,
                peer_submission=peer_submission,
                reviewer_answer=reviewer_answer,
                round_index=round_index,
            )
            self.marker["q2_round2_review_completed"] = True
            return review
        return super().review(question_context, peer_submission, reviewer_answer, round_index)


class FakeSingleAgentStructuredClient:
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
        stage = payload["stage"]
        if stage == "single_agent_answer":
            return {
                "selected_option_ids": ["A"],
                "reasoning": "A is best.",
                "changed_answer": True,
                "change_drivers": ["single_agent"],
                "change_summary": "Updated after self-review.",
            }
        if stage == "single_agent_batch_answer":
            return {
                "answers": [
                    {
                        "question_key": question["question_key"],
                        "selected_option_ids": ["A"],
                        "reasoning": "A is best.",
                        "changed_answer": True,
                        "change_drivers": ["single_agent"],
                        "change_summary": "Updated after self-review.",
                    }
                    for question in payload["question_contexts"]
                ]
            }
        if stage == "single_agent_self_review":
            return {
                "score": 8,
                "stance": "support",
                "main_reason": "The answer is well supported.",
            }
        if stage == "single_agent_batch_self_review":
            return {
                "reviews": [
                    {
                        "question_key": question["question_key"],
                        "score": 8,
                        "stance": "support",
                        "main_reason": "The answer is well supported.",
                    }
                    for question in payload["question_contexts"]
                ]
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class SingleAgentRunnerTest(unittest.TestCase):
    def test_single_agent_parse_answer_coerces_string_option_id(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
        )
        agent = SingleAgentModelBackedAgent(
            agent_config=config.single_agent.to_agent_config(),
            experiment_config=config,
            client=FakeSingleAgentStructuredClient(),
            default_model="fake-model",
        )
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Question?",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        answer = agent._parse_answer(
            question=question,
            raw_answer={
                "selected_option_ids": "A",
                "reasoning": "Choose A.",
            },
        )

        self.assertEqual(answer.selected_option_ids, ["A"])

    def test_single_agent_parse_answer_uses_alignment_judge_when_configured(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            alignment_judge=AlignmentJudgeConfig(
                enabled=True,
                judges=[AgentConfig(agent_id="judge_1", model="judge-model", temperature=0.0)],
            ),
        )

        class JudgeClient(FakeSingleAgentStructuredClient):
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
                if payload["stage"] == "alignment_judge":
                    return {
                        "selected_candidate": "structured",
                        "resolved_option_ids": ["A"],
                        "main_reason": "Structured answer better matches the defended conclusion.",
                    }
                return super().generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                )

        agent = SingleAgentModelBackedAgent(
            agent_config=config.single_agent.to_agent_config(),
            experiment_config=config,
            client=JudgeClient(),
            default_model="fake-model",
        )
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Question?",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        answer = agent._parse_answer(
            question=question,
            raw_answer={
                "selected_option_ids": ["A"],
                "reasoning": "A stray sentence says the correct answer is B, but the defended conclusion is A.",
            },
        )

        self.assertEqual(answer.selected_option_ids, ["A"])
        self.assertEqual(answer.alignment_resolution, "llm_judge_structured")

    def test_single_agent_parse_answer_uses_alignment_repair_judge_for_inconsistent_reasoning(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            alignment_judge=AlignmentJudgeConfig(
                enabled=True,
                judges=[AgentConfig(agent_id="judge_1", model="judge-model", temperature=0.0)],
            ),
        )

        class JudgeRepairClient(FakeSingleAgentStructuredClient):
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
                if payload["stage"] == "alignment_judge_repair":
                    return {
                        "resolved_option_ids": ["B"],
                        "main_reason": "The reasoning rejects A, so B is the better match.",
                    }
                return super().generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                )

        agent = SingleAgentModelBackedAgent(
            agent_config=config.single_agent.to_agent_config(),
            experiment_config=config,
            client=JudgeRepairClient(),
            default_model="fake-model",
        )
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Question?",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )

        answer = agent._parse_answer(
            question=question,
            raw_answer={
                "selected_option_ids": ["A"],
                "reasoning": "Option A is incorrect because it violates the key condition.",
            },
        )

        self.assertEqual(answer.selected_option_ids, ["B"])
        self.assertEqual(answer.alignment_resolution, "llm_judge_inferred")

    def test_run_dataset_answers_reviews_self_and_feeds_previous_self_review(self) -> None:
        questions = [
            QuestionRecord(
                question_id=f"q{index}",
                dataset_name="demo",
                task_type="single_choice",
                question=f"Question {index}?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            )
            for index in range(1, 3)
        ]
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(
                agent_id="baseline",
                model="baseline-model",
                temperature=0.1,
                run_id_prefix="baseline-run",
            ),
            runtime=RuntimeConfig(num_rounds=3, max_workers=2, output_dir="runs"),
        )
        scripted_agent = ScriptedSelfReviewAgent(config.single_agent.to_agent_config())

        def factory(experiment_config, llm_settings):
            return scripted_agent

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SingleAgentRunner(
                experiment_config=config,
                llm_settings=DashScopeSettings(api_key="test-key", model="default-model"),
                store=JsonRunStore(tmp_dir),
                agent_factory=factory,
            )
            artifacts = runner.run_dataset(questions)

            self.assertTrue(artifacts.run_id.startswith("baseline-run-"))
            self.assertEqual(len(artifacts.question_paths), 2)
            self.assertEqual(scripted_agent.answer_calls, 6)
            self.assertEqual(scripted_agent.review_calls, 6)
            self.assertEqual(
                scripted_agent.prior_feedback_payloads[0],
                None,
            )
            for payload in scripted_agent.prior_feedback_payloads:
                if payload is None:
                    continue
                self.assertIn("previous_answer", payload)
                self.assertIn("self_evaluation", payload)
                self.assertNotIn("total_score", payload)
                self.assertNotIn("average_score", payload)
                self.assertNotIn("feedback_summary", payload)

            for path in artifacts.question_paths:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                self.assertEqual(len(payload["rounds"]), 3)
                for round_index, round_payload in enumerate(payload["rounds"], start=1):
                    self.assertEqual(round_payload["round_index"], round_index)
                    agent_results = round_payload["agent_results"]
                    self.assertEqual(len(agent_results), 1)
                    agent_result = agent_results[0]
                    self.assertEqual(agent_result["agent_id"], "baseline")
                    self.assertEqual(
                        agent_result["answer"]["reasoning"],
                        f"baseline round {round_index} answer",
                    )
                    self.assertEqual(len(agent_result["reviews_given"]), 1)
                    self.assertEqual(len(agent_result["received_reviews"]), 1)
                    self.assertEqual(
                        agent_result["reviews_given"],
                        agent_result["received_reviews"],
                    )
                    self.assertEqual(
                        agent_result["reviews_given"][0]["reviewer_agent_id"],
                        "baseline",
                    )
                    self.assertEqual(
                        agent_result["reviews_given"][0]["target_agent_id"],
                        "baseline",
                    )
                    self.assertEqual(agent_result["reviews_given"][0]["score"], round_index + 5)
                    self.assertEqual(agent_result["total_score"], float(round_index + 5))
                    self.assertEqual(agent_result["average_score"], float(round_index + 5))

    def test_run_dataset_resume_skips_completed_questions(self) -> None:
        questions = [
            QuestionRecord(
                question_id=f"q{index}",
                dataset_name="demo",
                task_type="single_choice",
                question=f"Question {index}?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            )
            for index in range(1, 3)
        ]
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(
                agent_id="baseline",
                run_id_prefix="baseline-run",
            ),
            runtime=RuntimeConfig(num_rounds=2, max_workers=2, output_dir="runs"),
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")

        with tempfile.TemporaryDirectory() as tmp_dir:
            initial_agent = ScriptedSelfReviewAgent(config.single_agent.to_agent_config())
            initial_runner = SingleAgentRunner(
                experiment_config=config,
                llm_settings=settings,
                store=JsonRunStore(tmp_dir),
                agent_factory=lambda experiment_config, llm_settings: initial_agent,
            )
            initial_artifacts = initial_runner.run_dataset([questions[0]])

            resumed_agent = ScriptedSelfReviewAgent(config.single_agent.to_agent_config())
            resumed_runner = SingleAgentRunner(
                experiment_config=config,
                llm_settings=settings,
                store=JsonRunStore(tmp_dir),
                agent_factory=lambda experiment_config, llm_settings: resumed_agent,
            )
            resumed_artifacts = resumed_runner.run_dataset(
                questions,
                resume_run_path=initial_artifacts.run_path,
            )

        self.assertEqual(resumed_artifacts.run_path, initial_artifacts.run_path)
        self.assertEqual(len(resumed_artifacts.question_paths), 2)
        self.assertEqual(resumed_agent.answer_calls, 2)
        self.assertEqual(resumed_agent.review_calls, 2)
        self.assertTrue(
            resumed_artifacts.question_paths[0].endswith("questions/demo__q1__single_choice.json")
        )
        self.assertTrue(
            resumed_artifacts.question_paths[1].endswith("questions/demo__q2__single_choice.json")
        )

    def test_run_dataset_without_self_reflection_answers_once(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Question 1?",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(
                agent_id="baseline",
                enable_self_reflection=False,
                run_id_prefix="baseline-run",
            ),
            runtime=RuntimeConfig(num_rounds=1, max_workers=2, output_dir="runs"),
        )
        scripted_agent = ScriptedSelfReviewAgent(config.single_agent.to_agent_config())

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SingleAgentRunner(
                experiment_config=config,
                llm_settings=DashScopeSettings(api_key="test-key", model="default-model"),
                store=JsonRunStore(tmp_dir),
                agent_factory=lambda experiment_config, llm_settings: scripted_agent,
            )
            artifacts = runner.run_dataset([question])
            payload = json.loads(Path(artifacts.question_paths[0]).read_text(encoding="utf-8"))

        self.assertEqual(scripted_agent.answer_calls, 1)
        self.assertEqual(scripted_agent.review_calls, 0)
        self.assertEqual(scripted_agent.prior_feedback_payloads, [None])
        agent_result = payload["rounds"][0]["agent_results"][0]
        self.assertEqual(agent_result["reviews_given"], [])
        self.assertEqual(agent_result["received_reviews"], [])
        self.assertEqual(agent_result["total_score"], 0.0)
        self.assertEqual(agent_result["average_score"], 0.0)

    def test_run_dataset_continues_after_single_question_failure(self) -> None:
        questions = [
            QuestionRecord(
                question_id=f"q{index}",
                dataset_name="demo",
                task_type="single_choice",
                question=f"Question {index}?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            )
            for index in range(1, 4)
        ]
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(
                agent_id="baseline",
                enable_self_reflection=False,
                run_id_prefix="baseline-run",
            ),
            runtime=RuntimeConfig(num_rounds=1, max_workers=2, output_dir="runs"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            runner = SingleAgentRunner(
                experiment_config=config,
                llm_settings=DashScopeSettings(api_key="test-key", model="default-model"),
                store=store,
                agent_factory=lambda experiment_config, llm_settings: FailingSingleAgent(
                    experiment_config.single_agent.to_agent_config(),
                    fail_question_keys={"demo__q2__single_choice"},
                ),
            )

            artifacts = runner.run_dataset(questions)
            failed_questions = store.read_failed_questions(artifacts.run_path)

        self.assertEqual(len(artifacts.question_paths), 2)
        self.assertIsNotNone(failed_questions)
        self.assertEqual(len(failed_questions), 1)
        self.assertEqual(failed_questions[0]["question_index"], 2)
        self.assertEqual(failed_questions[0]["question_key"], "demo__q2__single_choice")
        persisted = {Path(path).stem for path in artifacts.question_paths}
        self.assertEqual(
            persisted,
            {"demo__q1__single_choice", "demo__q3__single_choice"},
        )

    def test_batch_runner_batches_answers_and_self_reviews_round_first(self) -> None:
        questions = [
            QuestionRecord(
                question_id=f"q{index}",
                dataset_name="demo",
                task_type="single_choice",
                question=f"Question {index}?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            )
            for index in range(1, 4)
        ]
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(
                agent_id="baseline",
                run_id_prefix="baseline-run",
            ),
            runtime=RuntimeConfig(num_rounds=2, max_workers=2, output_dir="runs"),
        )
        scripted_agent = ScriptedBatchSelfReviewAgent(config.single_agent.to_agent_config())

        def factory(experiment_config, llm_settings):
            return scripted_agent

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = BatchSingleAgentRunner(
                experiment_config=config,
                llm_settings=DashScopeSettings(api_key="test-key", model="default-model"),
                store=JsonRunStore(tmp_dir),
                agent_factory=factory,
                batch_options=BatchRuntimeOptions(
                    answer_max_input_tokens=50_000,
                    review_max_input_tokens=50_000,
                    answer_max_questions_per_batch=2,
                    review_max_questions_per_batch=2,
                ),
            )
            artifacts = runner.run_dataset(questions)
            events = (Path(artifacts.run_path) / "batch_events.jsonl").read_text(
                encoding="utf-8"
            )
            integrity_summary = json.loads(
                (Path(artifacts.run_path) / "batch_integrity_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            payloads = [
                json.loads(Path(path).read_text(encoding="utf-8"))
                for path in artifacts.question_paths
            ]

        self.assertEqual(len(artifacts.question_paths), 3)
        self.assertEqual(scripted_agent.answer_many_calls, 4)
        self.assertEqual(scripted_agent.review_many_questions_calls, 4)
        self.assertIn('"stage": "answer"', events)
        self.assertIn('"stage": "self_review"', events)
        self.assertEqual(integrity_summary["final_failed_question_count"], 0)
        self.assertIn("answer", integrity_summary["stage_counts"])
        self.assertIn("self_review", integrity_summary["stage_counts"])
        for payload in payloads:
            self.assertEqual(len(payload["rounds"]), 2)
            self.assertEqual(
                [
                    payload["rounds"][0]["agent_results"][0]["answer"]["reasoning"],
                    payload["rounds"][1]["agent_results"][0]["answer"]["reasoning"],
                ],
                ["baseline round 1 answer", "baseline round 2 answer"],
            )

    def test_batch_runner_without_self_reflection_answers_once(self) -> None:
        questions = [
            QuestionRecord(
                question_id=f"q{index}",
                dataset_name="demo",
                task_type="single_choice",
                question=f"Question {index}?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            )
            for index in range(1, 3)
        ]
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(
                agent_id="baseline",
                enable_self_reflection=False,
                run_id_prefix="baseline-run",
            ),
            runtime=RuntimeConfig(num_rounds=1, max_workers=2, output_dir="runs"),
        )
        scripted_agent = ScriptedBatchSelfReviewAgent(config.single_agent.to_agent_config())

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = BatchSingleAgentRunner(
                experiment_config=config,
                llm_settings=DashScopeSettings(api_key="test-key", model="default-model"),
                store=JsonRunStore(tmp_dir),
                agent_factory=lambda experiment_config, llm_settings: scripted_agent,
                batch_options=BatchRuntimeOptions(
                    answer_max_input_tokens=50_000,
                    review_max_input_tokens=50_000,
                    answer_max_questions_per_batch=2,
                    review_max_questions_per_batch=2,
                ),
            )
            artifacts = runner.run_dataset(questions)
            payloads = [
                json.loads(Path(path).read_text(encoding="utf-8"))
                for path in artifacts.question_paths
            ]

        self.assertEqual(scripted_agent.answer_many_calls, 1)
        self.assertEqual(scripted_agent.review_many_questions_calls, 0)
        self.assertEqual(scripted_agent.review_calls, 0)
        for payload in payloads:
            agent_result = payload["rounds"][0]["agent_results"][0]
            self.assertEqual(agent_result["reviews_given"], [])
            self.assertEqual(agent_result["received_reviews"], [])
            self.assertEqual(agent_result["total_score"], 0.0)
            self.assertEqual(agent_result["average_score"], 0.0)

    def test_model_backed_single_agent_uses_single_agent_prompt_contract(self) -> None:
        question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Question?",
            options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
            correct_option_ids=["A"],
        )
        client = FakeSingleAgentStructuredClient()
        agent = SingleAgentModelBackedAgent(
            agent_config=AgentConfig(agent_id="single_agent", temperature=0.3),
            experiment_config=ExperimentConfig(),
            client=client,
            default_model="fake-model",
        )

        answer = agent.answer_many(
            question_contexts=[question],
            prior_feedback_by_question={question.question_key: None},
            round_index=2,
        )[question.question_key]

        prompt_payload = json.loads(client.calls[0]["user_prompt"])
        self.assertEqual(prompt_payload["stage"], "single_agent_batch_answer")
        self.assertNotIn("change_drivers", prompt_payload["response_schema"]["answers"][0])
        self.assertTrue(answer.changed_answer)
        self.assertEqual(answer.change_drivers, [])
        self.assertEqual(answer.change_summary, "Updated after self-review.")

    def test_model_backed_single_agent_uses_dataset_specific_prompt(self) -> None:
        question = ShortAnswerQuestionRecord(
            question_id="q1",
            dataset_name="ciar",
            task_type="math_short_answer",
            question="What is 1+1?",
            acceptable_answers=["2"],
            adversarial_target_answers=["3"],
        )

        class ShortAnswerClient(FakeSingleAgentStructuredClient):
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
                    "final_answer": "2",
                    "reasoning": "1+1 = 2.",
                    "changed_answer": False,
                    "change_summary": "",
                }

        agent = SingleAgentModelBackedAgent(
            agent_config=AgentConfig(agent_id="single_agent", temperature=0.3),
            experiment_config=ExperimentConfig(
                single_agent=SingleAgentConfig(
                    answer_prompt="Default baseline prompt",
                    answer_prompts_by_dataset={"ciar": "CIAR-specific baseline prompt"},
                )
            ),
            client=ShortAnswerClient(),
            default_model="fake-model",
        )

        answer = agent.answer(
            question_context=question,
            prior_feedback=None,
            round_index=1,
        )

        self.assertEqual(answer.final_answer, "2")
        self.assertEqual(
            agent.client.calls[0]["system_prompt"],
            "CIAR-specific baseline prompt",
        )
        self.assertEqual(
            agent.answer_system_prompt_for_dataset("truthfulqa"),
            "Default baseline prompt",
        )

    def test_model_backed_single_agent_batch_rejects_mixed_datasets(self) -> None:
        questions = [
            QuestionRecord(
                question_id="q1",
                dataset_name="ciar",
                task_type="math_short_answer",
                question="What is 1+1?",
                options=[],
                correct_option_ids=[],
            ),
            QuestionRecord(
                question_id="q2",
                dataset_name="truthfulqa",
                task_type="single_choice",
                question="Question?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            ),
        ]
        agent = SingleAgentModelBackedAgent(
            agent_config=AgentConfig(agent_id="single_agent", temperature=0.3),
            experiment_config=ExperimentConfig(),
            client=FakeSingleAgentStructuredClient(),
            default_model="fake-model",
        )

        with self.assertRaisesRegex(ValidationError, "require a single dataset"):
            agent.answer_many(
                question_contexts=questions,
                prior_feedback_by_question={
                    question.question_key: None for question in questions
                },
                round_index=1,
            )

    def test_medmcqa_single_agent_uses_medical_single_choice_prompt(self) -> None:
        agent = SingleAgentModelBackedAgent(
            agent_config=AgentConfig(agent_id="single_agent", temperature=0.3),
            experiment_config=ExperimentConfig(
                single_agent=SingleAgentConfig(enable_self_reflection=True)
            ),
            client=FakeSingleAgentStructuredClient(),
            default_model="fake-model",
        )

        prompt = agent.answer_system_prompt_for_dataset("medmcqa")

        self.assertIn("medical single-choice QA baseline", prompt)
        self.assertIn("choose exactly one option id", prompt)
        self.assertIn("all of the following", prompt)
        self.assertIn("prior self-evaluation", prompt)
        self.assertNotIn("final_answer", prompt)
        self.assertNotIn("short-answer", prompt)

    def test_medmcqa_single_agent_no_reflection_prompt_omits_self_evaluation(self) -> None:
        agent = SingleAgentModelBackedAgent(
            agent_config=AgentConfig(agent_id="single_agent", temperature=0.3),
            experiment_config=ExperimentConfig(
                single_agent=SingleAgentConfig(enable_self_reflection=False)
            ),
            client=FakeSingleAgentStructuredClient(),
            default_model="fake-model",
        )

        prompt = agent.answer_system_prompt_for_dataset("medmcqa")

        self.assertIn("medical single-choice QA baseline", prompt)
        self.assertIn("choose exactly one option id", prompt)
        self.assertIn("no self-evaluation rounds", prompt)
        self.assertNotIn("prior self-evaluation", prompt)
        self.assertNotIn("short-answer", prompt)

    def test_faireval_single_agent_uses_response_comparison_prompt(self) -> None:
        agent = SingleAgentModelBackedAgent(
            agent_config=AgentConfig(agent_id="single_agent", temperature=0.3),
            experiment_config=ExperimentConfig(
                single_agent=SingleAgentConfig(enable_self_reflection=True)
            ),
            client=FakeSingleAgentStructuredClient(),
            default_model="fake-model",
        )

        prompt = agent.answer_system_prompt_for_dataset("faireval")

        self.assertIn("response-comparison baseline", prompt)
        self.assertIn("compare the two candidate responses", prompt)
        self.assertIn("Choose A if response 1 is better", prompt)
        self.assertIn("If the two responses are hard to distinguish overall, choose B.", prompt)
        self.assertIn("prior self-evaluation", prompt)

    def test_chess_single_agent_uses_chess_specific_prompt(self) -> None:
        agent = SingleAgentModelBackedAgent(
            agent_config=AgentConfig(agent_id="single_agent", temperature=0.3),
            experiment_config=ExperimentConfig(
                single_agent=SingleAgentConfig(enable_self_reflection=True)
            ),
            client=FakeSingleAgentStructuredClient(),
            default_model="fake-model",
        )

        prompt = agent.answer_system_prompt_for_dataset("chess")

        self.assertIn("chess state-tracking baseline", prompt)
        self.assertIn("exactly one legal destination square", prompt)
        self.assertIn("[a-h][1-8]", prompt)
        self.assertIn("prior self-evaluation", prompt)

    def test_single_agent_runner_builds_decoupled_model_agent(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="single_agent"),
            runtime=RuntimeConfig(num_rounds=1, max_workers=1, output_dir="runs"),
        )
        runner = SingleAgentRunner(
            experiment_config=config,
            llm_settings=DashScopeSettings(api_key="test-key", model="default-model"),
        )

        agent = runner._build_agent(config)

        self.assertIsInstance(agent, SingleAgentModelBackedAgent)

    def test_batch_runner_persists_questions_as_soon_as_complete(self) -> None:
        questions = [
            QuestionRecord(
                question_id=f"q{index}",
                dataset_name="demo",
                task_type="single_choice",
                question=f"Question {index}?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            )
            for index in range(1, 3)
        ]
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(
                agent_id="baseline",
                run_id_prefix="baseline-run",
            ),
            runtime=RuntimeConfig(num_rounds=2, max_workers=1, output_dir="runs"),
        )
        marker: dict[str, object] = {"q1_persisted_event": threading.Event()}

        def factory(experiment_config, llm_settings):
            return InspectingBatchSelfReviewAgent(
                experiment_config.single_agent.to_agent_config(),
                marker=marker,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = BatchSingleAgentRunner(
                experiment_config=config,
                llm_settings=DashScopeSettings(api_key="test-key", model="default-model"),
                store=InspectingJsonRunStore(tmp_dir, marker=marker),
                agent_factory=factory,
                batch_options=BatchRuntimeOptions(
                    answer_max_input_tokens=50_000,
                    review_max_input_tokens=50_000,
                    answer_max_questions_per_batch=1,
                    review_max_questions_per_batch=1,
                ),
            )
            artifacts = runner.run_dataset(questions)

        self.assertEqual(len(artifacts.question_paths), 2)
        self.assertIs(marker.get("q1_persisted_before_q2_round2_review_completed"), True)

    def test_batch_runner_resume_skips_completed_questions_even_when_non_contiguous(self) -> None:
        questions = [
            QuestionRecord(
                question_id=f"q{index}",
                dataset_name="demo",
                task_type="single_choice",
                question=f"Question {index}?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            )
            for index in range(1, 4)
        ]
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(
                agent_id="baseline",
                run_id_prefix="baseline-run",
            ),
            runtime=RuntimeConfig(num_rounds=2, max_workers=2, output_dir="runs"),
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")

        with tempfile.TemporaryDirectory() as tmp_dir:
            initial_agent = ScriptedBatchSelfReviewAgent(config.single_agent.to_agent_config())
            initial_runner = BatchSingleAgentRunner(
                experiment_config=config,
                llm_settings=settings,
                store=JsonRunStore(tmp_dir),
                agent_factory=lambda experiment_config, llm_settings: initial_agent,
                batch_options=BatchRuntimeOptions(
                    answer_max_input_tokens=50_000,
                    review_max_input_tokens=50_000,
                    answer_max_questions_per_batch=1,
                    review_max_questions_per_batch=1,
                ),
            )
            initial_artifacts = initial_runner.run_dataset([questions[1]])

            resumed_agent = ScriptedBatchSelfReviewAgent(config.single_agent.to_agent_config())
            resumed_runner = BatchSingleAgentRunner(
                experiment_config=config,
                llm_settings=settings,
                store=JsonRunStore(tmp_dir),
                agent_factory=lambda experiment_config, llm_settings: resumed_agent,
                batch_options=BatchRuntimeOptions(
                    answer_max_input_tokens=50_000,
                    review_max_input_tokens=50_000,
                    answer_max_questions_per_batch=2,
                    review_max_questions_per_batch=2,
                ),
            )
            resumed_artifacts = resumed_runner.run_dataset(
                questions,
                resume_run_path=initial_artifacts.run_path,
            )

        self.assertEqual(resumed_artifacts.run_path, initial_artifacts.run_path)
        self.assertEqual(len(resumed_artifacts.question_paths), 3)
        self.assertEqual(resumed_agent.answer_many_calls, 2)
        self.assertEqual(resumed_agent.review_many_questions_calls, 2)
        self.assertEqual(resumed_agent.answer_calls, 4)
        self.assertEqual(resumed_agent.review_calls, 4)
        self.assertTrue(
            any(path.endswith("questions/demo__q2__single_choice.json") for path in resumed_artifacts.question_paths)
        )


if __name__ == "__main__":
    unittest.main()
