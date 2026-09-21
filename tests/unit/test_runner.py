from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autogen_mas.config import AgentConfig, DashScopeSettings, ExperimentConfig, RuntimeConfig
from autogen_mas.models import (
    AnswerSubmission,
    CodeQuestionRecord,
    OptionRecord,
    QuestionRoundResult,
    QuestionRecord,
    QuestionRunResult,
    ReviewSubmission,
    RoundAgentResult,
    ShortAnswerQuestionRecord,
)
from autogen_mas.persistence import JsonRunStore
from autogen_mas.runtime import CompetitionRunner, MasAgent
from autogen_mas.runtime.agent import ModelBackedMasAgent


class ScriptedMasAgent(MasAgent):
    def __init__(
        self,
        agent_config: AgentConfig,
        answer_script: dict[int, AnswerSubmission],
        review_script: dict[int, dict[str, ReviewSubmission]],
    ) -> None:
        super().__init__(agent_config)
        self.answer_script = answer_script
        self.review_script = review_script
        self.answer_calls: list[tuple[int, object]] = []

    def answer(self, question_context, prior_feedback, round_index):
        self.answer_calls.append((round_index, prior_feedback))
        submission = self.answer_script[round_index]
        return AnswerSubmission(
            agent_id=self.agent_id,
            selected_option_ids=list(submission.selected_option_ids),
            reasoning=submission.reasoning,
            code=submission.code,
            final_answer=submission.final_answer,
            confidence=submission.confidence,
            changed_answer=submission.changed_answer,
            change_drivers=list(submission.change_drivers),
            change_summary=submission.change_summary,
        ).validate(question_context)

    def review(self, question_context, peer_submission, reviewer_answer, round_index):
        review = self.review_script[round_index][peer_submission.agent_id]
        return ReviewSubmission(
            reviewer_agent_id=self.agent_id,
            target_agent_id=peer_submission.agent_id,
            score=review.score,
            stance=review.stance,
            main_reason=review.main_reason,
        ).validate()


class FailingMasAgent(MasAgent):
    def __init__(self, agent_config: AgentConfig, *, fail_question_keys: set[str]) -> None:
        super().__init__(agent_config)
        self.fail_question_keys = set(fail_question_keys)

    def answer(self, question_context, prior_feedback, round_index):
        if question_context.question_key in self.fail_question_keys:
            raise RuntimeError(f"boom for {question_context.question_key}")
        return AnswerSubmission(
            agent_id=self.agent_id,
            selected_option_ids=["A"],
            reasoning="Option A is correct.",
        ).validate(question_context)

    def review(self, question_context, peer_submission, reviewer_answer, round_index):
        return ReviewSubmission(
            reviewer_agent_id=self.agent_id,
            target_agent_id=peer_submission.agent_id,
            score=8,
            stance="support",
            main_reason="Looks complete.",
        ).validate()


class CompetitionRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.question = QuestionRecord(
            question_id="q1",
            dataset_name="demo",
            task_type="single_choice",
            question="Which option is best?",
            options=[
                OptionRecord("A", "Alpha"),
                OptionRecord("B", "Beta"),
            ],
            correct_option_ids=["A"],
        )
        self.experiment_config = ExperimentConfig(
            agents=[
                AgentConfig(agent_id="agent_1"),
                AgentConfig(agent_id="agent_2"),
                AgentConfig(agent_id="agent_3"),
            ],
            runtime=RuntimeConfig(num_rounds=2, max_workers=6, output_dir="runs"),
        )
        self.llm_settings = DashScopeSettings(api_key="test-key")
        self.scripted_agents: list[ScriptedMasAgent] = []

    def _questions(self, count: int) -> list[QuestionRecord]:
        return [
            QuestionRecord(
                question_id=f"q{index}",
                dataset_name="demo",
                task_type="single_choice",
                question=f"Question {index}?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            )
            for index in range(1, count + 1)
        ]

    def test_clean_runner_enables_honest_answer_repair(self) -> None:
        runner = CompetitionRunner(
            experiment_config=self.experiment_config,
            llm_settings=self.llm_settings,
        )

        with patch(
            "autogen_mas.runtime.runner.build_structured_client",
            return_value=object(),
        ):
            agents = runner._build_agents()

        self.assertTrue(all(isinstance(agent, ModelBackedMasAgent) for agent in agents))
        self.assertTrue(all(agent.enable_answer_repair for agent in agents))

    def _complete_question_result(
        self,
        *,
        run_id: str,
        question: QuestionRecord,
        agent_ids: list[str],
        num_rounds: int,
    ) -> QuestionRunResult:
        rounds: list[QuestionRoundResult] = []
        for round_index in range(1, num_rounds + 1):
            agent_results: list[RoundAgentResult] = []
            for agent_id in agent_ids:
                peer_ids = [peer_id for peer_id in agent_ids if peer_id != agent_id]
                reviews_given = [
                    ReviewSubmission(
                        agent_id,
                        peer_id,
                        8,
                        "support",
                        "Looks complete.",
                    ).validate()
                    for peer_id in peer_ids
                ]
                received_reviews = [
                    ReviewSubmission(
                        peer_id,
                        agent_id,
                        8,
                        "support",
                        "Looks complete.",
                    ).validate()
                    for peer_id in peer_ids
                ]
                agent_results.append(
                    RoundAgentResult(
                        agent_id=agent_id,
                        answer=AnswerSubmission(
                            agent_id,
                            ["A"],
                            f"Round {round_index} answer.",
                        ).validate(question),
                        reviews_given=reviews_given,
                        received_reviews=received_reviews,
                        total_score=float(sum(review.score for review in received_reviews)),
                        average_score=8.0 if received_reviews else 0.0,
                    )
                )
            rounds.append(QuestionRoundResult(round_index=round_index, agent_results=agent_results))
        return QuestionRunResult.from_task(run_id=run_id, task=question, rounds=rounds)

    def _resume_agent_factory(self, config: ExperimentConfig, all_agents: list[ScriptedMasAgent]):
        def factory(experiment_config, llm_settings):
            agents: list[ScriptedMasAgent] = []
            for agent_config in experiment_config.agents:
                peer_ids = [
                    peer.agent_id
                    for peer in experiment_config.agents
                    if peer.agent_id != agent_config.agent_id
                ]
                answer_script = {
                    round_index: AnswerSubmission(
                        agent_config.agent_id,
                        ["A"],
                        f"Round {round_index} answer from {agent_config.agent_id}.",
                    )
                    for round_index in range(1, experiment_config.runtime.num_rounds + 1)
                }
                review_script = {
                    round_index: {
                        peer_id: ReviewSubmission(
                            agent_config.agent_id,
                            peer_id,
                            8,
                            "support",
                            "Looks good.",
                        )
                        for peer_id in peer_ids
                    }
                    for round_index in range(1, experiment_config.runtime.num_rounds + 1)
                }
                agent = ScriptedMasAgent(agent_config, answer_script, review_script)
                agents.append(agent)
                all_agents.append(agent)
            return agents

        return factory

    def _support_review_scripts(
        self,
        *,
        agent_ids: list[str],
        round_indices: list[int],
    ) -> dict[str, dict[int, dict[str, ReviewSubmission]]]:
        return {
            agent_id: {
                round_index: {
                    peer_id: ReviewSubmission(
                        agent_id,
                        peer_id,
                        8,
                        "support",
                        "Looks good.",
                    )
                    for peer_id in agent_ids
                    if peer_id != agent_id
                }
                for round_index in round_indices
            }
            for agent_id in agent_ids
        }

    def test_multi_round_feedback_and_score_aggregation(self) -> None:
        answer_script = {
            "agent_1": {
                1: AnswerSubmission("agent_1", ["A"], "Round 1 answer from agent 1"),
                2: AnswerSubmission("agent_1", ["A"], "Round 2 answer from agent 1"),
            },
            "agent_2": {
                1: AnswerSubmission("agent_2", ["B"], "Round 1 answer from agent 2"),
                2: AnswerSubmission(
                    "agent_2",
                    ["A"],
                    "Round 2 answer from agent 2",
                    changed_answer=True,
                    change_drivers=["agent_1", "agent_3"],
                    change_summary="Peer feedback pushed agent 2 to revisit option A.",
                ),
            },
            "agent_3": {
                1: AnswerSubmission("agent_3", ["A"], "Round 1 answer from agent 3"),
                2: AnswerSubmission("agent_3", ["A"], "Round 2 answer from agent 3"),
            },
        }
        review_script = {
            "agent_1": {
                1: {
                    "agent_2": ReviewSubmission("agent_1", "agent_2", 7, "oppose", "Switch to A"),
                    "agent_3": ReviewSubmission("agent_1", "agent_3", 8, "support", "Correct choice."),
                },
                2: {
                    "agent_2": ReviewSubmission("agent_1", "agent_2", 9, "support", "Improved after revision."),
                    "agent_3": ReviewSubmission("agent_1", "agent_3", 8, "support", "Stable answer."),
                },
            },
            "agent_2": {
                1: {
                    "agent_1": ReviewSubmission("agent_2", "agent_1", 6, "mixed", "Reasoning needs more support."),
                    "agent_3": ReviewSubmission("agent_2", "agent_3", 9, "support", "Strong answer."),
                },
                2: {
                    "agent_1": ReviewSubmission("agent_2", "agent_1", 8, "support", "Good final answer."),
                    "agent_3": ReviewSubmission("agent_2", "agent_3", 9, "support", "Strong answer."),
                },
            },
            "agent_3": {
                1: {
                    "agent_1": ReviewSubmission("agent_3", "agent_1", 5, "mixed", "Ground the answer more clearly."),
                    "agent_2": ReviewSubmission("agent_3", "agent_2", 4, "oppose", "Re-evaluate option A."),
                },
                2: {
                    "agent_1": ReviewSubmission("agent_3", "agent_1", 8, "support", "Good final answer."),
                    "agent_2": ReviewSubmission("agent_3", "agent_2", 9, "support", "Corrected answer."),
                },
            },
        }

        def factory(experiment_config, llm_settings):
            agents: list[ScriptedMasAgent] = []
            for agent_config in experiment_config.agents:
                scripted = ScriptedMasAgent(
                    agent_config=agent_config,
                    answer_script=answer_script[agent_config.agent_id],
                    review_script=review_script[agent_config.agent_id],
                )
                agents.append(scripted)
            self.scripted_agents = agents
            return agents

        runner = CompetitionRunner(
            experiment_config=self.experiment_config,
            llm_settings=self.llm_settings,
            agent_factory=factory,
        )

        result = runner.run_question(self.question, run_id="test-run")

        self.assertEqual(result.run_id, "test-run")
        self.assertEqual(len(result.rounds), 2)

        round_1 = result.rounds[0].agent_map()
        self.assertEqual(round_1["agent_1"].total_score, 11.0)
        self.assertEqual(round_1["agent_1"].average_score, 5.5)
        self.assertEqual(round_1["agent_2"].total_score, 11.0)
        self.assertEqual(round_1["agent_3"].total_score, 17.0)

        round_2 = result.rounds[1].agent_map()
        self.assertEqual(round_2["agent_1"].total_score, 16.0)
        self.assertEqual(round_2["agent_2"].total_score, 18.0)
        self.assertEqual(round_2["agent_3"].average_score, 8.5)

        second_round_feedback = {
            agent.agent_id: agent.answer_calls[1][1] for agent in self.scripted_agents
        }
        self.assertIsNotNone(second_round_feedback["agent_1"])
        self.assertEqual(
            second_round_feedback["agent_1"].previous_answer.selected_option_ids,  # type: ignore[union-attr]
            ["A"],
        )
        self.assertEqual(
            second_round_feedback["agent_2"].key_reviews[0].target_agent_id,  # type: ignore[union-attr]
            "agent_2",
        )
        self.assertEqual(second_round_feedback["agent_3"].total_score, 17.0)  # type: ignore[union-attr]
        self.assertEqual(round_2["agent_2"].answer.change_drivers, ["agent_1", "agent_3"])

    def test_debug_run_persists_feedback_summary_artifact(self) -> None:
        config = ExperimentConfig(
            agents=[
                AgentConfig(agent_id="agent_1"),
                AgentConfig(agent_id="agent_2"),
            ],
            runtime=RuntimeConfig(num_rounds=2, max_workers=2, output_dir="runs"),
        )
        answer_script = {
            "agent_1": {
                1: AnswerSubmission("agent_1", ["A"], "Agent 1 first reason.", confidence=0.61),
                2: AnswerSubmission("agent_1", ["B"], "Agent 1 second reason.", confidence=0.77),
            },
            "agent_2": {
                1: AnswerSubmission("agent_2", ["B"], "Agent 2 first reason.", confidence=0.54),
                2: AnswerSubmission("agent_2", ["A"], "Agent 2 second reason.", confidence=0.8),
            },
        }
        review_script = {
            "agent_1": {
                1: {
                    "agent_2": ReviewSubmission(
                        "agent_1",
                        "agent_2",
                        9,
                        "support",
                        "Agent 2 is plausible.",
                    ),
                },
                2: {
                    "agent_2": ReviewSubmission(
                        "agent_1",
                        "agent_2",
                        4,
                        "oppose",
                        "Agent 2 should revise.",
                    ),
                },
            },
            "agent_2": {
                1: {
                    "agent_1": ReviewSubmission(
                        "agent_2",
                        "agent_1",
                        2,
                        "oppose",
                        "Agent 1 missed the key issue.",
                    ),
                },
                2: {
                    "agent_1": ReviewSubmission(
                        "agent_2",
                        "agent_1",
                        8,
                        "support",
                        "Agent 1 improved.",
                    ),
                },
            },
        }

        def factory(experiment_config, llm_settings):
            return [
                ScriptedMasAgent(
                    agent_config=agent_config,
                    answer_script=answer_script[agent_config.agent_id],
                    review_script=review_script[agent_config.agent_id],
                )
                for agent_config in experiment_config.agents
            ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            runner = CompetitionRunner(
                experiment_config=config,
                llm_settings=self.llm_settings,
                store=store,
                agent_factory=factory,
                debug_mode=True,
                persist_feedback_summary=True,
            )

            artifacts = runner.run_dataset([self.question])

            summary_path = store.feedback_summary_path(
                artifacts.run_path,
                self.question.question_key,
            )
            self.assertTrue(summary_path.exists())
            payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            question_payload = json.loads(
                store.question_result_path(artifacts.run_path, self.question.question_key).read_text(
                    encoding="utf-8"
                )
            )

        agent_1_item = next(
            item
            for item in payload["items"]
            if item["round_index"] == 2 and item["agent_id"] == "agent_1"
        )
        self.assertNotIn("feedback_summary", agent_1_item)
        self.assertEqual(
            agent_1_item["prior_feedback"]["feedback_summary"]["total_score"],
            2.0,
        )
        self.assertEqual(
            agent_1_item["prior_feedback"]["feedback_summary"]["oppose_count"],
            1,
        )
        self.assertEqual(
            agent_1_item["prior_feedback"]["previous_answer"]["reasoning"],
            "Agent 1 first reason.",
        )
        self.assertEqual(
            agent_1_item["prior_feedback"]["previous_answer"]["confidence"],
            0.61,
        )
        self.assertEqual(
            question_payload["rounds"][0]["agent_results"][0]["answer"]["confidence"],
            0.61,
        )
        self.assertEqual(
            agent_1_item["prior_feedback"]["feedback_summary"]["key_reviews"][0][
                "main_reason"
            ],
            "Agent 1 missed the key issue.",
        )

    def test_choice_consensus_stops_after_first_round(self) -> None:
        config = ExperimentConfig(
            agents=[
                AgentConfig(agent_id="agent_1"),
                AgentConfig(agent_id="agent_2"),
                AgentConfig(agent_id="agent_3"),
            ],
            runtime=RuntimeConfig(num_rounds=3, max_workers=3, output_dir="runs"),
        )
        agent_ids = [agent.agent_id for agent in config.agents]
        answer_script = {
            agent_id: {
                1: AnswerSubmission(agent_id, ["A"], f"{agent_id} picks A."),
            }
            for agent_id in agent_ids
        }
        review_script = self._support_review_scripts(
            agent_ids=agent_ids,
            round_indices=[1],
        )

        def factory(experiment_config, llm_settings):
            self.scripted_agents = [
                ScriptedMasAgent(
                    agent_config=agent_config,
                    answer_script=answer_script[agent_config.agent_id],
                    review_script=review_script[agent_config.agent_id],
                )
                for agent_config in experiment_config.agents
            ]
            return self.scripted_agents

        runner = CompetitionRunner(
            experiment_config=config,
            llm_settings=self.llm_settings,
            agent_factory=factory,
        )

        result = runner.run_question(self.question, run_id="choice-consensus")

        self.assertEqual(len(result.rounds), 1)
        self.assertTrue(result.rounds[0].reached_consensus)
        self.assertTrue(result.to_dict()["rounds"][0]["reached_consensus"])
        self.assertEqual(
            [call[0] for agent in self.scripted_agents for call in agent.answer_calls],
            [1, 1, 1],
        )

    def test_choice_consensus_can_continue_when_short_circuit_disabled(self) -> None:
        config = ExperimentConfig(
            agents=[
                AgentConfig(agent_id="agent_1"),
                AgentConfig(agent_id="agent_2"),
                AgentConfig(agent_id="agent_3"),
            ],
            runtime=RuntimeConfig(
                num_rounds=3,
                max_workers=3,
                output_dir="runs",
                consensus_short_circuit=False,
            ),
        )
        agent_ids = [agent.agent_id for agent in config.agents]
        answer_script = {
            agent_id: {
                round_index: AnswerSubmission(
                    agent_id,
                    ["A"],
                    f"{agent_id} picks A in round {round_index}.",
                )
                for round_index in (1, 2, 3)
            }
            for agent_id in agent_ids
        }
        review_script = self._support_review_scripts(
            agent_ids=agent_ids,
            round_indices=[1, 2, 3],
        )

        def factory(experiment_config, llm_settings):
            self.scripted_agents = [
                ScriptedMasAgent(
                    agent_config=agent_config,
                    answer_script=answer_script[agent_config.agent_id],
                    review_script=review_script[agent_config.agent_id],
                )
                for agent_config in experiment_config.agents
            ]
            return self.scripted_agents

        runner = CompetitionRunner(
            experiment_config=config,
            llm_settings=self.llm_settings,
            agent_factory=factory,
        )

        result = runner.run_question(self.question, run_id="choice-consensus-no-stop")

        self.assertEqual(len(result.rounds), 3)
        self.assertTrue(all(round_data.reached_consensus for round_data in result.rounds))
        self.assertEqual(
            [call[0] for agent in self.scripted_agents for call in agent.answer_calls],
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
        )

    def test_math_consensus_stops_after_equivalent_second_round_answers(self) -> None:
        question = ShortAnswerQuestionRecord(
            question_id="m1",
            dataset_name="ciar",
            task_type="math_short_answer",
            question="What is the value?",
            acceptable_answers=["1.5", "3/2"],
            adversarial_target_answers=["2"],
        )
        config = ExperimentConfig(
            agents=[
                AgentConfig(agent_id="agent_1"),
                AgentConfig(agent_id="agent_2"),
                AgentConfig(agent_id="agent_3"),
            ],
            runtime=RuntimeConfig(num_rounds=3, max_workers=3, output_dir="runs"),
        )
        answer_script = {
            "agent_1": {
                1: AnswerSubmission("agent_1", [], "Final answer is 1.5", final_answer="1.5"),
                2: AnswerSubmission("agent_1", [], "Final answer is 1.5", final_answer="1.5"),
            },
            "agent_2": {
                1: AnswerSubmission("agent_2", [], "Final answer is 2", final_answer="2"),
                2: AnswerSubmission("agent_2", [], "Final answer is 3/2", final_answer="3/2"),
            },
            "agent_3": {
                1: AnswerSubmission("agent_3", [], "Final answer is 1.5", final_answer="1.5"),
                2: AnswerSubmission("agent_3", [], "Final answer is 1.500", final_answer="1.500"),
            },
        }
        agent_ids = [agent.agent_id for agent in config.agents]
        review_script = self._support_review_scripts(
            agent_ids=agent_ids,
            round_indices=[1, 2],
        )

        def factory(experiment_config, llm_settings):
            self.scripted_agents = [
                ScriptedMasAgent(
                    agent_config=agent_config,
                    answer_script=answer_script[agent_config.agent_id],
                    review_script=review_script[agent_config.agent_id],
                )
                for agent_config in experiment_config.agents
            ]
            return self.scripted_agents

        runner = CompetitionRunner(
            experiment_config=config,
            llm_settings=self.llm_settings,
            agent_factory=factory,
        )

        result = runner.run_question(question, run_id="math-consensus")

        self.assertEqual(len(result.rounds), 2)
        self.assertFalse(result.rounds[0].reached_consensus)
        self.assertTrue(result.rounds[1].reached_consensus)

    def test_code_generation_does_not_short_circuit_on_identical_code(self) -> None:
        question = CodeQuestionRecord(
            question_id="0",
            dataset_name="humaneval",
            prompt="def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
            entry_point="add_one",
            test="def check(candidate):\n    assert candidate(1) == 2\n",
            source_path="data/HumanEval/HumanEval.jsonl",
            source_task_id="HumanEval/0",
        )
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="agent_1"), AgentConfig(agent_id="agent_2")],
            runtime=RuntimeConfig(num_rounds=2, max_workers=2, output_dir="runs"),
        )
        answer_script = {
            agent_id: {
                1: AnswerSubmission(
                    agent_id,
                    [],
                    "Same implementation.",
                    code="def add_one(x):\n    return x + 1\n",
                ),
                2: AnswerSubmission(
                    agent_id,
                    [],
                    "Same implementation.",
                    code="def add_one(x):\n    return x + 1\n",
                ),
            }
            for agent_id in ["agent_1", "agent_2"]
        }
        review_script = self._support_review_scripts(
            agent_ids=["agent_1", "agent_2"],
            round_indices=[1, 2],
        )

        def factory(experiment_config, llm_settings):
            self.scripted_agents = [
                ScriptedMasAgent(
                    agent_config=agent_config,
                    answer_script=answer_script[agent_config.agent_id],
                    review_script=review_script[agent_config.agent_id],
                )
                for agent_config in experiment_config.agents
            ]
            return self.scripted_agents

        runner = CompetitionRunner(
            experiment_config=config,
            llm_settings=self.llm_settings,
            agent_factory=factory,
        )

        result = runner.run_question(question, run_id="code-no-consensus")

        self.assertEqual(len(result.rounds), 2)
        self.assertFalse(result.rounds[0].reached_consensus)
        self.assertFalse(result.rounds[1].reached_consensus)

    def test_code_generation_question_uses_code_answers(self) -> None:
        question = CodeQuestionRecord(
            question_id="0",
            dataset_name="humaneval",
            prompt="def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
            entry_point="add_one",
            test="def check(candidate):\n    assert candidate(1) == 2\n",
            source_path="data/HumanEval/HumanEval.jsonl",
            source_task_id="HumanEval/0",
        )
        answer_script = {
            "agent_1": {
                1: AnswerSubmission("agent_1", [], "Body implementation.", code="    return x + 1\n"),
            },
            "agent_2": {
                1: AnswerSubmission("agent_2", [], "Full function.", code="def add_one(x):\n    return x + 1\n"),
            },
        }
        review_script = {
            "agent_1": {
                1: {
                    "agent_2": ReviewSubmission("agent_1", "agent_2", 8, "support", "Works."),
                },
            },
            "agent_2": {
                1: {
                    "agent_1": ReviewSubmission("agent_2", "agent_1", 8, "support", "Works."),
                },
            },
        }
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="agent_1"), AgentConfig(agent_id="agent_2")],
            runtime=RuntimeConfig(num_rounds=1, max_workers=2, output_dir="runs"),
        )

        def factory(experiment_config, llm_settings):
            return [
                ScriptedMasAgent(
                    agent_config=agent_config,
                    answer_script=answer_script[agent_config.agent_id],
                    review_script=review_script[agent_config.agent_id],
                )
                for agent_config in experiment_config.agents
            ]

        runner = CompetitionRunner(
            experiment_config=config,
            llm_settings=self.llm_settings,
            agent_factory=factory,
        )

        result = runner.run_question(question, run_id="code-run")
        payload = result.to_dict()

        self.assertEqual(result.task_type, "code_generation")
        self.assertEqual(payload["prompt"], question.prompt)
        self.assertEqual(payload["entry_point"], "add_one")
        self.assertNotIn("options", payload)
        self.assertEqual(
            result.rounds[0].agent_results[0].answer.code,
            "    return x + 1",
        )

    def test_run_dataset_resumes_after_complete_prefix(self) -> None:
        questions = self._questions(5)
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="agent_1"), AgentConfig(agent_id="agent_2")],
            runtime=RuntimeConfig(num_rounds=2, max_workers=2, output_dir="runs"),
        )
        all_agents: list[ScriptedMasAgent] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="resume-run",
                experiment_config=config,
                llm_settings=self.llm_settings,
            )
            for question in questions[:2]:
                store.persist_question_result(
                    run_path,
                    self._complete_question_result(
                        run_id="resume-run",
                        question=question,
                        agent_ids=["agent_1", "agent_2"],
                        num_rounds=2,
                    ),
                )
            runner = CompetitionRunner(
                experiment_config=config,
                llm_settings=self.llm_settings,
                store=store,
                agent_factory=self._resume_agent_factory(config, all_agents),
            )

            artifacts = runner.run_dataset(questions, resume_run_path=run_path)

        self.assertEqual(artifacts.run_id, "resume-run")
        self.assertEqual(len(artifacts.question_paths), 5)
        self.assertEqual(sum(len(agent.answer_calls) for agent in all_agents), 6)

    def test_run_dataset_resumes_after_consensus_short_circuit_question(self) -> None:
        questions = self._questions(2)
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="agent_1"), AgentConfig(agent_id="agent_2")],
            runtime=RuntimeConfig(num_rounds=3, max_workers=2, output_dir="runs"),
        )
        all_agents: list[ScriptedMasAgent] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="resume-consensus-run",
                experiment_config=config,
                llm_settings=self.llm_settings,
            )
            completed_result = self._complete_question_result(
                run_id="resume-consensus-run",
                question=questions[0],
                agent_ids=["agent_1", "agent_2"],
                num_rounds=1,
            )
            completed_result.rounds[0].reached_consensus = True
            store.persist_question_result(run_path, completed_result)
            runner = CompetitionRunner(
                experiment_config=config,
                llm_settings=self.llm_settings,
                store=store,
                agent_factory=self._resume_agent_factory(config, all_agents),
            )

            artifacts = runner.run_dataset(questions, resume_run_path=run_path)

        self.assertEqual(artifacts.run_id, "resume-consensus-run")
        self.assertEqual(len(artifacts.question_paths), 2)
        self.assertEqual(sum(len(agent.answer_calls) for agent in all_agents), 2)

    def test_code_generation_short_payload_with_consensus_marker_is_incomplete(self) -> None:
        question = CodeQuestionRecord(
            question_id="0",
            dataset_name="humaneval",
            prompt="def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
            entry_point="add_one",
            test="def check(candidate):\n    assert candidate(1) == 2\n",
            source_path="data/HumanEval/HumanEval.jsonl",
            source_task_id="HumanEval/0",
        )
        agent_results = []
        for agent_id, peer_id in [("agent_1", "agent_2"), ("agent_2", "agent_1")]:
            agent_results.append(
                RoundAgentResult(
                    agent_id=agent_id,
                    answer=AnswerSubmission(
                        agent_id,
                        [],
                        "Implemented.",
                        code="def add_one(x):\n    return x + 1\n",
                    ).validate(question),
                    reviews_given=[
                        ReviewSubmission(agent_id, peer_id, 8, "support", "Works.").validate()
                    ],
                    received_reviews=[
                        ReviewSubmission(peer_id, agent_id, 8, "support", "Works.").validate()
                    ],
                    total_score=8.0,
                    average_score=8.0,
                )
            )
        result = QuestionRunResult.from_task(
            run_id="code-short",
            task=question,
            rounds=[
                QuestionRoundResult(
                    round_index=1,
                    agent_results=agent_results,
                    reached_consensus=True,
                )
            ],
        )
        runner = CompetitionRunner(
            experiment_config=ExperimentConfig(),
            llm_settings=self.llm_settings,
        )

        self.assertFalse(
            runner._is_complete_question_payload(
                payload=result.to_dict(),
                question_record=question,
                agent_ids=["agent_1", "agent_2"],
                num_rounds=2,
            )
        )

    def test_run_dataset_restarts_at_question_with_missing_review(self) -> None:
        questions = self._questions(4)
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="agent_1"), AgentConfig(agent_id="agent_2")],
            runtime=RuntimeConfig(num_rounds=2, max_workers=2, output_dir="runs"),
        )
        all_agents: list[ScriptedMasAgent] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="resume-run",
                experiment_config=config,
                llm_settings=self.llm_settings,
            )
            for question in questions[:2]:
                store.persist_question_result(
                    run_path,
                    self._complete_question_result(
                        run_id="resume-run",
                        question=question,
                        agent_ids=["agent_1", "agent_2"],
                        num_rounds=2,
                    ),
                )
            q2_path = store.question_result_path(run_path, questions[1].question_key)
            payload = json.loads(q2_path.read_text(encoding="utf-8"))
            payload["rounds"][0]["agent_results"][0]["reviews_given"] = []
            q2_path.write_text(json.dumps(payload), encoding="utf-8")
            runner = CompetitionRunner(
                experiment_config=config,
                llm_settings=self.llm_settings,
                store=store,
                agent_factory=self._resume_agent_factory(config, all_agents),
            )

            runner.run_dataset(questions, resume_run_path=run_path)

        self.assertEqual(sum(len(agent.answer_calls) for agent in all_agents), 6)

    def test_run_dataset_restarts_at_corrupt_question_json(self) -> None:
        questions = self._questions(3)
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="agent_1"), AgentConfig(agent_id="agent_2")],
            runtime=RuntimeConfig(num_rounds=2, max_workers=2, output_dir="runs"),
        )
        all_agents: list[ScriptedMasAgent] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="resume-run",
                experiment_config=config,
                llm_settings=self.llm_settings,
            )
            store.persist_question_result(
                run_path,
                self._complete_question_result(
                    run_id="resume-run",
                    question=questions[0],
                    agent_ids=["agent_1", "agent_2"],
                    num_rounds=2,
                ),
            )
            corrupt_path = store.question_result_path(run_path, questions[1].question_key)
            corrupt_path.write_text("{not-json", encoding="utf-8")
            runner = CompetitionRunner(
                experiment_config=config,
                llm_settings=self.llm_settings,
                store=store,
                agent_factory=self._resume_agent_factory(config, all_agents),
            )

            runner.run_dataset(questions, resume_run_path=run_path)

        self.assertEqual(sum(len(agent.answer_calls) for agent in all_agents), 4)

    def test_run_dataset_records_failed_question_and_continues(self) -> None:
        questions = self._questions(3)
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="agent_1"), AgentConfig(agent_id="agent_2")],
            runtime=RuntimeConfig(num_rounds=1, max_workers=2, output_dir="runs"),
        )

        def factory(experiment_config, llm_settings):
            return [
                FailingMasAgent(
                    agent_config=agent_config,
                    fail_question_keys={"demo__q2__single_choice"},
                )
                for agent_config in experiment_config.agents
            ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            runner = CompetitionRunner(
                experiment_config=config,
                llm_settings=self.llm_settings,
                store=store,
                agent_factory=factory,
            )

            artifacts = runner.run_dataset(questions)
            failed_questions = store.read_failed_questions(artifacts.run_path)

            self.assertEqual(len(artifacts.question_paths), 2)
            self.assertIsNotNone(failed_questions)
            self.assertEqual(len(failed_questions), 1)
            self.assertEqual(failed_questions[0]["question_index"], 2)
            self.assertEqual(failed_questions[0]["question_key"], "demo__q2__single_choice")
            persisted = {
                Path(path).stem
                for path in artifacts.question_paths
            }
            self.assertEqual(
                persisted,
                {"demo__q1__single_choice", "demo__q3__single_choice"},
            )

    def test_resume_run_removes_completed_question_from_failed_questions_file(self) -> None:
        questions = self._questions(1)
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="agent_1"), AgentConfig(agent_id="agent_2")],
            runtime=RuntimeConfig(num_rounds=1, max_workers=2, output_dir="runs"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonRunStore(tmp_dir)
            run_path = store.initialize_run(
                run_id="competition-old",
                experiment_config=config,
                llm_settings=self.llm_settings,
            )
            store.persist_failed_questions(
                run_path,
                [
                    {
                        "question_index": 26,
                        "question_id": "q1",
                        "question_key": "demo__q1__single_choice",
                        "error_type": "ValidationError",
                        "error_message": "old error",
                    }
                ],
            )
            runner = CompetitionRunner(
                experiment_config=config,
                llm_settings=self.llm_settings,
                store=store,
                agent_factory=lambda experiment_config, llm_settings: [
                    ScriptedMasAgent(
                        agent_config=agent_config,
                        answer_script={
                            1: AnswerSubmission(
                                agent_id=agent_config.agent_id,
                                selected_option_ids=["A"],
                                reasoning="Option A is correct.",
                            )
                        },
                        review_script={
                            1: {
                                peer_id: ReviewSubmission(
                                    reviewer_agent_id=agent_config.agent_id,
                                    target_agent_id=peer_id,
                                    score=8,
                                    stance="support",
                                    main_reason="Looks complete.",
                                )
                                for peer_id in [
                                    peer.agent_id
                                    for peer in experiment_config.agents
                                    if peer.agent_id != agent_config.agent_id
                                ]
                            }
                        },
                    )
                    for agent_config in experiment_config.agents
                ],
            )

            artifacts = runner.run_dataset(questions, resume_run_path=run_path)
            failed_questions = store.read_failed_questions(artifacts.run_path)

            self.assertEqual(len(artifacts.question_paths), 1)
            self.assertEqual(failed_questions, [])

    def test_collect_reviews_cancels_executor_without_wait_on_keyboard_interrupt(self) -> None:
        answer_script = {
            "agent_1": {1: AnswerSubmission("agent_1", ["A"], "Answer 1")},
            "agent_2": {1: AnswerSubmission("agent_2", ["A"], "Answer 2")},
        }
        review_script = {
            "agent_1": {
                1: {
                    "agent_2": ReviewSubmission("agent_1", "agent_2", 8, "support", "Looks good."),
                }
            },
            "agent_2": {
                1: {
                    "agent_1": ReviewSubmission("agent_2", "agent_1", 8, "support", "Looks good."),
                }
            },
        }

        def factory(experiment_config, llm_settings):
            return [
                ScriptedMasAgent(
                    agent_config=agent_config,
                    answer_script=answer_script[agent_config.agent_id],
                    review_script=review_script[agent_config.agent_id],
                )
                for agent_config in experiment_config.agents[:2]
            ]

        runner = CompetitionRunner(
            experiment_config=ExperimentConfig(
                agents=[
                    AgentConfig(agent_id="agent_1"),
                    AgentConfig(agent_id="agent_2"),
                ],
                runtime=RuntimeConfig(num_rounds=1, max_workers=2, output_dir="runs"),
            ),
            llm_settings=self.llm_settings,
            agent_factory=factory,
        )
        agents = runner._build_agents()
        answers = {
            agent.agent_id: answer_script[agent.agent_id][1].validate(self.question)
            for agent in agents
        }

        class InterruptingFuture:
            def result(self):
                raise KeyboardInterrupt

            def cancel(self):
                return True

        class RecordingExecutor:
            instances: list["RecordingExecutor"] = []

            def __init__(self, max_workers):
                self.max_workers = max_workers
                self.shutdown_calls: list[tuple[bool, bool]] = []
                RecordingExecutor.instances.append(self)

            def submit(self, fn, item):
                return InterruptingFuture()

            def shutdown(self, wait, cancel_futures=False):
                self.shutdown_calls.append((wait, cancel_futures))

        with patch("autogen_mas.runtime.runner.ThreadPoolExecutor", RecordingExecutor):
            with self.assertRaises(KeyboardInterrupt):
                runner._collect_reviews_and_scores(
                    agents=agents,
                    question_record=self.question,
                    answers=answers,
                    round_index=1,
                    max_workers=2,
                )

        self.assertEqual(len(RecordingExecutor.instances), 1)
        self.assertEqual(
            RecordingExecutor.instances[0].shutdown_calls,
            [(False, True)],
        )


if __name__ == "__main__":
    unittest.main()
