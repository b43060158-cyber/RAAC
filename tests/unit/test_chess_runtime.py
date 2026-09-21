from __future__ import annotations

import json
import tempfile
import unittest

from autogen_mas.config import AgentConfig, DashScopeSettings, ExperimentConfig, RuntimeConfig
from autogen_mas.models import ChessQuestionRecord
from autogen_mas.persistence import JsonRunStore
from autogen_mas.runtime.chess_agent import ChessMasAgent
from autogen_mas.runtime.chess_runner import ChessCompetitionRunner


class FakeChessStructuredClient:
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
        if stage == "chess_answer":
            return {
                "final_answer": "The piece on d2 can move to e3.",
                "reasoning": "From d2 the piece can legally move to e3, so the move is valid.",
                "changed_answer": False,
                "change_drivers": [],
                "change_summary": "",
            }
        if stage == "chess_review":
            return {
                "score": 8,
                "stance": "support",
                "main_reason": "The proposed destination square is legal in the given position.",
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class ChessRuntimeTest(unittest.TestCase):
    def test_chess_agent_parses_square_from_explanation(self) -> None:
        question = ChessQuestionRecord(
            question_id="Chess_0",
            dataset_name="chess",
            game="a2a4 a7a5",
            source_square="d2",
            legal_target_squares=["e3", "c3", "d1"],
            rendered_question="Chess question.",
            metadata={"source_square": "d2"},
        )
        agent = ChessMasAgent(
            agent_config=AgentConfig(agent_id="agent_1"),
            experiment_config=ExperimentConfig(),
            client=FakeChessStructuredClient(),
            default_model="fake-model",
        )

        answer = agent.answer(question, None, 1)

        self.assertEqual(answer.final_answer, "e3")

    def test_chess_runner_reaches_consensus_on_equivalent_square_forms(self) -> None:
        question = ChessQuestionRecord(
            question_id="Chess_0",
            dataset_name="chess",
            game="a2a4 a7a5",
            source_square="d2",
            legal_target_squares=["e3", "c3", "d1"],
            rendered_question="Chess question.",
            metadata={"source_square": "d2"},
        )
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="agent_1"), AgentConfig(agent_id="agent_2")],
            runtime=RuntimeConfig(num_rounds=2, max_workers=2, output_dir="runs"),
        )
        client_a = FakeChessStructuredClient()
        client_b = FakeChessStructuredClient()

        def factory(experiment_config, llm_settings):
            agent_1 = ChessMasAgent(
                agent_config=experiment_config.agents[0],
                experiment_config=experiment_config,
                client=client_a,
                default_model="fake-model",
            )
            agent_2 = ChessMasAgent(
                agent_config=experiment_config.agents[1],
                experiment_config=experiment_config,
                client=client_b,
                default_model="fake-model",
            )
            return [agent_1, agent_2]

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = ChessCompetitionRunner(
                experiment_config=config,
                llm_settings=DashScopeSettings(api_key="test-key", model="fake-model"),
                store=JsonRunStore(tmp_dir),
                agent_factory=factory,
            )
            result = runner.run_question(question)

        self.assertTrue(result.rounds[0].reached_consensus)
        self.assertEqual(len(result.rounds), 1)


if __name__ == "__main__":
    unittest.main()
