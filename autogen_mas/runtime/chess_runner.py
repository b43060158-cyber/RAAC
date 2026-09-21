from __future__ import annotations

from autogen_mas.chess_parsing import chess_answer_tie_key
from autogen_mas.config import DashScopeSettings, ExperimentConfig
from autogen_mas.models import AnswerSubmission, ChessQuestionRecord

from .chess_agent import ChessMasAgent
from .clients import build_structured_client
from .runner import CompetitionRunner


class ChessCompetitionRunner(CompetitionRunner):
    def _build_agents(self, experiment_config: ExperimentConfig | None = None):
        config = experiment_config or self.experiment_config
        if self.agent_factory is not None:
            return self.agent_factory(config, self.llm_settings)
        client = build_structured_client(self.llm_settings)
        return [
            ChessMasAgent(
                agent_config=agent_config,
                experiment_config=config,
                client=client,
                default_model=self.llm_settings.model,
                debug_mode=self.debug_mode,
            )
            for agent_config in config.agents
        ]

    def _supports_consensus_short_circuit(self, question_record) -> bool:
        if isinstance(question_record, ChessQuestionRecord):
            return True
        return super()._supports_consensus_short_circuit(question_record)

    def _consensus_answer_key(self, question_record, answer: object) -> tuple:
        if isinstance(question_record, ChessQuestionRecord):
            return (
                "final_answer",
                chess_answer_tie_key(
                    str(getattr(answer, "final_answer", "")),
                    source_square=question_record.source_square,
                ),
            )
        return super()._consensus_answer_key(question_record, answer)
