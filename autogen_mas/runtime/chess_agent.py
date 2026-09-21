from __future__ import annotations

from autogen_mas.chess_parsing import canonicalize_chess_final_answer
from autogen_mas.config import AgentConfig, ExperimentConfig
from autogen_mas.models import (
    AnswerSubmission,
    ChessQuestionRecord,
    PriorRoundFeedback,
    ReviewSubmission,
    ValidationError,
)

from .agent import MasAgent
from .chess_prompting import (
    build_chess_answer_user_prompt,
    build_chess_review_user_prompt,
)
from .clients import StructuredLLMClient


DEFAULT_CHESS_ANSWER_PROMPT = """You are an expert chess state-tracking competitor in a clean multi-agent reasoning experiment.
Return only valid JSON.
Reason carefully from the provided move history and the current source square.
Your final_answer must be exactly one destination square in the form [a-h][1-8].
Keep the reasoning concise, concrete, and aligned with the final square.
"""

DEFAULT_CHESS_REVIEW_PROMPT = """You are an expert chess state-tracking judge in a clean multi-agent reasoning experiment.
Return only valid JSON.
Evaluate whether the peer's proposed destination square is legal for the specified piece in the given position.
Score the answer from 1 to 10, choose a stance, and give one concise main reason.
"""


def _debug_chain_of_thought(
    payload: dict,
    *,
    debug_mode: bool,
    context: str,
) -> str | None:
    if not debug_mode:
        return None
    chain_of_thought = str(payload.get("chain_of_thought", "")).strip()
    if not chain_of_thought:
        raise ValidationError(f"{context} must include chain_of_thought in debug mode.")
    return chain_of_thought


def _debug_confidence(
    payload: dict,
    *,
    debug_mode: bool,
    context: str,
) -> float | None:
    if not debug_mode:
        return None
    raw_confidence = payload.get("confidence")
    if raw_confidence is None:
        raise ValidationError(f"{context} must include confidence in debug mode.")
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"{context} confidence must be numeric in debug mode."
        ) from error
    if not 0.0 <= confidence <= 1.0:
        raise ValidationError(
            f"{context} confidence must be between 0.0 and 1.0 in debug mode."
        )
    return confidence


class ChessMasAgent(MasAgent):
    def __init__(
        self,
        agent_config: AgentConfig,
        experiment_config: ExperimentConfig,
        client: StructuredLLMClient,
        default_model: str,
        debug_mode: bool = False,
    ) -> None:
        super().__init__(agent_config)
        self.experiment_config = experiment_config
        self.client = client
        self.default_model = default_model
        self.debug_mode = debug_mode

    def answer(
        self,
        question_context: ChessQuestionRecord,
        prior_feedback: PriorRoundFeedback | None,
        round_index: int,
    ) -> AnswerSubmission:
        response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_answer_prompt(
                DEFAULT_CHESS_ANSWER_PROMPT
            ),
            user_prompt=build_chess_answer_user_prompt(
                question_record=question_context,
                round_index=round_index,
                prior_feedback=prior_feedback,
                debug_mode=self.debug_mode,
            ),
            model=self.agent_config.model or self.default_model,
            temperature=self.agent_config.temperature,
        )
        return AnswerSubmission(
            agent_id=self.agent_id,
            selected_option_ids=[],
            reasoning=str(response.get("reasoning", "")),
            final_answer=canonicalize_chess_final_answer(
                str(response.get("final_answer", "")),
                source_square=question_context.source_square,
            ),
            confidence=_debug_confidence(
                response,
                debug_mode=self.debug_mode,
                context=f"Agent {self.agent_id} answer",
            ),
            changed_answer=bool(response.get("changed_answer", False)),
            change_drivers=list(response.get("change_drivers", [])),
            change_summary=str(response.get("change_summary", "")),
            chain_of_thought=_debug_chain_of_thought(
                response,
                debug_mode=self.debug_mode,
                context=f"Agent {self.agent_id} answer",
            ),
        ).validate(question_context)

    def review(
        self,
        question_context: ChessQuestionRecord,
        peer_submission: AnswerSubmission,
        reviewer_answer: AnswerSubmission,
        round_index: int,
    ) -> ReviewSubmission:
        response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_review_prompt(
                DEFAULT_CHESS_REVIEW_PROMPT
            ),
            user_prompt=build_chess_review_user_prompt(
                question_record=question_context,
                round_index=round_index,
                reviewer_answer=reviewer_answer,
                peer_submission=peer_submission,
                debug_mode=self.debug_mode,
            ),
            model=self.agent_config.model or self.default_model,
            temperature=self.agent_config.temperature,
        )
        return ReviewSubmission(
            reviewer_agent_id=self.agent_id,
            target_agent_id=peer_submission.agent_id,
            score=int(response.get("score", 0)),
            stance=str(response.get("stance", "")),
            main_reason=str(response.get("main_reason", "")),
            chain_of_thought=_debug_chain_of_thought(
                response,
                debug_mode=self.debug_mode,
                context=f"Agent {self.agent_id} review",
            ),
        ).validate()
