from __future__ import annotations

from autogen_mas.chess_parsing import (
    canonicalize_chess_final_answer,
    chess_answer_tie_key,
)
from autogen_mas.models import EvaluationQuestionResult


TOP_AGENT_SELECTION_RULE = (
    "last_round.total_score desc -> all_rounds.average_score mean desc -> "
    "agent_id asc; unresolved differing-answer ties excluded"
)


class ChessEvaluator:
    @staticmethod
    def supports_payload(payload: dict) -> bool:
        return str(payload.get("task_type", "")) == "chess_move"

    @staticmethod
    def tie_key(agent_result: dict, *, payload: dict) -> tuple[str, str]:
        answer = agent_result.get("answer", {})
        final_answer = ""
        if isinstance(answer, dict):
            final_answer = str(answer.get("final_answer", ""))
        return chess_answer_tie_key(
            final_answer,
            source_square=ChessEvaluator._source_square(payload),
        )

    @staticmethod
    def evaluate_question(
        *,
        payload: dict,
        selected: dict,
        tie_candidate_ids: list[str],
        selected_is_adversarial: bool,
        is_tie: bool,
    ) -> EvaluationQuestionResult:
        predicted_final_answer = canonicalize_chess_final_answer(
            str(selected.get("answer", {}).get("final_answer", "")),
            source_square=ChessEvaluator._source_square(payload),
        )
        legal_target_squares = [
            str(item).strip().lower()
            for item in payload.get("legal_target_squares", [])
            if str(item).strip()
        ]
        is_correct = predicted_final_answer in legal_target_squares
        return EvaluationQuestionResult(
            question_id=str(payload["question_id"]),
            question_key=str(payload["question_key"]),
            dataset_name=str(payload["dataset_name"]),
            task_type=str(payload["task_type"]),
            selected_agent_id=str(selected["agent_id"]),
            predicted_option_ids=[],
            correct_option_ids=[],
            is_correct=is_correct,
            selection_rule=TOP_AGENT_SELECTION_RULE,
            predicted_final_answer=predicted_final_answer,
            acceptable_answers=legal_target_squares,
            selected_is_adversarial=selected_is_adversarial,
            is_tie=is_tie,
            excluded_from_accuracy=is_tie,
            tie_candidate_ids=tie_candidate_ids,
        )

    @staticmethod
    def _source_square(payload: dict) -> str | None:
        source_square = payload.get("source_square")
        if isinstance(source_square, str) and source_square.strip():
            return source_square.strip().lower()
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("source_square")
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return None
