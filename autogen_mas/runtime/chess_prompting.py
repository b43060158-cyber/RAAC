from __future__ import annotations

import json

from autogen_mas.models import AnswerSubmission, ChessQuestionRecord, PriorRoundFeedback


def _question_payload(question_record: ChessQuestionRecord) -> dict[str, object]:
    return {
        "question_id": question_record.question_id,
        "question_key": question_record.question_key,
        "dataset_name": question_record.dataset_name,
        "task_type": question_record.task_type,
        "game": question_record.game,
        "source_square": question_record.source_square,
        "legal_target_squares": list(question_record.legal_target_squares),
        "rendered_question": question_record.rendered_question,
    }


def build_chess_answer_user_prompt(
    *,
    question_record: ChessQuestionRecord,
    round_index: int,
    prior_feedback: PriorRoundFeedback | None,
    debug_mode: bool = False,
) -> str:
    response_schema: dict[str, object] = {
        "final_answer": "single destination square matching [a-h][1-8]",
        "reasoning": "brief but concrete explanation for why the move is legal",
        "changed_answer": "boolean",
        "change_drivers": ["list of reviewer agent ids that influenced your update"],
        "change_summary": "brief explanation of what changed and why",
    }
    constraints = [
        "Return valid JSON only.",
        "final_answer must be non-empty and be exactly one square in the form [a-h][1-8].",
        "Do not return selected_option_ids.",
        "Do not use natural-language phrases instead of the final square.",
        "reasoning must explain why the final_answer is a legal move in the given position.",
        "If reasoning explicitly names a final square, it must match final_answer exactly.",
        "For round 1, changed_answer should be false and change_drivers should be empty.",
        "List only peer agent ids in change_drivers.",
        "If changed_answer is true, change_summary must explain the update.",
    ]
    if debug_mode:
        response_schema["confidence"] = (
            "float 0.0-1.0 for your subjective confidence that your current answer is correct"
        )
        response_schema["chain_of_thought"] = (
            "detailed step-by-step reasoning trace for debug logging"
        )
        constraints.extend(
            [
                "confidence must be a number between 0.0 and 1.0.",
                "confidence must represent your subjective confidence that final_answer is correct.",
                "chain_of_thought must be non-empty and contain the full reasoning trace.",
            ]
        )
    payload: dict[str, object] = {
        "stage": "chess_answer",
        "round_index": round_index,
        "question_context": _question_payload(question_record),
        "response_schema": response_schema,
        "constraints": constraints,
        "prior_feedback": None if prior_feedback is None else prior_feedback.to_dict(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_chess_review_user_prompt(
    *,
    question_record: ChessQuestionRecord,
    round_index: int,
    reviewer_answer: AnswerSubmission,
    peer_submission: AnswerSubmission,
    debug_mode: bool = False,
) -> str:
    response_schema: dict[str, object] = {
        "score": "integer 1-10",
        "stance": "support | oppose | mixed",
        "main_reason": "one concise reason focused on move legality and reasoning quality",
    }
    constraints = [
        "Return valid JSON only.",
        "Score must be an integer from 1 to 10.",
        "stance must be support, oppose, or mixed.",
        "main_reason must be non-empty.",
        "Judge whether the peer's final_answer is a legal destination square for the specified piece in the given position.",
        "Use the peer's reasoning quality and move legality as the basis for the score.",
    ]
    if debug_mode:
        response_schema["chain_of_thought"] = (
            "detailed step-by-step evaluation trace for debug logging"
        )
        constraints.append(
            "chain_of_thought must be non-empty and contain the full evaluation trace."
        )
    payload = {
        "stage": "chess_review",
        "round_index": round_index,
        "question_context": _question_payload(question_record),
        "reviewer_answer": reviewer_answer.to_dict(include_chain_of_thought=False),
        "peer_submission": peer_submission.to_dict(include_chain_of_thought=False),
        "response_schema": response_schema,
        "constraints": constraints,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
