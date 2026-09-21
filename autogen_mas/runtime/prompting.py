from __future__ import annotations

import json

from autogen_mas.models import (
    AnswerSubmission,
    CodeQuestionRecord,
    PriorRoundFeedback,
    QuestionRecord,
    ShortAnswerQuestionRecord,
    TaskRecord,
)


def _fair_eval_candidate_responses(question_record: QuestionRecord) -> dict[str, object] | None:
    if question_record.dataset_name != "faireval":
        return None
    response_1_text = question_record.metadata.get("response_1_text")
    response_2_text = question_record.metadata.get("response_2_text")
    if not isinstance(response_1_text, str) or not isinstance(response_2_text, str):
        return None
    return {
        "response_1": response_1_text,
        "response_2": response_2_text,
    }


def _question_payload(question_record: TaskRecord) -> dict[str, object]:
    if isinstance(question_record, CodeQuestionRecord):
        return {
            "question_id": question_record.question_id,
            "question_key": question_record.question_key,
            "dataset_name": question_record.dataset_name,
            "task_type": question_record.task_type,
            "prompt": question_record.prompt,
            "entry_point": question_record.entry_point,
        }
    if isinstance(question_record, ShortAnswerQuestionRecord):
        return {
            "question_id": question_record.question_id,
            "question_key": question_record.question_key,
            "dataset_name": question_record.dataset_name,
            "task_type": question_record.task_type,
            "question": question_record.question,
        }
    payload = {
        "question_id": question_record.question_id,
        "question_key": question_record.question_key,
        "dataset_name": question_record.dataset_name,
        "task_type": question_record.task_type,
        "question": question_record.question,
        "options": [
            {"option_id": option.option_id, "text": option.text}
            for option in question_record.options
        ],
    }
    candidate_responses = _fair_eval_candidate_responses(question_record)
    if candidate_responses is not None:
        payload["candidate_responses"] = candidate_responses
    return payload


def build_answer_user_prompt(
    *,
    question_record: TaskRecord,
    round_index: int,
    prior_feedback: PriorRoundFeedback | None,
    mitigation_message: str | None = None,
    debug_mode: bool = False,
) -> str:
    if isinstance(question_record, CodeQuestionRecord):
        response_schema: dict[str, object] = {
            "code": "Python implementation for the requested function",
            "reasoning": "brief but concrete explanation",
            "changed_answer": "boolean",
            "change_drivers": ["list of reviewer agent ids that influenced your update"],
            "change_summary": "brief explanation of what changed and why",
        }
        constraints = [
            "Return valid JSON only.",
            "code must be non-empty Python code.",
            "Implement the requested entry_point function.",
            "Do not include Markdown fenced code blocks.",
            "Do not include tests or calls to check().",
            "For round 1, changed_answer should be false and change_drivers should be empty.",
            "List only peer agent ids in change_drivers.",
            "If changed_answer is true, change_summary must explain the update.",
        ]
    elif isinstance(question_record, ShortAnswerQuestionRecord):
        response_schema = {
            "final_answer": "short final answer",
            "reasoning": "brief but concrete explanation for the current answer",
            "changed_answer": "boolean",
            "change_drivers": ["list of reviewer agent ids that influenced your update"],
            "change_summary": "brief explanation of what changed and why",
        }
        constraints = [
            "Return valid JSON only.",
            "final_answer must be non-empty.",
            "reasoning must explain the current final_answer.",
            "If reasoning states an explicit final answer, final_answer must match it exactly.",
            "Do not return selected_option_ids.",
            "For round 1, changed_answer should be false and change_drivers should be empty.",
            "List only peer agent ids in change_drivers.",
            "If changed_answer is true, change_summary must explain the update.",
        ]
    else:
        response_schema = {
            "selected_option_ids": ["exactly one option id, for example A"],
            "reasoning": "brief but concrete explanation",
            "changed_answer": "boolean",
            "change_drivers": ["list of reviewer agent ids that influenced your update"],
            "change_summary": "brief explanation of what changed and why",
        }
        constraints = [
            "Return valid JSON only.",
            "selected_option_ids must use only the provided option ids.",
            "Return exactly one option id in selected_option_ids.",
            "Never return an empty selected_option_ids list or more than one option id for single_choice.",
            "The selected_option_ids JSON shape must be like [\"A\"], not \"A\" and not [\"A\", \"B\"].",
            "For single_choice EXCEPT/all-of-the-following style questions, still return exactly one final option id rather than listing every true statement.",
            "reasoning must justify the current selected_option_ids.",
            "Do not claim that any unselected option is correct, preferable, or the final answer.",
            "If reasoning explicitly names the correct answer or chosen option ids, they must match selected_option_ids exactly.",
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
                (
                    "confidence must represent your subjective confidence that your "
                    "current answer is correct."
                ),
                "Set confidence based only on your own certainty about the answer.",
                (
                    "Do not adjust confidence based on agreement or disagreement from "
                    "other agents."
                ),
                "Do not default to a high confidence value just to sound decisive.",
                (
                    "Use 0.95 or higher only if you judge your answer to be correct "
                    "with very little internal uncertainty."
                ),
                (
                    "Lower confidence when you notice ambiguity, missing evidence, "
                    "uncertain recall, or unresolved doubts in your own reasoning."
                ),
            ]
        )
        constraints.append(
            "chain_of_thought must be non-empty and contain the full reasoning trace."
        )
    payload: dict[str, object] = {
        "stage": "answer",
        "round_index": round_index,
        "question_context": _question_payload(question_record),
        "response_schema": response_schema,
        "constraints": constraints,
    }
    if mitigation_message:
        cleaned_message = mitigation_message.strip()
        if cleaned_message:
            payload["mitigation"] = {"enabled": True, "message": cleaned_message}
            constraints.append(cleaned_message)
    if prior_feedback is None:
        payload["prior_feedback"] = None
    else:
        payload["prior_feedback"] = prior_feedback.to_dict()
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_single_choice_answer_repair_user_prompt(
    *,
    question_record: QuestionRecord,
    round_index: int,
    raw_response: dict,
) -> str:
    payload = {
        "stage": "answer_repair",
        "round_index": round_index,
        "question_context": _question_payload(question_record),
        "invalid_response": raw_response,
        "response_schema": {
            "selected_option_ids": ["exactly one option id, for example A"],
            "reasoning": "brief but concrete explanation for the repaired single choice",
        },
        "constraints": [
            "Return valid JSON only.",
            "Choose exactly one option id from the provided options.",
            "Return selected_option_ids in the JSON shape [\"A\"], not \"A\" and not [\"A\", \"B\"].",
            "Do not return an empty selected_option_ids list.",
            "The reasoning must defend the single selected_option_ids value.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_review_user_prompt(
    *,
    question_record: TaskRecord,
    round_index: int,
    reviewer_answer: AnswerSubmission,
    peer_submission: AnswerSubmission,
    debug_mode: bool = False,
) -> str:
    response_schema = {
        "score": "integer 1-10",
        "stance": "support | oppose | mixed",
        "main_reason": "one concise reason for the score and stance",
    }
    constraints = [
        "Return valid JSON only.",
        "Score must be an integer from 1 to 10.",
        "stance must be support, oppose, or mixed.",
        "main_reason must be non-empty.",
    ]
    if debug_mode:
        response_schema["chain_of_thought"] = (
            "detailed step-by-step evaluation trace for debug logging"
        )
        constraints.append(
            "chain_of_thought must be non-empty and contain the full evaluation trace."
        )
    payload = {
        "stage": "review",
        "round_index": round_index,
        "question_context": _question_payload(question_record),
        "reviewer_answer": reviewer_answer.to_dict(include_chain_of_thought=False),
        "peer_submission": peer_submission.to_dict(include_chain_of_thought=False),
        "response_schema": response_schema,
        "constraints": constraints,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_batch_answer_user_prompt(
    *,
    question_records: list[TaskRecord],
    round_index: int,
    prior_feedback_by_question: dict[str, PriorRoundFeedback | None],
    mitigation_message: str | None = None,
    debug_mode: bool = False,
) -> str:
    answer_schema: dict[str, object] = {
        "question_key": "question_key from question_contexts",
        "selected_option_ids": ["exactly one option id for choice tasks only, for example A"],
        "final_answer": "short final answer for math_short_answer tasks only",
        "code": "Python implementation for code_generation tasks only",
        "reasoning": "brief but concrete explanation",
        "changed_answer": "boolean",
        "change_drivers": ["list of reviewer agent ids that influenced your update"],
        "change_summary": "brief explanation of what changed and why",
    }
    constraints = [
        "Return valid JSON only.",
        "Return exactly one answer for each question_context.",
        "question_key must match one of the provided question_context question_key values.",
        "For choice tasks, selected_option_ids must use only the provided option ids for that question.",
        "For choice tasks, return exactly one option id in selected_option_ids.",
        "For choice tasks, never return an empty selected_option_ids list or more than one option id.",
        "For choice tasks, the selected_option_ids JSON shape must be like [\"A\"], not \"A\" and not [\"A\", \"B\"].",
        "For single_choice EXCEPT/all-of-the-following style questions, still return exactly one final option id rather than listing every true statement.",
        "For choice tasks, reasoning must justify the current selected_option_ids.",
        "For choice tasks, do not claim that any unselected option is correct, preferable, or the final answer.",
        "For choice tasks, if reasoning explicitly names the correct answer or chosen option ids, they must match selected_option_ids exactly.",
        "If task_type is math_short_answer, return non-empty final_answer and no selected_option_ids.",
        "For math_short_answer, if reasoning states an explicit final answer, final_answer must match it exactly.",
        "If task_type is code_generation, return non-empty code and no selected_option_ids.",
        "For code_generation, do not include Markdown fenced code blocks, tests, or calls to check().",
        "For round 1, changed_answer should be false and change_drivers should be empty.",
        "List only peer agent ids in change_drivers.",
        "If changed_answer is true, change_summary must explain the update.",
    ]
    if debug_mode:
        answer_schema["confidence"] = (
            "float 0.0-1.0 for your subjective confidence that your current answer is correct"
        )
        constraints.extend(
            [
                "Each answer confidence must be a number between 0.0 and 1.0.",
                (
                    "Each answer confidence must represent your subjective confidence "
                    "that your current answer is correct."
                ),
                "Set each confidence based only on your own certainty about that answer.",
                (
                    "Do not adjust confidence based on agreement or disagreement from "
                    "other agents."
                ),
                "Do not default to high confidence values just to sound decisive.",
                (
                    "Use 0.95 or higher only when you judge that answer to be correct "
                    "with very little internal uncertainty."
                ),
                (
                    "Lower confidence when you notice ambiguity, missing evidence, "
                    "uncertain recall, or unresolved doubts in your own reasoning."
                ),
            ]
        )
    payload: dict[str, object] = {
        "stage": "batch_answer",
        "round_index": round_index,
        "question_contexts": [_question_payload(question) for question in question_records],
        "prior_feedback_by_question": {
            question.question_key: (
                None
                if prior_feedback_by_question.get(question.question_key) is None
                else prior_feedback_by_question[question.question_key].to_dict()
            )
            for question in question_records
        },
        "response_schema": {
            "answers": [answer_schema]
        },
        "constraints": constraints,
    }
    if mitigation_message:
        cleaned_message = mitigation_message.strip()
        if cleaned_message:
            payload["mitigation"] = {"enabled": True, "message": cleaned_message}
            constraints.append(cleaned_message)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_batch_review_user_prompt(
    *,
    question_record: TaskRecord,
    round_index: int,
    reviewer_answer: AnswerSubmission,
    peer_submissions: list[AnswerSubmission],
    debug_mode: bool = False,
) -> str:
    review_schema = {
        "target_agent_id": "peer agent id being reviewed",
        "score": "integer 1-10",
        "stance": "support | oppose | mixed",
        "main_reason": "one concise reason for the score and stance",
    }
    constraints = [
        "Return valid JSON only.",
        "Return exactly one review for each peer_submission.",
        "target_agent_id must match one of the provided peer_submission agent_id values.",
        "Score must be an integer from 1 to 10.",
        "stance must be support, oppose, or mixed.",
        "main_reason must be non-empty.",
    ]
    if debug_mode:
        review_schema["chain_of_thought"] = (
            "detailed step-by-step evaluation trace for debug logging"
        )
        constraints.append(
            "Each review chain_of_thought must be non-empty and contain the full evaluation trace."
        )
    payload = {
        "stage": "batch_review",
        "round_index": round_index,
        "question_context": _question_payload(question_record),
        "reviewer_answer": reviewer_answer.to_dict(include_chain_of_thought=False),
        "peer_submissions": [
            submission.to_dict(include_chain_of_thought=False)
            for submission in peer_submissions
        ],
        "response_schema": {
            "reviews": [review_schema]
        },
        "constraints": constraints,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_batch_questions_review_user_prompt(
    *,
    question_records: list[TaskRecord],
    round_index: int,
    reviewer_answers_by_question: dict[str, AnswerSubmission],
    peer_submissions_by_question: dict[str, list[AnswerSubmission]],
) -> str:
    payload = {
        "stage": "batch_questions_review",
        "round_index": round_index,
        "question_contexts": [_question_payload(question) for question in question_records],
        "reviewer_answers_by_question": {
            question.question_key: reviewer_answers_by_question[
                question.question_key
            ].to_dict(include_chain_of_thought=False)
            for question in question_records
        },
        "peer_submissions_by_question": {
            question.question_key: [
                submission.to_dict(include_chain_of_thought=False)
                for submission in peer_submissions_by_question[question.question_key]
            ]
            for question in question_records
        },
        "response_schema": {
            "reviews": [
                {
                    "question_key": "question_key from question_contexts",
                    "target_agent_id": "peer agent id being reviewed",
                    "score": "integer 1-10",
                    "stance": "support | oppose | mixed",
                    "main_reason": "one concise reason for the score and stance",
                }
            ]
        },
        "constraints": [
            "Return valid JSON only.",
            "Return exactly one review for each question_key and peer submission.",
            "question_key must match one of the provided question_context question_key values.",
            "target_agent_id must match one peer submission agent_id for that question_key.",
            "Score must be an integer from 1 to 10.",
            "stance must be support, oppose, or mixed.",
            "main_reason must be non-empty.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
