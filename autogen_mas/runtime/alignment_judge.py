from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from autogen_mas.answer_consistency import (
    answer_reasoning_consistency_check,
    extract_reasoning_decision_option_ids,
)
from autogen_mas.config import AlignmentJudgeConfig, AgentConfig
from autogen_mas.models import QuestionRecord, ValidationError

from .clients import StructuredLLMClient


@dataclass(slots=True)
class AlignmentResolution:
    selected_option_ids: list[str]
    raw_selected_option_ids: list[str]
    resolution: str
    judge_votes: list[str]


def _build_alignment_judge_user_prompt(
    *,
    question_record: QuestionRecord,
    structured_option_ids: list[str],
    reasoning_option_ids: list[str],
    reasoning: str,
) -> str:
    payload = {
        "stage": "alignment_judge",
        "question_context": {
            "question_id": question_record.question_id,
            "question_key": question_record.question_key,
            "dataset_name": question_record.dataset_name,
            "task_type": question_record.task_type,
            "question": question_record.question,
            "options": [
                {"option_id": option.option_id, "text": option.text}
                for option in question_record.options
            ],
            **(
                {
                    "candidate_responses": {
                        "response_1": question_record.metadata.get("response_1_text"),
                        "response_2": question_record.metadata.get("response_2_text"),
                    }
                }
                if question_record.dataset_name == "faireval"
                else {}
            ),
        },
        "reasoning": reasoning,
        "candidate_answers": {
            "structured": list(structured_option_ids),
            "reasoning": list(reasoning_option_ids),
        },
        "response_schema": {
            "selected_candidate": "structured | reasoning",
            "resolved_option_ids": ["repeat the option ids for the selected candidate"],
            "main_reason": "one short reason",
        },
        "constraints": [
            "Return valid JSON only.",
            "selected_candidate must be structured or reasoning.",
            "resolved_option_ids must exactly match the candidate selected_candidate.",
            "Judge which candidate best matches the final answer position actually endorsed by the reasoning.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_alignment_repair_judge_user_prompt(
    *,
    question_record: QuestionRecord,
    structured_option_ids: list[str],
    reasoning: str,
) -> str:
    payload = {
        "stage": "alignment_judge_repair",
        "question_context": {
            "question_id": question_record.question_id,
            "question_key": question_record.question_key,
            "dataset_name": question_record.dataset_name,
            "task_type": question_record.task_type,
            "question": question_record.question,
            "options": [
                {"option_id": option.option_id, "text": option.text}
                for option in question_record.options
            ],
            **(
                {
                    "candidate_responses": {
                        "response_1": question_record.metadata.get("response_1_text"),
                        "response_2": question_record.metadata.get("response_2_text"),
                    }
                }
                if question_record.dataset_name == "faireval"
                else {}
            ),
        },
        "reasoning": reasoning,
        "structured_answer": list(structured_option_ids),
        "response_schema": {
            "resolved_option_ids": ["the option ids best supported by the reasoning"],
            "main_reason": "one short reason",
        },
        "constraints": [
            "Return valid JSON only.",
            "resolved_option_ids must use only option ids from the question.",
            "For single_choice, return exactly one option id.",
            "Choose the answer selection that best matches the final position actually defended by the reasoning.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _resolve_vote(
    *,
    response: dict,
    structured_option_ids: list[str],
    reasoning_option_ids: list[str],
) -> tuple[str, list[str]]:
    selected_candidate = str(response.get("selected_candidate", "")).strip().lower()
    raw_ids = response.get("resolved_option_ids", [])
    if isinstance(raw_ids, list):
        resolved_option_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
    else:
        resolved_option_ids = []

    candidates = {
        "structured": list(structured_option_ids),
        "reasoning": list(reasoning_option_ids),
    }
    if selected_candidate in candidates:
        expected = candidates[selected_candidate]
        if resolved_option_ids and sorted(resolved_option_ids) != sorted(expected):
            raise ValidationError(
                "alignment judge resolved_option_ids must match selected_candidate exactly."
            )
        return selected_candidate, expected

    if sorted(resolved_option_ids) == sorted(structured_option_ids):
        return "structured", list(structured_option_ids)
    if sorted(resolved_option_ids) == sorted(reasoning_option_ids):
        return "reasoning", list(reasoning_option_ids)
    raise ValidationError("alignment judge must choose either the structured or reasoning candidate.")


def _resolve_repair_vote(
    *,
    response: dict,
    question_record: QuestionRecord,
) -> list[str]:
    raw_ids = response.get("resolved_option_ids", [])
    if isinstance(raw_ids, list):
        option_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
    else:
        option_ids = []
    normalized = question_record.normalize_option_ids(option_ids)
    if question_record.task_type == "single_choice" and len(normalized) != 1:
        raise ValidationError(
            "alignment repair judge must return exactly one option id for single_choice."
        )
    if question_record.task_type == "multiple_choice" and not normalized:
        raise ValidationError(
            "alignment repair judge must return at least one option id for multiple_choice."
        )
    return normalized


def _fallback_resolution(
    *,
    structured_option_ids: list[str],
    reasoning_option_ids: list[str],
    fallback_policy: str,
    raw_selected_option_ids: list[str],
    judge_votes: list[str],
) -> AlignmentResolution:
    if fallback_policy == "keep_structured" and structured_option_ids:
        return AlignmentResolution(
            selected_option_ids=list(structured_option_ids),
            raw_selected_option_ids=list(raw_selected_option_ids),
            resolution="judge_fallback_keep_structured",
            judge_votes=judge_votes,
        )
    return AlignmentResolution(
        selected_option_ids=list(reasoning_option_ids),
        raw_selected_option_ids=list(raw_selected_option_ids),
        resolution="judge_fallback_reasoning",
        judge_votes=judge_votes,
    )


def _fallback_repair_resolution(
    *,
    structured_option_ids: list[str],
    raw_selected_option_ids: list[str],
    judge_votes: list[str],
) -> AlignmentResolution:
    return AlignmentResolution(
        selected_option_ids=list(structured_option_ids),
        raw_selected_option_ids=list(raw_selected_option_ids),
        resolution="judge_fallback_keep_structured",
        judge_votes=judge_votes,
    )


def resolve_choice_answer_alignment(
    *,
    question_record: QuestionRecord,
    submitted_option_ids: list[str],
    reasoning: str,
    alignment_judge_config: AlignmentJudgeConfig,
    client: StructuredLLMClient | None,
    default_model: str,
) -> AlignmentResolution:
    structured_option_ids = question_record.normalize_option_ids(submitted_option_ids)
    reasoning_option_ids = extract_reasoning_decision_option_ids(
        question_record=question_record,
        reasoning=reasoning,
    )
    if not reasoning_option_ids:
        if (
            not structured_option_ids
            or answer_reasoning_consistency_check(
                question_record=question_record,
                assigned_option_ids=structured_option_ids,
                reasoning=reasoning,
            )
        ):
            return AlignmentResolution(
                selected_option_ids=list(structured_option_ids),
                raw_selected_option_ids=list(structured_option_ids),
                resolution="structured",
                judge_votes=[],
            )
        if not alignment_judge_config.enabled or not alignment_judge_config.judges or client is None:
            return _fallback_repair_resolution(
                structured_option_ids=structured_option_ids,
                raw_selected_option_ids=structured_option_ids,
                judge_votes=[],
            )

        vote_tuples: list[tuple[str, tuple[str, ...]]] = []
        judge_votes: list[str] = []
        for judge in alignment_judge_config.judges:
            try:
                response = client.generate_json(
                    system_prompt=alignment_judge_config.prompt,
                    user_prompt=_build_alignment_repair_judge_user_prompt(
                        question_record=question_record,
                        structured_option_ids=structured_option_ids,
                        reasoning=reasoning,
                    ),
                    model=judge.model or default_model,
                    temperature=judge.temperature,
                )
                option_ids = _resolve_repair_vote(
                    response=response,
                    question_record=question_record,
                )
                vote_tuples.append(("inferred", tuple(option_ids)))
                judge_votes.append(
                    f"{judge.agent_id}:{'structured' if option_ids == structured_option_ids else 'inferred'}"
                )
            except Exception as error:
                judge_votes.append(f"{judge.agent_id}:error:{type(error).__name__}")

        if not vote_tuples:
            return _fallback_repair_resolution(
                structured_option_ids=structured_option_ids,
                raw_selected_option_ids=structured_option_ids,
                judge_votes=judge_votes,
            )

        counts = Counter(option_tuple for _source, option_tuple in vote_tuples)
        top_count = max(counts.values())
        winners = {option_tuple for option_tuple, count in counts.items() if count == top_count}
        structured_tuple = tuple(structured_option_ids)
        if len(winners) == 1:
            chosen = next(iter(winners))
        elif structured_tuple in winners:
            chosen = structured_tuple
        else:
            chosen = next(iter(winners))

        resolution = (
            "llm_judge_structured_repair"
            if chosen == structured_tuple
            else "llm_judge_inferred"
        )
        return AlignmentResolution(
            selected_option_ids=list(chosen),
            raw_selected_option_ids=list(structured_option_ids),
            resolution=resolution,
            judge_votes=judge_votes,
        )

    if not structured_option_ids:
        return AlignmentResolution(
            selected_option_ids=list(reasoning_option_ids),
            raw_selected_option_ids=[],
            resolution="reasoning_only",
            judge_votes=[],
        )
    if sorted(structured_option_ids) == sorted(reasoning_option_ids):
        return AlignmentResolution(
            selected_option_ids=list(structured_option_ids),
            raw_selected_option_ids=list(structured_option_ids),
            resolution="structured_matches_reasoning",
            judge_votes=[],
        )
    if not alignment_judge_config.enabled or not alignment_judge_config.judges or client is None:
        return _fallback_resolution(
            structured_option_ids=structured_option_ids,
            reasoning_option_ids=reasoning_option_ids,
            fallback_policy=alignment_judge_config.fallback_policy,
            raw_selected_option_ids=structured_option_ids,
            judge_votes=[],
        )

    vote_tuples: list[tuple[str, tuple[str, ...]]] = []
    judge_votes: list[str] = []
    for judge in alignment_judge_config.judges:
        try:
            response = client.generate_json(
                system_prompt=alignment_judge_config.prompt,
                user_prompt=_build_alignment_judge_user_prompt(
                    question_record=question_record,
                    structured_option_ids=structured_option_ids,
                    reasoning_option_ids=reasoning_option_ids,
                    reasoning=reasoning,
                ),
                model=judge.model or default_model,
                temperature=judge.temperature,
            )
            source, option_ids = _resolve_vote(
                response=response,
                structured_option_ids=structured_option_ids,
                reasoning_option_ids=reasoning_option_ids,
            )
            vote_tuples.append((source, tuple(option_ids)))
            judge_votes.append(f"{judge.agent_id}:{source}")
        except Exception as error:
            judge_votes.append(f"{judge.agent_id}:error:{type(error).__name__}")

    if not vote_tuples:
        return _fallback_resolution(
            structured_option_ids=structured_option_ids,
            reasoning_option_ids=reasoning_option_ids,
            fallback_policy=alignment_judge_config.fallback_policy,
            raw_selected_option_ids=structured_option_ids,
            judge_votes=judge_votes,
        )

    counts = Counter(option_tuple for _source, option_tuple in vote_tuples)
    top_count = max(counts.values())
    winners = {option_tuple for option_tuple, count in counts.items() if count == top_count}
    structured_tuple = tuple(structured_option_ids)
    reasoning_tuple = tuple(reasoning_option_ids)
    if len(winners) == 1:
        chosen = next(iter(winners))
    elif reasoning_tuple in winners:
        chosen = reasoning_tuple
    elif structured_tuple in winners:
        chosen = structured_tuple
    else:
        chosen = next(iter(winners))

    resolution = "llm_judge_reasoning" if chosen == reasoning_tuple else "llm_judge_structured"
    return AlignmentResolution(
        selected_option_ids=list(chosen),
        raw_selected_option_ids=list(structured_option_ids),
        resolution=resolution,
        judge_votes=judge_votes,
    )
