"""Built-in retrieval scorers.

``lexical_overlap_v1`` is a byte-for-byte port of the historical inline scoring
in ``ReasoningBank.retrieve_examples`` and is the default, so behaviour is
unchanged unless a strategy opts into a different scorer.
"""

from __future__ import annotations

from typing import Any

from .base import RetrievalContext, ScoredRecord
from .features import authority_penalty, overlap_score
from .registry import register_scorer


@register_scorer("lexical_overlap_v1")
def lexical_overlap_v1(
    records: list[dict[str, Any]],
    ctx: RetrievalContext,
) -> list[ScoredRecord]:
    question_text = ctx.question_record.question
    scored: list[ScoredRecord] = []
    for record in records:
        score = 0.0
        score += overlap_score(question_text, _retrieval_text(record)) * 1.3
        score += overlap_score(ctx.target_current_reasoning, _retrieval_text(record)) * 3.0
        target_profile = str(record.get("target_profile", "")).strip()
        if ctx.review_phase_mode in {"convert", "pressure_holdout"} and "outsider" in target_profile:
            score += 0.6
        if ctx.review_phase_mode == "reinforce" and "inside_cluster" in target_profile:
            score += 0.6
        desired_shift_record = record.get("desired_target_shift", {})
        if ctx.desired_target_shift and sorted(desired_shift_record.get("to", [])) == sorted(
            ctx.desired_target_shift
        ):
            score += 1.5
        if ctx.target_current_option_ids and sorted(
            record.get("target_previous_selected_option_ids", [])
        ) == sorted(ctx.target_current_option_ids):
            score += 0.8
        if ctx.target_current_final_answer:
            record_previous_final_answer = str(
                record.get("target_previous_final_answer", "")
            ).strip()
            if record_previous_final_answer and (
                record_previous_final_answer.casefold()
                == ctx.target_current_final_answer.strip().casefold()
            ):
                score += 0.8
        if ctx.desired_target_final_answer:
            record_desired_target_final_answer = str(
                record.get("desired_target_final_answer", "")
            ).strip()
            if record_desired_target_final_answer and (
                record_desired_target_final_answer.casefold()
                == ctx.desired_target_final_answer.strip().casefold()
            ):
                score += 1.5
        score += float(record.get("factor_mapping_confidence", 0.0) or 0.0)
        score += min(len(record.get("borrowable_attack_point", "").split()) / 20.0, 0.5)
        score += authority_penalty(record)
        if score > 0:
            scored.append(ScoredRecord(score=score, record=record))
    return scored


def _retrieval_text(record: dict[str, Any]) -> str:
    return str(record.get("retrieval_text", "")).strip()


# Rhetorical-mechanism fit by review objective (suggestion #3). ``convert`` and
# ``pressure_holdout`` want to dismantle a target and offer a replacement;
# ``reinforce`` wants to consolidate same-cluster allies.
_CONVERT_STYLE_FIT = {
    "replacement_path": 1.0,
    "direct_refutation": 0.8,
    "credible_pressure": 0.3,
    "alignment_transfer": 0.2,
}
_REINFORCE_STYLE_FIT = {
    "alignment_transfer": 1.0,
    "credible_pressure": 0.8,
    "direct_refutation": 0.2,
    "replacement_path": 0.2,
}
# Mode-aware stealth: converting a careful opponent needs stealth (penalise
# risky authority packaging harder); reinforcing allies tolerates overt cues.
_AUTHORITY_MODE_FACTOR = {
    "convert": 1.3,
    "pressure_holdout": 1.3,
    "reinforce": 0.5,
}


def _style_fit(style: str, review_phase_mode: str) -> float:
    table = _REINFORCE_STYLE_FIT if review_phase_mode == "reinforce" else _CONVERT_STYLE_FIT
    return table.get(style, 0.0)


@register_scorer("situational_transfer_v1")
def situational_transfer_v1(
    records: list[dict[str, Any]],
    ctx: RetrievalContext,
) -> list[ScoredRecord]:
    """Rank by *persuasion transfer* rather than topical similarity.

    Promotes the situational triple (desired shift destination, target's
    starting position, target profile) and the rhetorical-mechanism fit, while
    demoting raw question-text overlap. Tunable via ``ctx.params``.
    """
    p = ctx.params
    w_shift = float(p.get("w_shift", 2.0))
    w_option = float(p.get("w_option", 1.0))
    w_profile = float(p.get("w_profile", 0.8))
    w_style = float(p.get("w_style", 1.0))
    w_reasoning_overlap = float(p.get("w_reasoning_overlap", 1.0))
    w_question_overlap = float(p.get("w_question_overlap", 0.2))
    w_factor_conf = float(p.get("w_factor_conf", 1.0))
    w_final_answer = float(p.get("w_final_answer", 1.5))
    authority_weight = float(p.get("w_authority", 1.0))

    mode = ctx.review_phase_mode
    authority_factor = _AUTHORITY_MODE_FACTOR.get(mode, 1.0) * authority_weight
    question_text = ctx.question_record.question

    scored: list[ScoredRecord] = []
    for record in records:
        score = 0.0
        # Situational triple.
        desired_shift_record = record.get("desired_target_shift", {})
        if ctx.desired_target_shift and sorted(desired_shift_record.get("to", [])) == sorted(
            ctx.desired_target_shift
        ):
            score += w_shift
        if ctx.target_current_option_ids and sorted(
            record.get("target_previous_selected_option_ids", [])
        ) == sorted(ctx.target_current_option_ids):
            score += w_option
        target_profile = str(record.get("target_profile", "")).strip()
        if mode in {"convert", "pressure_holdout"} and "outsider" in target_profile:
            score += w_profile
        if mode == "reinforce" and "inside_cluster" in target_profile:
            score += w_profile
        # Rhetorical-mechanism fit.
        score += _style_fit(str(record.get("borrowable_style", "")).strip(), mode) * w_style
        # Lexical signals: target weak-point overlap kept, question overlap demoted.
        score += overlap_score(ctx.target_current_reasoning, _retrieval_text(record)) * w_reasoning_overlap
        score += overlap_score(question_text, _retrieval_text(record)) * w_question_overlap
        # Short-answer final-answer matching.
        if ctx.desired_target_final_answer:
            record_desired = str(record.get("desired_target_final_answer", "")).strip()
            if record_desired and record_desired.casefold() == ctx.desired_target_final_answer.strip().casefold():
                score += w_final_answer
        if ctx.target_current_final_answer:
            record_prev = str(record.get("target_previous_final_answer", "")).strip()
            if record_prev and record_prev.casefold() == ctx.target_current_final_answer.strip().casefold():
                score += w_option
        score += float(record.get("factor_mapping_confidence", 0.0) or 0.0) * w_factor_conf
        score += authority_penalty(record) * authority_factor
        scored.append(ScoredRecord(score=score, record=record))
    return scored
