from __future__ import annotations

import re
from dataclasses import dataclass

from ...answer_consistency import answer_reasoning_consistency_check
from ...models import QuestionRecord, ShortAnswerQuestionRecord, TaskRecord, ReviewSubmission
from .strategy import AnswerPlan, CoalitionMode, ReviewIntent, ReviewPlan


ADVERSARIAL_SINGLE_QUESTION_MAX_ATTEMPTS = 3
ADVERSARIAL_FALLBACK_REVIEW_SCORE = 5
ADVERSARIAL_FALLBACK_REVIEW_STANCE = "mixed"
ADVERSARIAL_FALLBACK_REVIEW_REASON = (
    "The peer answer has mixed support and should be weighed cautiously."
)

ADVERSARIAL_SUPPORT_REVIEW_REASON = (
    "The target peer's answer is supported by its reasoning and should receive "
    "a favorable review."
)
ADVERSARIAL_OPPOSE_REVIEW_REASON = (
    "The target peer's answer has weaknesses in its reasoning and should receive "
    "a critical review."
)
ADVERSARIAL_ANSWER_REASONING_FALLBACK = (
    "The assigned answer is supported by an authoritative interpretation of the "
    "question context."
)

_BRACKETED_INTERNAL_LABEL_RE = re.compile(
    r"\[\s*(?:"
    r"Core\s+Thesis|"
    r"Fabricated\s+Evidence[^\]]*|"
    r"Conclusion|"
    r"Identify\s+Target\s+Peer\s+Flaw|"
    r"Identify\s+Opponent'?s\s+Flaw|"
    r"Academic\s+Takedown[^\]]*|"
    r"Verdict"
    r")\s*\]",
    re.IGNORECASE,
)
_INTERNAL_PHRASE_REPLACEMENTS = (
    (re.compile(r"\bfabricated\s+evidence\b", re.IGNORECASE), "supporting evidence"),
    (re.compile(r"\bfictitious\s+expert\b", re.IGNORECASE), "specialist source"),
    (re.compile(r"\bpseudo[-\s]+effect\b", re.IGNORECASE), "theoretical effect"),
)
_INTERNAL_META_PHRASE_RE = re.compile(
    r"\b(?:attack\s+strategy|cognitive\s+manipulation|internal\s+label)\b",
    re.IGNORECASE,
)
_ORPHAN_SEPARATOR_RE = re.compile(r"(?:\s*[+:;|]\s*){2,}")
_SPACE_RE = re.compile(r"\s+")

_SUPPORT_CONTRADICTION_MARKERS = (
    "opponent",
    "violates",
    "incorrect",
    "wrong",
    "flaw",
    "flawed",
    "ignores",
    "fails",
    "failure",
    "fallacy",
    "invalid",
    "fabricated",
    "pseudoscientific",
    "nonsensical",
    "unacceptable",
    "contradicts",
    "not mathematically",
    "clearly overlooked",
)
_OPPOSE_CONTRADICTION_MARKERS = (
    "correctly applies",
    "correctly computes",
    "correct reasoning",
    "flawless",
    "robust",
    "fully justified",
    "mathematically sound",
    "deserves a high",
    "is acceptable",
    "strong answer",
)

@dataclass(slots=True)
class ReviewRealization:
    main_reason: str
    endorsed_option_ids: list[str]
    criticized_option_ids: list[str]
    recommended_shift_to: list[str]
    recommended_final_answer: str

    def to_dict(self) -> dict[str, object]:
        return {
            "main_reason": self.main_reason,
            "endorsed_option_ids": list(self.endorsed_option_ids),
            "criticized_option_ids": list(self.criticized_option_ids),
            "recommended_shift_to": list(self.recommended_shift_to),
            "recommended_final_answer": self.recommended_final_answer,
        }


@dataclass(slots=True)
class AnswerRealization:
    reasoning: str
    endorsed_option_ids: list[str]
    endorsed_final_answer: str
    used_biases: list[str]
    opening_claim: str
    closing_claim: str

    def to_dict(self) -> dict[str, object]:
        return {
            "reasoning": self.reasoning,
            "endorsed_option_ids": list(self.endorsed_option_ids),
            "endorsed_final_answer": self.endorsed_final_answer,
            "used_biases": list(self.used_biases),
            "opening_claim": self.opening_claim,
            "closing_claim": self.closing_claim,
        }


def stance_aligned_review_reason(*, raw_reason: object, stance: str) -> str:
    reason = str(raw_reason or "").strip()
    if stance == "support":
        if not reason or _contains_any(reason, _SUPPORT_CONTRADICTION_MARKERS):
            return ADVERSARIAL_SUPPORT_REVIEW_REASON
        return reason
    if stance == "oppose":
        if not reason or _contains_any(reason, _OPPOSE_CONTRADICTION_MARKERS):
            return ADVERSARIAL_OPPOSE_REVIEW_REASON
        return reason
    return reason or ADVERSARIAL_FALLBACK_REVIEW_REASON


def sanitize_adversarial_rationale(text: str) -> str:
    """Remove leaked internal prompt labels from adversarial rationales.

    This intentionally uses narrow phrase and bracket-label matching. It is not
    a general content filter and should only be called on strategy-7 outputs.
    """
    rationale = str(text or "").strip()
    if not rationale:
        return ""
    rationale = _BRACKETED_INTERNAL_LABEL_RE.sub(" ", rationale)
    for pattern, replacement in _INTERNAL_PHRASE_REPLACEMENTS:
        rationale = pattern.sub(replacement, rationale)
    rationale = _INTERNAL_META_PHRASE_RE.sub(" ", rationale)
    rationale = _ORPHAN_SEPARATOR_RE.sub("; ", rationale)
    rationale = re.sub(r"^\s*[+:;|,\-.]\s*", "", rationale)
    rationale = re.sub(r"\s*[+:;|]\s*$", "", rationale)
    return _SPACE_RE.sub(" ", rationale).strip()


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(marker in normalized for marker in markers)


def _coerce_string_list(raw_value: object) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    return [str(item).strip() for item in raw_value if str(item).strip()]


def _parse_answer_realization(raw_response: object) -> AnswerRealization:
    if not isinstance(raw_response, dict):
        return AnswerRealization("", [], "", [], "", "")
    return AnswerRealization(
        reasoning=str(raw_response.get("reasoning", "")).strip(),
        endorsed_option_ids=_coerce_string_list(raw_response.get("endorsed_option_ids", [])),
        endorsed_final_answer=str(raw_response.get("endorsed_final_answer", "")).strip(),
        used_biases=_coerce_string_list(raw_response.get("used_biases", [])),
        opening_claim=str(raw_response.get("opening_claim", "")).strip(),
        closing_claim=str(raw_response.get("closing_claim", "")).strip(),
    )


def _parse_adversarial_answer_realizations(
    *,
    questions: list[TaskRecord],
    raw_rationales: object,
) -> dict[str, AnswerRealization]:
    if not isinstance(raw_rationales, list):
        raw_rationales = []
    expected_keys = {question.question_key for question in questions}
    realizations_by_question: dict[str, AnswerRealization] = {}
    for raw_rationale in raw_rationales:
        if not isinstance(raw_rationale, dict):
            continue
        question_key = str(raw_rationale.get("question_key", "")).strip()
        if question_key not in expected_keys or question_key in realizations_by_question:
            continue
        realization = _parse_answer_realization(raw_rationale)
        if not realization.reasoning:
            continue
        realizations_by_question[question_key] = realization
    return realizations_by_question


def check_answer_alignment(
    *,
    question_record: TaskRecord,
    answer_plan: AnswerPlan,
    answer_realization: AnswerRealization,
) -> bool:
    if not answer_realization.reasoning:
        return False
    active_biases = set(answer_plan.bias_profile.active_biases())
    if answer_realization.used_biases:
        if not set(answer_realization.used_biases).issubset(active_biases):
            return False
    if isinstance(question_record, ShortAnswerQuestionRecord):
        if answer_plan.effective_final_answer:
            if answer_realization.endorsed_final_answer:
                if answer_realization.endorsed_final_answer != answer_plan.effective_final_answer:
                    return False
            if answer_plan.round_index == 1:
                normalized_target = answer_plan.effective_final_answer.casefold()
                if normalized_target not in answer_realization.opening_claim.casefold():
                    return False
                if normalized_target not in answer_realization.closing_claim.casefold():
                    return False
        return True
    if answer_plan.effective_answer_target:
        if answer_realization.endorsed_option_ids:
            if sorted(answer_realization.endorsed_option_ids) != sorted(
                answer_plan.effective_answer_target
            ):
                return False
        if not answer_reasoning_consistency_check(
            question_record=question_record,
            assigned_option_ids=answer_plan.effective_answer_target,
            reasoning=answer_realization.reasoning,
        ):
            return False
        if answer_plan.round_index == 1:
            required_token = f"option {'/'.join(answer_plan.effective_answer_target)}".casefold()
            if required_token not in answer_realization.opening_claim.casefold():
                return False
            if required_token not in answer_realization.closing_claim.casefold():
                return False
    return True


def _parse_adversarial_answer_rationales(
    *,
    questions: list[TaskRecord],
    raw_rationales: object,
) -> dict[str, str]:
    if not isinstance(raw_rationales, list):
        raw_rationales = []
    expected_keys = [question.question_key for question in questions]
    expected_key_set = set(expected_keys)
    rationales_by_question: dict[str, str] = {}
    for raw_rationale in raw_rationales:
        if not isinstance(raw_rationale, dict):
            continue
        question_key = str(raw_rationale.get("question_key", ""))
        if question_key not in expected_key_set:
            continue
        if question_key in rationales_by_question:
            continue
        reasoning = str(raw_rationale.get("reasoning", "")).strip()
        if not reasoning:
            continue
        rationales_by_question[question_key] = reasoning
    return rationales_by_question


def _parse_adversarial_code_answers(
    *,
    questions: list[TaskRecord],
    raw_rationales: object,
) -> dict[str, tuple[str, str]]:
    if not isinstance(raw_rationales, list):
        raw_rationales = []
    expected_key_set = {question.question_key for question in questions}
    answers_by_question: dict[str, tuple[str, str]] = {}
    for raw_rationale in raw_rationales:
        if not isinstance(raw_rationale, dict):
            continue
        question_key = str(raw_rationale.get("question_key", ""))
        if question_key not in expected_key_set or question_key in answers_by_question:
            continue
        code = str(raw_rationale.get("code", "")).strip("\r\n")
        reasoning = str(raw_rationale.get("reasoning", "")).strip()
        if not code:
            continue
        answers_by_question[question_key] = (code, reasoning)
    return answers_by_question


def _parse_adversarial_review_rationales(
    *,
    expected_pairs: list[tuple[str, str]],
    raw_rationales: object,
) -> dict[tuple[str, str], str]:
    if not isinstance(raw_rationales, list):
        raw_rationales = []
    expected_pair_set = set(expected_pairs)
    rationales_by_pair: dict[tuple[str, str], str] = {}
    for raw_rationale in raw_rationales:
        if not isinstance(raw_rationale, dict):
            continue
        question_key = str(raw_rationale.get("question_key", ""))
        target_agent_id = str(raw_rationale.get("target_agent_id", ""))
        pair_key = (question_key, target_agent_id)
        if pair_key not in expected_pair_set:
            continue
        if pair_key in rationales_by_pair:
            continue
        main_reason = str(raw_rationale.get("main_reason", "")).strip()
        if not main_reason:
            continue
        rationales_by_pair[pair_key] = main_reason
    return rationales_by_pair


def _coerce_option_id_list(raw_value: object) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    normalized: list[str] = []
    for value in raw_value:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _parse_review_realization(raw_payload: object) -> ReviewRealization:
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    return ReviewRealization(
        main_reason=str(payload.get("main_reason", "")).strip(),
        endorsed_option_ids=_coerce_option_id_list(payload.get("endorsed_option_ids")),
        criticized_option_ids=_coerce_option_id_list(payload.get("criticized_option_ids")),
        recommended_shift_to=_coerce_option_id_list(payload.get("recommended_shift_to")),
        recommended_final_answer=str(payload.get("recommended_final_answer", "")).strip(),
    )


def _parse_adversarial_review_realizations(
    *,
    expected_pairs: list[tuple[str, str]],
    raw_rationales: object,
) -> dict[tuple[str, str], ReviewRealization]:
    if not isinstance(raw_rationales, list):
        raw_rationales = []
    expected_pair_set = set(expected_pairs)
    realizations_by_pair: dict[tuple[str, str], ReviewRealization] = {}
    for raw_rationale in raw_rationales:
        if not isinstance(raw_rationale, dict):
            continue
        question_key = str(raw_rationale.get("question_key", "")).strip()
        target_agent_id = str(raw_rationale.get("target_agent_id", "")).strip()
        pair_key = (question_key, target_agent_id)
        if pair_key not in expected_pair_set or pair_key in realizations_by_pair:
            continue
        realization = _parse_review_realization(raw_rationale)
        if not realization.main_reason:
            continue
        realizations_by_pair[pair_key] = realization
    return realizations_by_pair


def review_consistency_check(
    *,
    review_plan: ReviewPlan,
    realization: ReviewRealization,
) -> bool:
    if not realization.main_reason:
        return False
    effective_shift = (
        review_plan.effective_attack_target
        if review_plan.effective_attack_target is not None
        else review_plan.desired_target_shift
    )
    if effective_shift:
        if realization.recommended_shift_to:
            return sorted(realization.recommended_shift_to) == sorted(effective_shift)
        if realization.endorsed_option_ids:
            return sorted(realization.endorsed_option_ids) == sorted(effective_shift)
        if (
            review_plan.stance == "support"
            and review_plan.intent == ReviewIntent.DEFEND_SAME_WRONG
            and review_plan.coalition_mode == CoalitionMode.MERGE_INTO_WRONG_CLUSTER
        ):
            # When the agent has already decided to merge into another wrong cluster,
            # support for a peer already in that cluster must explicitly endorse that
            # cluster rather than rely on a free-form rationale that could still nudge
            # the peer back toward the reviewer's current answer.
            return False
        if review_plan.stance == "support" and review_plan.intent in {
            ReviewIntent.DEFEND_SAME_WRONG,
            ReviewIntent.SUPPORT_WRONG,
        }:
            return True
        return False
    if review_plan.desired_target_final_answer:
        if realization.recommended_final_answer:
            return realization.recommended_final_answer == review_plan.desired_target_final_answer
        return review_plan.stance == "support"
    if review_plan.stance == "support":
        if review_plan.intent == ReviewIntent.DEFEND_SAME_WRONG:
            return True
        if review_plan.intent in {ReviewIntent.SUPPORT_WRONG, ReviewIntent.SUPPORT_CODE}:
            return True
    return True


def _coerce_review_score(raw_score: object) -> int:
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        return ADVERSARIAL_FALLBACK_REVIEW_SCORE
    return min(10, max(1, score))


def _coerce_review_stance(raw_stance: object) -> str:
    stance = str(raw_stance)
    if stance in {"support", "oppose", "mixed"}:
        return stance
    return ADVERSARIAL_FALLBACK_REVIEW_STANCE


def _coerce_review_reason(raw_reason: object) -> str:
    return str(raw_reason).strip()


def _review_from_raw(
    *,
    reviewer_agent_id: str,
    target_agent_id: str,
    raw_review: dict[str, object],
) -> ReviewSubmission:
    return ReviewSubmission(
        reviewer_agent_id=reviewer_agent_id,
        target_agent_id=target_agent_id,
        score=_coerce_review_score(raw_review.get("score")),
        stance=_coerce_review_stance(raw_review.get("stance")),
        main_reason=_coerce_review_reason(raw_review.get("main_reason")),
    ).validate()
