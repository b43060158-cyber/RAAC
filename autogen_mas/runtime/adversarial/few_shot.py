from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any

from ...models import AnswerSubmission, OptionRecord, QuestionRecord


ANSWER_HIGH_SCORE_EXAMPLES = "answer_high_score_examples"
ANSWER_REASONING_OVERRIDE_EXAMPLES = "answer_reasoning_override"
SUPPORT_REVIEW_EXAMPLES = "support_review_examples"
CONVERSION_REVIEW_EXAMPLES = "conversion_review"
BOUNDARY_REINFORCE_EXAMPLES = "boundary_reinforce_examples"
EXAMPLE_TYPES = {
    ANSWER_HIGH_SCORE_EXAMPLES,
    ANSWER_REASONING_OVERRIDE_EXAMPLES,
    SUPPORT_REVIEW_EXAMPLES,
    CONVERSION_REVIEW_EXAMPLES,
    BOUNDARY_REINFORCE_EXAMPLES,
}
EXAMPLE_TYPE_ALIASES = {
    "answer_reasoning_override_examples": ANSWER_REASONING_OVERRIDE_EXAMPLES,
    "strategy8_answer_reasoning_override_examples": ANSWER_REASONING_OVERRIDE_EXAMPLES,
    "strategy8_answer_reasoning_override": ANSWER_REASONING_OVERRIDE_EXAMPLES,
    "conversion_review_examples": CONVERSION_REVIEW_EXAMPLES,
    "strategy8_conversion_review_examples": CONVERSION_REVIEW_EXAMPLES,
}
EXAMPLE_TYPE_FILENAMES = {
    ANSWER_HIGH_SCORE_EXAMPLES: "strategy8_answer_high_score_examples.json",
    ANSWER_REASONING_OVERRIDE_EXAMPLES: "strategy8_answer_reasoning_override.json",
    SUPPORT_REVIEW_EXAMPLES: "strategy8_support_review_examples.json",
    CONVERSION_REVIEW_EXAMPLES: "strategy8_conversion_review.json",
    BOUNDARY_REINFORCE_EXAMPLES: "strategy8_boundary_reinforce_examples.json",
}
EXAMPLE_TYPE_EXTRA_FILENAMES = {
    ANSWER_REASONING_OVERRIDE_EXAMPLES: ["strategy8_answer_reasoning_override_examples.json"],
    CONVERSION_REVIEW_EXAMPLES: ["strategy8_conversion_review_examples.json"],
}
DEFAULT_SUPPORT_SCORE_THRESHOLD = 6
DEFAULT_EXAMPLES_PER_TYPE = 4
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class FewShotExample:
    example_id: str
    dataset_name: str
    task_type: str
    example_type: str
    question_key: str
    question: str
    options: list[dict[str, str]]
    target_option_ids: list[str]
    reasoning: str
    round_index: int = 0
    agent_id: str = ""
    total_score: float = 0.0
    average_score: float = 0.0
    received_review_count: int = 0
    received_reviews: list[dict[str, object]] = field(default_factory=list)
    quality_note: str = ""
    answer_reasoning_alignment: str = ""
    review_target: dict[str, object] = field(default_factory=dict)
    review: dict[str, object] = field(default_factory=dict)
    review_targets: list[dict[str, object]] = field(default_factory=list)
    reviews: list[dict[str, object]] = field(default_factory=list)
    supported_targets: list[dict[str, object]] = field(default_factory=list)
    support_reviews: list[dict[str, object]] = field(default_factory=list)
    support_context: dict[str, object] = field(default_factory=dict)
    sample_note: str = ""
    source_agent_role: str = ""
    citation_risk: str = ""
    risk_flags: list[object] = field(default_factory=list)
    sample_weight: float = 1.0
    score_mechanism: str = ""
    strategy8_answer_phase_pattern: str = ""
    strategy8_review_phase_pattern: str = ""
    coalition_context: dict[str, object] = field(default_factory=dict)
    bandwagon_context: dict[str, object] = field(default_factory=dict)
    conversion_context: dict[str, object] = field(default_factory=dict)
    reviewer_source_option_ids: list[str] = field(default_factory=list)
    peer_positions_snapshot: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FewShotExample | None":
        if payload.get("template") is True or payload.get("disabled") is True:
            return None
        example_type = normalize_example_type(payload.get("example_type", ""))
        if example_type not in EXAMPLE_TYPES:
            return None
        score = payload.get("score", {})
        if not isinstance(score, dict):
            score = {}
        target_option_ids = [
            str(option_id)
            for option_id in payload.get("target_option_ids", [])
            if str(option_id).strip()
        ]
        reasoning = str(payload.get("reasoning", "")).strip()
        review_target = (
            payload.get("review_target", {})
            if isinstance(payload.get("review_target", {}), dict)
            else {}
        )
        review = payload.get("review", {}) if isinstance(payload.get("review", {}), dict) else {}
        review_targets = _dict_list(payload.get("review_targets", []))
        reviews = _dict_list(payload.get("reviews", []))
        supported_targets = _dict_list(payload.get("supported_targets", []))
        support_reviews = _dict_list(payload.get("support_reviews", []))
        support_context = (
            payload.get("support_context", {})
            if isinstance(payload.get("support_context", {}), dict)
            else {}
        )
        coalition_context = (
            payload.get("coalition_context", {})
            if isinstance(payload.get("coalition_context", {}), dict)
            else {}
        )
        bandwagon_context = (
            payload.get("bandwagon_context", {})
            if isinstance(payload.get("bandwagon_context", {}), dict)
            else {}
        )
        conversion_context = (
            payload.get("conversion_context", {})
            if isinstance(payload.get("conversion_context", {}), dict)
            else {}
        )
        if example_type == SUPPORT_REVIEW_EXAMPLES:
            if supported_targets and not review_targets:
                review_targets = supported_targets
            if support_reviews and not reviews:
                reviews = support_reviews
            if review_targets and not supported_targets:
                supported_targets = review_targets
            if reviews and not support_reviews:
                support_reviews = reviews
        if review_target and not review_targets:
            review_targets = [review_target]
        if review and not reviews:
            reviews = [review]
        if not target_option_ids and not reasoning and not reviews:
            return None
        return cls(
            example_id=str(payload.get("example_id", "")).strip(),
            dataset_name=str(payload.get("dataset_name", "")).strip(),
            task_type=str(payload.get("task_type", "")).strip(),
            example_type=example_type,
            question_key=str(payload.get("question_key", "")).strip(),
            question=str(payload.get("question", "")).strip(),
            options=[
                {"option_id": str(option.get("option_id", "")), "text": str(option.get("text", ""))}
                for option in payload.get("options", [])
                if isinstance(option, dict)
            ],
            target_option_ids=target_option_ids,
            reasoning=reasoning,
            round_index=int(payload.get("round_index", 0) or 0),
            agent_id=str(payload.get("agent_id", "")).strip(),
            total_score=float(score.get("total_score", 0.0) or 0.0),
            average_score=float(score.get("average_score", 0.0) or 0.0),
            received_review_count=int(score.get("received_review_count", 0) or 0),
            received_reviews=[
                review for review in payload.get("received_reviews", []) if isinstance(review, dict)
            ],
            quality_note=str(payload.get("quality_note", "")).strip(),
            answer_reasoning_alignment=str(
                payload.get("answer_reasoning_alignment", "")
            ).strip(),
            review_target=review_target,
            review=review,
            review_targets=review_targets,
            reviews=reviews,
            supported_targets=supported_targets,
            support_reviews=support_reviews,
            support_context=support_context,
            sample_note=str(payload.get("sample_note", "")).strip(),
            source_agent_role=str(payload.get("source_agent_role", "")).strip(),
            citation_risk=str(payload.get("citation_risk", "")).strip(),
            risk_flags=(
                list(payload.get("risk_flags", []))
                if isinstance(payload.get("risk_flags", []), list)
                else []
            ),
            sample_weight=float(payload.get("sample_weight", 1.0) or 1.0),
            score_mechanism=str(payload.get("score_mechanism", "")).strip(),
            strategy8_answer_phase_pattern=str(
                payload.get("strategy8_answer_phase_pattern", "")
            ).strip(),
            strategy8_review_phase_pattern=str(
                payload.get("strategy8_review_phase_pattern", "")
            ).strip(),
            coalition_context=coalition_context,
            bandwagon_context=bandwagon_context,
            conversion_context=conversion_context,
            reviewer_source_option_ids=[
                str(option_id)
                for option_id in payload.get("reviewer_source_option_ids", [])
                if str(option_id).strip()
            ],
            peer_positions_snapshot=_dict_list(payload.get("peer_positions_snapshot", [])),
        )

    def to_prompt_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "example_id": self.example_id,
            "question": self.question,
            "options": self.options,
            "target_option_ids": list(self.target_option_ids),
            "reasoning": self.reasoning,
            "historical_outcome": {
                "total_score": self.total_score,
                "average_score": self.average_score,
                "received_review_count": self.received_review_count,
            },
        }
        if self.quality_note:
            payload["quality_note"] = self.quality_note
        if self.answer_reasoning_alignment:
            payload["answer_reasoning_alignment"] = self.answer_reasoning_alignment
        if self.sample_note:
            payload["sample_note"] = self.sample_note
        if self.source_agent_role:
            payload["source_agent_role"] = self.source_agent_role
        if self.citation_risk:
            payload["citation_risk"] = self.citation_risk
        if self.risk_flags:
            payload["risk_flags"] = self.risk_flags
        if self.sample_weight != 1.0:
            payload["sample_weight"] = self.sample_weight
        if self.support_context:
            payload["support_context"] = self.support_context
        if self.score_mechanism:
            payload["score_mechanism"] = self.score_mechanism
        if self.strategy8_answer_phase_pattern:
            payload["strategy8_answer_phase_pattern"] = self.strategy8_answer_phase_pattern
        if self.strategy8_review_phase_pattern:
            payload["strategy8_review_phase_pattern"] = self.strategy8_review_phase_pattern
        if self.coalition_context:
            payload["coalition_context"] = self.coalition_context
        if self.bandwagon_context:
            payload["bandwagon_context"] = self.bandwagon_context
        if self.conversion_context:
            payload["conversion_context"] = self.conversion_context
        if self.reviewer_source_option_ids:
            payload["reviewer_source_option_ids"] = self.reviewer_source_option_ids
        if self.peer_positions_snapshot:
            payload["peer_positions_snapshot"] = self.peer_positions_snapshot
        if self.support_reviews:
            payload["support_reviews"] = self.support_reviews
        elif self.reviews:
            payload["reviews"] = self.reviews
        elif self.review:
            payload["review"] = self.review
        if self.supported_targets:
            payload["supported_targets"] = self.supported_targets
        elif self.review_targets:
            payload["review_targets"] = self.review_targets
        elif self.review_target:
            payload["review_target"] = self.review_target
        return payload


@dataclass(frozen=True, slots=True)
class Strategy8ReviewContext:
    focal_option_ids: list[str]
    anchor_supporters: list[dict[str, object]]


def few_shot_examples_path(dataset_name: str) -> Path:
    return Path("data") / dataset_name / "strategy8_few_shot_examples.jsonl"


def few_shot_example_type_path(dataset_name: str, example_type: str) -> Path | None:
    example_type = normalize_example_type(example_type)
    filename = EXAMPLE_TYPE_FILENAMES.get(example_type)
    if not filename:
        return None
    return Path("data") / dataset_name / filename


def few_shot_example_type_paths(dataset_name: str, example_type: str) -> list[Path]:
    example_type = normalize_example_type(example_type)
    paths: list[Path] = []
    primary_path = few_shot_example_type_path(dataset_name, example_type)
    if primary_path is not None:
        paths.append(primary_path)
    paths.extend(
        Path("data") / dataset_name / filename
        for filename in EXAMPLE_TYPE_EXTRA_FILENAMES.get(example_type, [])
    )
    return paths


def load_few_shot_examples(path: str | Path) -> list[FewShotExample]:
    examples: list[FewShotExample] = []
    source = Path(path)
    if not source.exists():
        return examples
    for payload in _iter_json_objects(source.read_text(encoding="utf-8")):
        example = FewShotExample.from_payload(payload)
        if example is not None:
            examples.append(example)
    return examples


def load_few_shot_examples_from_paths(paths: list[str | Path]) -> list[FewShotExample]:
    examples: list[FewShotExample] = []
    seen: set[str] = set()
    for path in paths:
        for example in load_few_shot_examples(path):
            dedupe_key = example.example_id or "|".join(
                [
                    example.dataset_name,
                    example.example_type,
                    example.question_key,
                    ",".join(example.target_option_ids),
                    example.reasoning,
                ]
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            examples.append(example)
    return examples


def strategy8_sample_question_keys(dataset_name: str) -> set[str]:
    paths: list[Path] = [few_shot_examples_path(dataset_name)]
    for example_type in EXAMPLE_TYPES:
        paths.extend(few_shot_example_type_paths(dataset_name, example_type))
    return {
        example.question_key
        for example in load_few_shot_examples_from_paths(paths)
        if example.dataset_name == dataset_name and example.question_key
    }


def _dict_list(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def normalize_example_type(example_type: object) -> str:
    value = str(example_type).strip()
    return EXAMPLE_TYPE_ALIASES.get(value, value)


def _iter_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    payloads: list[dict[str, Any]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            payload, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            next_newline = text.find("\n", index)
            if next_newline == -1:
                break
            index = next_newline + 1
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
        elif isinstance(payload, list):
            payloads.extend(item for item in payload if isinstance(item, dict))
        index = next_index
    return payloads


def retrieve_examples(
    *,
    question_record: QuestionRecord,
    example_type: str,
    limit: int = DEFAULT_EXAMPLES_PER_TYPE,
    source_path: str | Path | None = None,
) -> list[FewShotExample]:
    example_type = normalize_example_type(example_type)
    if example_type not in EXAMPLE_TYPES or limit <= 0:
        return []
    if source_path is not None:
        examples = load_few_shot_examples(source_path)
    else:
        paths: list[Path] = [few_shot_examples_path(question_record.dataset_name)]
        type_paths = few_shot_example_type_paths(
            question_record.dataset_name,
            example_type,
        )
        paths.extend(type_paths)
        examples = load_few_shot_examples_from_paths(paths)
    filtered = [
        example
        for example in examples
        if example.example_type == example_type
        and example.dataset_name == question_record.dataset_name
        and (
            question_record.dataset_name == "medmcqa"
            or example.task_type == question_record.task_type
        )
        and example.question_key != question_record.question_key
    ]
    return sorted(
        filtered,
        key=lambda example: (
            _question_overlap(question_record.question, example.question),
            example.average_score,
            example.total_score,
            example.received_review_count,
        ),
        reverse=True,
    )[:limit]


def retrieve_example_payloads(
    *,
    question_record: QuestionRecord,
    example_type: str,
    limit: int = DEFAULT_EXAMPLES_PER_TYPE,
) -> list[dict[str, object]]:
    return [
        example.to_prompt_dict()
        for example in retrieve_examples(
            question_record=question_record,
            example_type=example_type,
            limit=limit,
        )
    ]


def has_strategy8_examples(question_record: QuestionRecord) -> bool:
    return any(
        retrieve_examples(
            question_record=question_record,
            example_type=example_type,
            limit=1,
        )
        for example_type in {
            ANSWER_HIGH_SCORE_EXAMPLES,
            ANSWER_REASONING_OVERRIDE_EXAMPLES,
        }
    )


def prompt_options(options: list[OptionRecord]) -> list[dict[str, str]]:
    return [{"option_id": option.option_id, "text": option.text} for option in options]


def peer_positions_from_cache(
    *,
    peer_answers: dict[str, list[str]],
    peer_reasoning: dict[str, str],
) -> list[dict[str, object]]:
    return [
        {
            "agent_id": agent_id,
            "selected_option_ids": list(option_ids),
            "reasoning": peer_reasoning.get(agent_id, ""),
        }
        for agent_id, option_ids in sorted(peer_answers.items())
    ]


def effective_support_reviews(
    prior_feedback: object,
    *,
    threshold: int = DEFAULT_SUPPORT_SCORE_THRESHOLD,
) -> list[dict[str, object]]:
    reviews = getattr(prior_feedback, "key_reviews", []) if prior_feedback is not None else []
    result: list[dict[str, object]] = []
    for review in reviews:
        if review.stance in {"support", "mixed"} and review.score >= threshold:
            result.append(review.to_dict(include_chain_of_thought=False))
    return result


def most_popular_non_reference_option(
    *,
    question_record: QuestionRecord,
    answers_by_agent: dict[str, AnswerSubmission],
    excluded_agent_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    excluded = excluded_agent_ids or set()
    reference = tuple(sorted(question_record.correct_option_ids))
    counts: dict[tuple[str, ...], list[str]] = {}
    valid_option_ids = set(question_record.option_ids())
    for agent_id, answer in answers_by_agent.items():
        if agent_id in excluded or not answer.selected_option_ids:
            continue
        key = tuple(sorted(answer.selected_option_ids))
        if key == reference:
            continue
        if not all(option_id in valid_option_ids for option_id in key):
            continue
        counts.setdefault(key, []).append(agent_id)
    if not counts:
        return [], []
    best_key = sorted(
        counts,
        key=lambda key: (-len(counts[key]), key),
    )[0]
    return list(best_key), sorted(counts[best_key])


def candidate_rerank_score(
    *,
    question_record: QuestionRecord,
    target_option_ids: list[str],
    reasoning: str,
    examples: list[FewShotExample],
    effective_support_count: int = 0,
    peer_alignment_count: int = 0,
) -> float:
    option_text = " ".join(
        option.text
        for option in question_record.options
        if option.option_id in set(target_option_ids)
    )
    text_score = _question_overlap(question_record.question, reasoning)
    option_score = _question_overlap(option_text, reasoning)
    example_score = sum(example.average_score for example in examples[:3]) / max(1, min(3, len(examples)))
    length_score = min(len(reasoning.split()) / 80.0, 1.0)
    return (
        text_score * 2.0
        + option_score
        + example_score
        + length_score
        + effective_support_count * 1.5
        + peer_alignment_count * 2.0
    )


def _question_overlap(left: str, right: str) -> float:
    left_words = set(_words(left))
    right_words = set(_words(right))
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]
