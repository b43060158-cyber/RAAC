from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from autogen_mas.answer_matching import (
    canonicalize_short_answer,
    explicit_short_answer_from_reasoning,
    short_answer_tie_key,
)
from autogen_mas.answer_consistency import (
    answer_reasoning_consistency_check,
    extract_reasoning_decision_option_ids,
)
from autogen_mas.chess_parsing import (
    canonicalize_chess_final_answer,
    chess_answer_tie_key,
    extract_chess_square,
)


class ValidationError(ValueError):
    """Raised when a model-backed submission fails schema validation."""


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


@dataclass(slots=True)
class OptionRecord:
    option_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QuestionRecord:
    question_id: str
    dataset_name: str
    task_type: str
    question: str
    options: list[OptionRecord]
    correct_option_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def question_key(self) -> str:
        return f"{self.dataset_name}__{self.question_id}__{self.task_type}"

    def option_ids(self) -> list[str]:
        return [option.option_id for option in self.options]

    def normalize_option_ids(self, selected_option_ids: list[str]) -> list[str]:
        valid_order = self.option_ids()
        selected = _dedupe_preserve_order(selected_option_ids)
        invalid = [option_id for option_id in selected if option_id not in valid_order]
        if invalid:
            raise ValidationError(f"Invalid option ids: {invalid}")
        return [option_id for option_id in valid_order if option_id in selected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "question": self.question,
            "options": [option.to_dict() for option in self.options],
            "correct_option_ids": list(self.correct_option_ids),
            "metadata": self.metadata,
            "question_key": self.question_key,
        }


@dataclass(slots=True)
class CodeQuestionRecord:
    question_id: str
    dataset_name: str
    prompt: str
    entry_point: str
    test: str
    source_path: str
    source_task_id: str
    task_type: str = "code_generation"

    @property
    def question_key(self) -> str:
        return f"{self.dataset_name}__{self.question_id}__{self.task_type}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "test": self.test,
            "source_path": self.source_path,
            "source_task_id": self.source_task_id,
            "question_key": self.question_key,
        }


@dataclass(slots=True)
class ShortAnswerQuestionRecord:
    question_id: str
    dataset_name: str
    task_type: str
    question: str
    acceptable_answers: list[str]
    adversarial_target_answers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def question_key(self) -> str:
        return f"{self.dataset_name}__{self.question_id}__{self.task_type}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "question": self.question,
            "acceptable_answers": list(self.acceptable_answers),
            "adversarial_target_answers": list(self.adversarial_target_answers),
            "metadata": self.metadata,
            "question_key": self.question_key,
        }


@dataclass(slots=True)
class ChessQuestionRecord:
    question_id: str
    dataset_name: str
    game: str
    source_square: str
    legal_target_squares: list[str]
    rendered_question: str
    metadata: dict[str, Any] = field(default_factory=dict)
    task_type: str = "chess_move"

    @property
    def question(self) -> str:
        return self.rendered_question

    @property
    def question_key(self) -> str:
        return f"{self.dataset_name}__{self.question_id}__{self.task_type}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "game": self.game,
            "source_square": self.source_square,
            "legal_target_squares": list(self.legal_target_squares),
            "rendered_question": self.rendered_question,
            "question": self.rendered_question,
            "metadata": self.metadata,
            "question_key": self.question_key,
        }


TaskRecord = QuestionRecord | CodeQuestionRecord | ShortAnswerQuestionRecord | ChessQuestionRecord


@dataclass(slots=True)
class AnswerSubmission:
    agent_id: str
    selected_option_ids: list[str]
    reasoning: str
    code: str = ""
    final_answer: str = ""
    raw_selected_option_ids: list[str] = field(default_factory=list)
    alignment_resolution: str = ""
    alignment_judge_votes: list[str] = field(default_factory=list)
    changed_answer: bool = False
    change_drivers: list[str] = field(default_factory=list)
    change_summary: str = ""
    chain_of_thought: str | None = None
    confidence: float | None = None

    def validate(
        self, question_record: TaskRecord, *, is_adversarial: bool = False
    ) -> "AnswerSubmission":
        self.code = self.code.strip("\r\n")
        self.final_answer = self.final_answer.strip()
        self.chain_of_thought = _clean_optional_text(self.chain_of_thought)
        self.alignment_resolution = self.alignment_resolution.strip()
        if isinstance(question_record, CodeQuestionRecord):
            if self.selected_option_ids:
                raise ValidationError(
                    f"Agent {self.agent_id} must not return selected_option_ids for code_generation."
                )
            if not self.code:
                raise ValidationError(f"Agent {self.agent_id} returned empty code.")
            normalized: list[str] = []
        elif isinstance(question_record, ChessQuestionRecord):
            if self.selected_option_ids:
                raise ValidationError(
                    f"Agent {self.agent_id} must not return selected_option_ids for "
                    f"{question_record.task_type}."
                )
            self.final_answer = canonicalize_chess_final_answer(
                self.final_answer,
                source_square=question_record.source_square,
            )
            if not self.final_answer:
                raise ValidationError(
                    f"Agent {self.agent_id} returned empty final_answer."
                )
            explicit_final_answer = extract_chess_square(
                self.reasoning,
                source_square=question_record.source_square,
            )
            if (
                explicit_final_answer
                and chess_answer_tie_key(
                    explicit_final_answer,
                    source_square=question_record.source_square,
                )
                != chess_answer_tie_key(
                    self.final_answer,
                    source_square=question_record.source_square,
                )
            ):
                self.final_answer = explicit_final_answer
            normalized = []
        elif isinstance(question_record, ShortAnswerQuestionRecord):
            if self.selected_option_ids:
                raise ValidationError(
                    f"Agent {self.agent_id} must not return selected_option_ids for "
                    f"{question_record.task_type}."
                )
            self.final_answer = canonicalize_short_answer(
                self.final_answer,
                metadata=question_record.metadata,
            )
            if not self.final_answer:
                raise ValidationError(
                    f"Agent {self.agent_id} returned empty final_answer."
                )
            explicit_final_answer = explicit_short_answer_from_reasoning(
                self.reasoning,
                metadata=question_record.metadata,
            )
            if (
                explicit_final_answer
                and short_answer_tie_key(
                    explicit_final_answer,
                    metadata=question_record.metadata,
                )
                != short_answer_tie_key(
                    self.final_answer,
                    metadata=question_record.metadata,
                )
            ):
                self.final_answer = explicit_final_answer
            normalized = []
        else:
            normalized = question_record.normalize_option_ids(self.selected_option_ids)
        if not self.reasoning.strip():
            raise ValidationError(f"Agent {self.agent_id} returned empty reasoning.")
        self.reasoning = self.reasoning.strip()
        alignment_judge_resolved = self.alignment_resolution.startswith(
            "llm_judge_"
        ) or self.alignment_resolution.startswith("judge_fallback_")
        if isinstance(question_record, QuestionRecord):
            reasoning_decision = extract_reasoning_decision_option_ids(
                question_record=question_record,
                reasoning=self.reasoning,
            )
            if (
                reasoning_decision
                and not alignment_judge_resolved
                and (
                    (
                        question_record.task_type == "single_choice"
                        and len(reasoning_decision) == 1
                    )
                    or question_record.task_type == "multiple_choice"
                )
            ):
                normalized = question_record.normalize_option_ids(reasoning_decision)
        if question_record.task_type == "single_choice" and len(normalized) != 1:
            raise ValidationError(
                f"Agent {self.agent_id} must choose exactly one option for single_choice."
            )
        if question_record.task_type == "multiple_choice" and not normalized:
            raise ValidationError(
                f"Agent {self.agent_id} must choose at least one option for multiple_choice."
            )
        self.selected_option_ids = normalized
        if isinstance(question_record, QuestionRecord) and normalized:
            if (
                not is_adversarial
                and not alignment_judge_resolved
                and not answer_reasoning_consistency_check(
                    question_record=question_record,
                    assigned_option_ids=normalized,
                    reasoning=self.reasoning,
                )
            ):
                raise ValidationError(
                    f"Agent {self.agent_id} reasoning must defend the selected option ids."
                )
        self.raw_selected_option_ids = (
            question_record.normalize_option_ids(self.raw_selected_option_ids)
            if isinstance(question_record, QuestionRecord) and self.raw_selected_option_ids
            else []
        )
        self.alignment_judge_votes = _dedupe_preserve_order(
            [vote.strip() for vote in self.alignment_judge_votes if vote.strip()]
        )
        if self.confidence is not None:
            self.confidence = float(self.confidence)
            if not 0.0 <= self.confidence <= 1.0:
                raise ValidationError(
                    f"Agent {self.agent_id} confidence must be between 0.0 and 1.0."
                )
        self.change_drivers = _dedupe_preserve_order(
            [driver.strip() for driver in self.change_drivers if driver.strip()]
        )
        if self.agent_id in self.change_drivers:
            raise ValidationError(f"Agent {self.agent_id} cannot list itself as a change driver.")
        self.change_summary = self.change_summary.strip()
        if self.changed_answer and not self.change_summary:
            raise ValidationError(
                f"Agent {self.agent_id} must provide change_summary when changed_answer is true."
            )
        return self

    def to_dict(self, *, include_chain_of_thought: bool = True) -> dict[str, Any]:
        payload = {
            "agent_id": self.agent_id,
            "selected_option_ids": list(self.selected_option_ids),
            "reasoning": self.reasoning,
            "code": self.code,
            "final_answer": self.final_answer,
            "changed_answer": self.changed_answer,
            "change_drivers": list(self.change_drivers),
            "change_summary": self.change_summary,
        }
        if self.raw_selected_option_ids:
            payload["raw_selected_option_ids"] = list(self.raw_selected_option_ids)
        if self.alignment_resolution:
            payload["alignment_resolution"] = self.alignment_resolution
        if self.alignment_judge_votes:
            payload["alignment_judge_votes"] = list(self.alignment_judge_votes)
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if include_chain_of_thought and self.chain_of_thought:
            payload["chain_of_thought"] = self.chain_of_thought
        return payload


@dataclass(slots=True)
class ReviewSubmission:
    reviewer_agent_id: str
    target_agent_id: str
    score: int
    stance: str
    main_reason: str
    chain_of_thought: str | None = None

    def validate(self) -> "ReviewSubmission":
        if self.reviewer_agent_id == self.target_agent_id:
            raise ValidationError("Agents cannot review themselves.")
        if not 1 <= self.score <= 10:
            raise ValidationError("Review score must be between 1 and 10.")
        if self.stance not in {"support", "oppose", "mixed"}:
            raise ValidationError(f"Invalid stance: {self.stance}")
        if not isinstance(self.main_reason, str) or not self.main_reason.strip():
            raise ValidationError("Review field 'main_reason' must be non-empty.")
        self.main_reason = self.main_reason.strip()
        self.chain_of_thought = _clean_optional_text(self.chain_of_thought)
        return self

    def to_dict(self, *, include_chain_of_thought: bool = True) -> dict[str, Any]:
        payload = {
            "reviewer_agent_id": self.reviewer_agent_id,
            "target_agent_id": self.target_agent_id,
            "score": self.score,
            "stance": self.stance,
            "main_reason": self.main_reason,
        }
        if include_chain_of_thought and self.chain_of_thought:
            payload["chain_of_thought"] = self.chain_of_thought
        return payload


@dataclass(slots=True)
class PriorRoundFeedback:
    previous_answer: AnswerSubmission
    key_reviews: list[ReviewSubmission]
    total_score: float
    average_score: float

    def to_dict(self) -> dict[str, Any]:
        support_count = sum(1 for review in self.key_reviews if review.stance == "support")
        oppose_count = sum(1 for review in self.key_reviews if review.stance == "oppose")
        mixed_count = sum(1 for review in self.key_reviews if review.stance == "mixed")
        return {
            "previous_answer": self.previous_answer.to_dict(
                include_chain_of_thought=False,
            ),
            "feedback_summary": {
                "total_score": self.total_score,
                "average_score": self.average_score,
                "support_count": support_count,
                "oppose_count": oppose_count,
                "mixed_count": mixed_count,
                "key_reviews": [
                    review.to_dict(include_chain_of_thought=False)
                    for review in self.key_reviews
                ],
            },
            "total_score": self.total_score,
            "average_score": self.average_score,
        }


@dataclass(slots=True)
class RoundAgentResult:
    agent_id: str
    answer: AnswerSubmission
    reviews_given: list[ReviewSubmission]
    received_reviews: list[ReviewSubmission]
    total_score: float
    average_score: float
    is_adversarial_agent: bool = False

    def to_prior_feedback(self) -> PriorRoundFeedback:
        return PriorRoundFeedback(
            previous_answer=self.answer,
            key_reviews=self.received_reviews,
            total_score=self.total_score,
            average_score=self.average_score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "answer": self.answer.to_dict(),
            "reviews_given": [review.to_dict() for review in self.reviews_given],
            "received_reviews": [review.to_dict() for review in self.received_reviews],
            "total_score": self.total_score,
            "average_score": self.average_score,
            "advers-agent": self.is_adversarial_agent,
        }


@dataclass(slots=True)
class QuestionRoundResult:
    round_index: int
    agent_results: list[RoundAgentResult]
    reached_consensus: bool = False

    def agent_map(self) -> dict[str, RoundAgentResult]:
        return {result.agent_id: result for result in self.agent_results}

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "reached_consensus": self.reached_consensus,
            "agent_results": [result.to_dict() for result in self.agent_results],
        }


@dataclass(slots=True)
class QuestionRunResult:
    run_id: str
    question_id: str
    dataset_name: str
    task_type: str
    rounds: list[QuestionRoundResult]
    question: str = ""
    options: list[OptionRecord] = field(default_factory=list)
    correct_option_ids: list[str] = field(default_factory=list)
    prompt: str = ""
    entry_point: str = ""
    test: str = ""
    source_path: str = ""
    source_task_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def question_key(self) -> str:
        return f"{self.dataset_name}__{self.question_id}__{self.task_type}"

    @classmethod
    def from_task(
        cls,
        *,
        run_id: str,
        task: TaskRecord,
        rounds: list[QuestionRoundResult],
    ) -> "QuestionRunResult":
        if isinstance(task, CodeQuestionRecord):
            return cls(
                run_id=run_id,
                question_id=task.question_id,
                dataset_name=task.dataset_name,
                task_type=task.task_type,
                rounds=rounds,
                prompt=task.prompt,
                entry_point=task.entry_point,
                test=task.test,
                source_path=task.source_path,
                source_task_id=task.source_task_id,
            )
        if isinstance(task, ChessQuestionRecord):
            return cls(
                run_id=run_id,
                question_id=task.question_id,
                dataset_name=task.dataset_name,
                task_type=task.task_type,
                rounds=rounds,
                question=task.rendered_question,
                metadata={
                    **task.metadata,
                    "game": task.game,
                    "source_square": task.source_square,
                    "legal_target_squares": list(task.legal_target_squares),
                    "rendered_question": task.rendered_question,
                },
            )
        if isinstance(task, ShortAnswerQuestionRecord):
            return cls(
                run_id=run_id,
                question_id=task.question_id,
                dataset_name=task.dataset_name,
                task_type=task.task_type,
                rounds=rounds,
                question=task.question,
                metadata={
                    **task.metadata,
                    "acceptable_answers": list(task.acceptable_answers),
                    "adversarial_target_answers": list(task.adversarial_target_answers),
                },
            )
        return cls(
            run_id=run_id,
            question_id=task.question_id,
            dataset_name=task.dataset_name,
            task_type=task.task_type,
            rounds=rounds,
            question=task.question,
            options=task.options,
            correct_option_ids=task.correct_option_ids,
            metadata=task.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "question_id": self.question_id,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "rounds": [round_result.to_dict() for round_result in self.rounds],
            "question_key": self.question_key,
        }
        if self.task_type == "code_generation":
            payload.update(
                {
                    "prompt": self.prompt,
                    "entry_point": self.entry_point,
                    "test": self.test,
                    "source_path": self.source_path,
                    "source_task_id": self.source_task_id,
                }
            )
        elif self.task_type == "chess_move":
            payload.update(
                {
                    "question": self.question,
                    "rendered_question": str(
                        self.metadata.get("rendered_question", self.question)
                    ),
                    "game": str(self.metadata.get("game", "")),
                    "source_square": str(self.metadata.get("source_square", "")),
                    "legal_target_squares": list(
                        self.metadata.get("legal_target_squares", [])
                    ),
                    "metadata": self.metadata,
                }
            )
        elif self.task_type == "math_short_answer":
            payload.update(
                {
                    "question": self.question,
                    "acceptable_answers": list(
                        self.metadata.get("acceptable_answers", [])
                    ),
                    "adversarial_target_answers": list(
                        self.metadata.get("adversarial_target_answers", [])
                    ),
                    "metadata": self.metadata,
                }
            )
        else:
            payload.update(
                {
                    "question": self.question,
                    "options": [option.to_dict() for option in self.options],
                    "correct_option_ids": list(self.correct_option_ids),
                    "metadata": self.metadata,
                }
            )
        return payload


@dataclass(slots=True)
class RunArtifacts:
    run_id: str
    run_path: str
    manifest_path: str
    question_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationQuestionResult:
    question_id: str
    question_key: str
    dataset_name: str
    task_type: str
    selected_agent_id: str
    predicted_option_ids: list[str]
    correct_option_ids: list[str]
    is_correct: bool
    selection_rule: str
    predicted_final_answer: str = ""
    acceptable_answers: list[str] = field(default_factory=list)
    selected_is_adversarial: bool = False
    is_tie: bool = False
    excluded_from_accuracy: bool = False
    tie_candidate_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationReport:
    run_path: str
    question_results: list[EvaluationQuestionResult]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_path": self.run_path,
            "question_results": [item.to_dict() for item in self.question_results],
            "summary": self.summary,
        }
