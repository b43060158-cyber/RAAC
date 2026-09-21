from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
from typing import Callable

from autogen_mas.config import AgentConfig, DashScopeSettings, ExperimentConfig
from autogen_mas.models import (
    AnswerSubmission,
    ChessQuestionRecord,
    QuestionRecord,
    QuestionRoundResult,
    QuestionRunResult,
    ReviewSubmission,
    RoundAgentResult,
    RunArtifacts,
    ShortAnswerQuestionRecord,
    TaskRecord,
    ValidationError,
)
from autogen_mas.persistence import JsonRunStore

from .agent import MasAgent, ModelBackedMasAgent, coerce_selected_option_ids
from .alignment_judge import resolve_choice_answer_alignment
from .batch_runner import (
    BatchAttemptTimeout,
    BatchFailure,
    BatchRuntimeOptions,
    _is_fatal_provider_error,
    _should_fallback_to_smaller_batches,
    estimate_tokens,
    write_batch_integrity_summary,
)
from .clients import build_structured_client
from .runner import QuestionProgressCallback


DEFAULT_SINGLE_AGENT_ANSWER_PROMPT = """You are an independent single-agent QA baseline.
Return only valid JSON.
Follow the response schema for the task exactly.
Use prior self-evaluation only as your own previous reasoning context.
There are no peer agents and no external change drivers.
Be concise but explicit in the reasoning.
"""


DEFAULT_SINGLE_AGENT_ANSWER_PROMPT_NO_REFLECTION = """You are an independent single-agent QA baseline.
Return only valid JSON.
Follow the response schema for the task exactly.
There are no peer agents, no self-evaluation rounds, and no external change drivers.
Be concise but explicit in the reasoning.
"""


MEDMCQA_SINGLE_AGENT_ANSWER_PROMPT = """You are an independent single-agent medical single-choice QA baseline.
Return only valid JSON.
Follow the response schema exactly and choose exactly one option id in selected_option_ids.
Even if the question stem says EXCEPT, NOT, LEAST, INCORRECT, FALSE, or all of the following, still output exactly one final option id.
Use prior self-evaluation only as your own previous reasoning context.
There are no peer agents and no external change drivers.
Be concise but explicit in the reasoning.
"""


MEDMCQA_SINGLE_AGENT_ANSWER_PROMPT_NO_REFLECTION = """You are an independent single-agent medical single-choice QA baseline.
Return only valid JSON.
Follow the response schema exactly and choose exactly one option id in selected_option_ids.
Even if the question stem says EXCEPT, NOT, LEAST, INCORRECT, FALSE, or all of the following, still output exactly one final option id.
There are no peer agents, no self-evaluation rounds, and no external change drivers.
Be concise but explicit in the reasoning.
"""


CHESS_SINGLE_AGENT_ANSWER_PROMPT = """You are an independent single-agent chess state-tracking baseline.
Return only valid JSON.
Follow the response schema exactly.
For chess tasks, reason from the provided move history and the current source square, then return exactly one legal destination square in final_answer using the form [a-h][1-8].
Use prior self-evaluation only as your own previous reasoning context.
There are no peer agents and no external change drivers.
Keep the reasoning concise and ensure it supports the final square exactly.
"""


CHESS_SINGLE_AGENT_ANSWER_PROMPT_NO_REFLECTION = """You are an independent single-agent chess state-tracking baseline.
Return only valid JSON.
Follow the response schema exactly.
For chess tasks, reason from the provided move history and the current source square, then return exactly one legal destination square in final_answer using the form [a-h][1-8].
There are no peer agents, no self-evaluation rounds, and no external change drivers.
Keep the reasoning concise and ensure it supports the final square exactly.
"""


FAIREVAL_SINGLE_AGENT_ANSWER_PROMPT = """You are an independent single-agent response-comparison baseline.
Return only valid JSON.
Follow the response schema exactly and choose exactly one option id in selected_option_ids.
This task is not a general knowledge QA question. Instead, compare the two candidate responses shown in the question context and judge their overall quality.
Choose A if response 1 is better, choose B if the two responses are equally good overall, and choose C if response 2 is better.
If the two responses are hard to distinguish overall, choose B.
Use prior self-evaluation only as your own previous reasoning context.
There are no peer agents and no external change drivers.
Be concise but explicit in the reasoning.
"""


FAIREVAL_SINGLE_AGENT_ANSWER_PROMPT_NO_REFLECTION = """You are an independent single-agent response-comparison baseline.
Return only valid JSON.
Follow the response schema exactly and choose exactly one option id in selected_option_ids.
This task is not a general knowledge QA question. Instead, compare the two candidate responses shown in the question context and judge their overall quality.
Choose A if response 1 is better, choose B if the two responses are equally good overall, and choose C if response 2 is better.
If the two responses are hard to distinguish overall, choose B.
There are no peer agents, no self-evaluation rounds, and no external change drivers.
Be concise but explicit in the reasoning.
"""


def _default_single_agent_answer_prompt(
    *,
    dataset_name: str,
    enable_self_reflection: bool,
) -> str:
    if dataset_name == "chess":
        return (
            CHESS_SINGLE_AGENT_ANSWER_PROMPT
            if enable_self_reflection
            else CHESS_SINGLE_AGENT_ANSWER_PROMPT_NO_REFLECTION
        )
    if dataset_name == "medmcqa":
        return (
            MEDMCQA_SINGLE_AGENT_ANSWER_PROMPT
            if enable_self_reflection
            else MEDMCQA_SINGLE_AGENT_ANSWER_PROMPT_NO_REFLECTION
        )
    if dataset_name == "faireval":
        return (
            FAIREVAL_SINGLE_AGENT_ANSWER_PROMPT
            if enable_self_reflection
            else FAIREVAL_SINGLE_AGENT_ANSWER_PROMPT_NO_REFLECTION
        )
    return (
        DEFAULT_SINGLE_AGENT_ANSWER_PROMPT
        if enable_self_reflection
        else DEFAULT_SINGLE_AGENT_ANSWER_PROMPT_NO_REFLECTION
    )


SingleAgentFactory = Callable[[ExperimentConfig, DashScopeSettings], MasAgent]


@dataclass(slots=True)
class _SingleAgentPriorFeedback:
    previous_answer: AnswerSubmission
    self_review: ReviewSubmission

    def to_dict(self) -> dict[str, object]:
        return {
            "previous_answer": self.previous_answer.to_dict(),
            "self_evaluation": {
                "score": self.self_review.score,
                "stance": self.self_review.stance,
                "main_reason": self.self_review.main_reason,
            },
        }


def _question_payload(question_record: TaskRecord) -> dict[str, object]:
    if isinstance(question_record, ChessQuestionRecord):
        return {
            "question_id": question_record.question_id,
            "question_key": question_record.question_key,
            "dataset_name": question_record.dataset_name,
            "task_type": question_record.task_type,
            "game": question_record.game,
            "source_square": question_record.source_square,
            "legal_target_squares": list(question_record.legal_target_squares),
            "question": question_record.rendered_question,
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
    if question_record.dataset_name == "faireval":
        response_1_text = question_record.metadata.get("response_1_text")
        response_2_text = question_record.metadata.get("response_2_text")
        if isinstance(response_1_text, str) and isinstance(response_2_text, str):
            payload["candidate_responses"] = {
                "response_1": response_1_text,
                "response_2": response_2_text,
            }
    return payload


def _build_self_review_user_prompt(
    *,
    question_record: QuestionRecord,
    round_index: int,
    answer: AnswerSubmission,
) -> str:
    payload = {
        "stage": "single_agent_self_review",
        "round_index": round_index,
        "question_context": _question_payload(question_record),
        "submitted_answer": answer.to_dict(),
        "response_schema": {
            "score": "integer 1-10",
            "stance": "support | oppose | mixed",
            "main_reason": "one concise self-evaluation reason for the score and stance",
        },
        "constraints": [
            "Return valid JSON only.",
            "Evaluate your own submitted answer honestly before the next round.",
            "Score must be an integer from 1 to 10.",
            "stance must be support, oppose, or mixed.",
            "main_reason must be non-empty.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_single_answer_user_prompt(
    *,
    question_record: QuestionRecord,
    round_index: int,
    prior_feedback: _SingleAgentPriorFeedback | None,
) -> str:
    payload: dict[str, object] = {
        "stage": "single_agent_answer",
        "round_index": round_index,
        "question_context": _question_payload(question_record),
        "prior_self_feedback": None if prior_feedback is None else prior_feedback.to_dict(),
        "response_schema": (
            {
                "final_answer": "short final answer",
                "reasoning": "brief but concrete explanation for the current answer",
                "changed_answer": "boolean",
                "change_summary": "brief explanation of what changed and why",
            }
            if isinstance(question_record, (ShortAnswerQuestionRecord, ChessQuestionRecord))
            else {
                "selected_option_ids": ["list of option ids"],
                "reasoning": "brief but concrete explanation",
                "changed_answer": "boolean",
                "change_summary": "brief explanation of what changed and why",
            }
        ),
        "constraints": (
            [
                "Return valid JSON only.",
                "final_answer must be non-empty.",
                "reasoning must explain the current final_answer.",
                "If reasoning states an explicit final answer, final_answer must match it exactly.",
                "Do not return selected_option_ids.",
                "For round 1, changed_answer should be false.",
                "Do not include change_drivers; this is a single-agent baseline with no peer agents.",
                "If changed_answer is true, change_summary must explain the self-correction.",
            ]
            if isinstance(question_record, (ShortAnswerQuestionRecord, ChessQuestionRecord))
            else [
                "Return valid JSON only.",
                "selected_option_ids must use only the provided option ids.",
                "Return exactly one option id in selected_option_ids.",
                "For single_choice EXCEPT/all-of-the-following style questions, still return exactly one final option id rather than listing every true statement.",
                "reasoning must justify the current selected_option_ids.",
                "Do not claim that any unselected option is correct, preferable, or the final answer.",
                "If reasoning explicitly names the correct answer or chosen option ids, they must match selected_option_ids exactly.",
                "For round 1, changed_answer should be false.",
                "Do not include change_drivers; this is a single-agent baseline with no peer agents.",
                "If changed_answer is true, change_summary must explain the self-correction.",
            ]
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_batch_single_answer_user_prompt(
    *,
    question_records: list[QuestionRecord],
    round_index: int,
    prior_feedback_by_question: dict[str, _SingleAgentPriorFeedback | None],
) -> str:
    payload: dict[str, object] = {
        "stage": "single_agent_batch_answer",
        "round_index": round_index,
        "question_contexts": [_question_payload(question) for question in question_records],
        "prior_self_feedback_by_question": {
            question.question_key: (
                None
                if prior_feedback_by_question.get(question.question_key) is None
                else prior_feedback_by_question[question.question_key].to_dict()
            )
            for question in question_records
        },
        "response_schema": {
            "answers": [
                {
                    "question_key": "question_key from question_contexts",
                    "selected_option_ids": ["list of option ids"],
                    "final_answer": "short final answer for math_short_answer tasks only",
                    "reasoning": "brief but concrete explanation",
                    "changed_answer": "boolean",
                    "change_summary": "brief explanation of what changed and why",
                }
            ]
        },
        "constraints": [
            "Return valid JSON only.",
            "Return exactly one answer for each question_context.",
            "question_key must match one of the provided question_context question_key values.",
            "selected_option_ids must use only the provided option ids for that question.",
            "Return exactly one option id in selected_option_ids.",
            "For single_choice EXCEPT/all-of-the-following style questions, still return exactly one final option id rather than listing every true statement.",
            "For choice tasks, reasoning must justify the current selected_option_ids.",
            "For choice tasks, do not claim that any unselected option is correct, preferable, or the final answer.",
            "For choice tasks, if reasoning explicitly names the correct answer or chosen option ids, they must match selected_option_ids exactly.",
            "If task_type is math_short_answer or chess_move, return non-empty final_answer and no selected_option_ids.",
            "For math_short_answer or chess_move, if reasoning states an explicit final answer, final_answer must match it exactly.",
            "For round 1, changed_answer should be false.",
            "Do not include change_drivers; this is a single-agent baseline with no peer agents.",
            "If changed_answer is true, change_summary must explain the self-correction.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_batch_self_review_user_prompt(
    *,
    question_records: list[QuestionRecord],
    round_index: int,
    answers_by_question: dict[str, AnswerSubmission],
) -> str:
    payload = {
        "stage": "single_agent_batch_self_review",
        "round_index": round_index,
        "question_contexts": [_question_payload(question) for question in question_records],
        "submitted_answers_by_question": {
            question.question_key: answers_by_question[question.question_key].to_dict()
            for question in question_records
        },
        "response_schema": {
            "reviews": [
                {
                    "question_key": "question_key from question_contexts",
                    "score": "integer 1-10",
                    "stance": "support | oppose | mixed",
                    "main_reason": "one concise self-evaluation reason for the score and stance",
                }
            ]
        },
        "constraints": [
            "Return valid JSON only.",
            "Return exactly one review for each question_context.",
            "question_key must match one of the provided question_context question_key values.",
            "Evaluate each submitted answer honestly before the next round.",
            "Score must be an integer from 1 to 10.",
            "stance must be support, oppose, or mixed.",
            "main_reason must be non-empty.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _validate_self_review(review: ReviewSubmission) -> ReviewSubmission:
    if not 1 <= review.score <= 10:
        raise ValidationError("Self-review score must be between 1 and 10.")
    if review.stance not in {"support", "oppose", "mixed"}:
        raise ValidationError(f"Invalid self-review stance: {review.stance}")
    if not isinstance(review.main_reason, str) or not review.main_reason.strip():
        raise ValidationError("Self-review field 'main_reason' must be non-empty.")
    review.main_reason = review.main_reason.strip()
    return review


def _parse_batch_self_reviews(
    *,
    agent_id: str,
    questions: list[QuestionRecord],
    raw_reviews: object,
) -> dict[str, ReviewSubmission]:
    if not isinstance(raw_reviews, list):
        raise ValidationError("Batch self-review response must contain a reviews list.")
    expected_keys = [question.question_key for question in questions]
    expected_key_set = set(expected_keys)
    reviews_by_question: dict[str, ReviewSubmission] = {}
    for raw_review in raw_reviews:
        if not isinstance(raw_review, dict):
            raise ValidationError("Each batch self-review item must be an object.")
        question_key = str(raw_review.get("question_key", ""))
        if question_key not in expected_key_set:
            raise ValidationError(f"Unexpected batch self-review question_key: {question_key}")
        if question_key in reviews_by_question:
            raise ValidationError(f"Duplicate batch self-review question_key: {question_key}")
        reviews_by_question[question_key] = _validate_self_review(
            ReviewSubmission(
                reviewer_agent_id=agent_id,
                target_agent_id=agent_id,
                score=int(raw_review.get("score", 0)),
                stance=str(raw_review.get("stance", "")),
                main_reason=str(raw_review.get("main_reason", "")),
            )
        )
    missing_keys = [key for key in expected_keys if key not in reviews_by_question]
    if missing_keys:
        raise ValidationError(f"Missing batch self-review question_keys: {missing_keys}")
    return {
        question.question_key: reviews_by_question[question.question_key]
        for question in questions
    }


class SingleAgentModelBackedAgent(MasAgent):
    def __init__(
        self,
        agent_config: AgentConfig,
        experiment_config: ExperimentConfig,
        client,
        default_model: str,
    ) -> None:
        super().__init__(agent_config)
        self.experiment_config = experiment_config
        self.client = client
        self.default_model = default_model

    @property
    def model(self) -> str:
        return self.agent_config.model or self.default_model

    def answer_system_prompt_for_dataset(self, dataset_name: str) -> str:
        default_prompt = _default_single_agent_answer_prompt(
            dataset_name=dataset_name,
            enable_self_reflection=self.experiment_config.single_agent.enable_self_reflection,
        )
        return self.experiment_config.single_agent.resolved_answer_prompt_for_dataset(
            dataset_name,
            default_prompt,
        )

    def answer(
        self,
        question_context: QuestionRecord,
        prior_feedback: _SingleAgentPriorFeedback | None,
        round_index: int,
    ) -> AnswerSubmission:
        response = self.client.generate_json(
            system_prompt=self.answer_system_prompt_for_dataset(
                question_context.dataset_name
            ),
            user_prompt=_build_single_answer_user_prompt(
                question_record=question_context,
                round_index=round_index,
                prior_feedback=prior_feedback,
            ),
            model=self.model,
            temperature=self.agent_config.temperature,
        )
        return self._parse_answer(question=question_context, raw_answer=response)

    def answer_many(
        self,
        question_contexts: list[QuestionRecord],
        prior_feedback_by_question: dict[str, _SingleAgentPriorFeedback | None],
        round_index: int,
    ) -> dict[str, AnswerSubmission]:
        if not question_contexts:
            return {}
        dataset_names = {question.dataset_name for question in question_contexts}
        if len(dataset_names) != 1:
            raise ValidationError(
                "Single-agent batch answers require a single dataset so the system "
                "prompt remains well-defined."
            )
        dataset_name = question_contexts[0].dataset_name
        response = self.client.generate_json(
            system_prompt=self.answer_system_prompt_for_dataset(dataset_name),
            user_prompt=_build_batch_single_answer_user_prompt(
                question_records=question_contexts,
                round_index=round_index,
                prior_feedback_by_question=prior_feedback_by_question,
            ),
            model=self.model,
            temperature=self.agent_config.temperature,
        )
        raw_answers = response.get("answers", [])
        if not isinstance(raw_answers, list):
            raise ValidationError("Single-agent batch answer response must contain an answers list.")

        questions_by_key = {question.question_key: question for question in question_contexts}
        answers_by_key: dict[str, AnswerSubmission] = {}
        for raw_answer in raw_answers:
            if not isinstance(raw_answer, dict):
                raise ValidationError("Each single-agent batch answer item must be an object.")
            question_key = str(raw_answer.get("question_key", ""))
            if question_key not in questions_by_key:
                raise ValidationError(
                    f"Unexpected single-agent batch answer question_key: {question_key}"
                )
            if question_key in answers_by_key:
                raise ValidationError(
                    f"Duplicate single-agent batch answer question_key: {question_key}"
                )
            answers_by_key[question_key] = self._parse_answer(
                question=questions_by_key[question_key],
                raw_answer=raw_answer,
            )

        missing_keys = [
            question.question_key
            for question in question_contexts
            if question.question_key not in answers_by_key
        ]
        if missing_keys:
            raise ValidationError(f"Missing single-agent batch answer question_keys: {missing_keys}")
        return {
            question.question_key: answers_by_key[question.question_key]
            for question in question_contexts
        }

    def review(
        self,
        question_context: QuestionRecord,
        peer_submission: AnswerSubmission,
        reviewer_answer: AnswerSubmission,
        round_index: int,
    ) -> ReviewSubmission:
        return self.self_review(
            question_context=question_context,
            answer=reviewer_answer,
            round_index=round_index,
        )

    def self_review(
        self,
        *,
        question_context: QuestionRecord,
        answer: AnswerSubmission,
        round_index: int,
    ) -> ReviewSubmission:
        response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_review_prompt(
                self.experiment_config.review_prompt
            ),
            user_prompt=_build_self_review_user_prompt(
                question_record=question_context,
                round_index=round_index,
                answer=answer,
            ),
            model=self.model,
            temperature=self.agent_config.temperature,
        )
        return _validate_self_review(
            ReviewSubmission(
                reviewer_agent_id=self.agent_id,
                target_agent_id=self.agent_id,
                score=int(response.get("score", 0)),
                stance=str(response.get("stance", "")),
                main_reason=str(response.get("main_reason", "")),
            )
        )

    def self_review_many(
        self,
        *,
        questions: list[QuestionRecord],
        answers_by_question: dict[str, AnswerSubmission],
        round_index: int,
    ) -> dict[str, ReviewSubmission]:
        response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_review_prompt(
                self.experiment_config.review_prompt
            ),
            user_prompt=_build_batch_self_review_user_prompt(
                question_records=questions,
                round_index=round_index,
                answers_by_question=answers_by_question,
            ),
            model=self.model,
            temperature=self.agent_config.temperature,
        )
        return _parse_batch_self_reviews(
            agent_id=self.agent_id,
            questions=questions,
            raw_reviews=response.get("reviews", []),
        )

    def _parse_answer(self, *, question: TaskRecord, raw_answer: dict) -> AnswerSubmission:
        reasoning = str(raw_answer.get("reasoning", ""))
        final_answer = str(raw_answer.get("final_answer", ""))
        submission = AnswerSubmission(
            agent_id=self.agent_id,
            selected_option_ids=coerce_selected_option_ids(
                raw_answer.get("selected_option_ids", []),
                question_context=question,
                reasoning=reasoning,
                final_answer=final_answer,
            ),
            reasoning=reasoning,
            final_answer=final_answer,
            changed_answer=bool(raw_answer.get("changed_answer", False)),
            change_drivers=[],
            change_summary=str(raw_answer.get("change_summary", "")),
        )
        if isinstance(question, QuestionRecord):
            alignment = resolve_choice_answer_alignment(
                question_record=question,
                submitted_option_ids=list(submission.selected_option_ids),
                reasoning=submission.reasoning,
                alignment_judge_config=self.experiment_config.alignment_judge,
                client=self.client,
                default_model=self.default_model,
            )
            submission.raw_selected_option_ids = alignment.raw_selected_option_ids
            submission.selected_option_ids = alignment.selected_option_ids
            submission.alignment_resolution = alignment.resolution
            submission.alignment_judge_votes = alignment.judge_votes
        return submission.validate(question)


@dataclass(slots=True)
class SingleAgentRunner:
    experiment_config: ExperimentConfig
    llm_settings: DashScopeSettings
    store: JsonRunStore | None = None
    agent_factory: SingleAgentFactory | None = None

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = JsonRunStore(self.experiment_config.runtime.output_dir)

    def _build_agent(self, experiment_config: ExperimentConfig | None = None) -> MasAgent:
        config = experiment_config or self.experiment_config
        if self.agent_factory is not None:
            return self.agent_factory(config, self.llm_settings)
        client = build_structured_client(self.llm_settings)
        return SingleAgentModelBackedAgent(
            agent_config=config.single_agent.to_agent_config(),
            experiment_config=config,
            client=client,
            default_model=self.llm_settings.model,
        )

    @staticmethod
    def _self_reflection_enabled(config: ExperimentConfig) -> bool:
        return config.single_agent.enable_self_reflection

    @staticmethod
    def _build_round_agent_result(
        *,
        agent_id: str,
        answer: AnswerSubmission,
        self_review: ReviewSubmission | None,
    ) -> RoundAgentResult:
        if self_review is None:
            return RoundAgentResult(
                agent_id=agent_id,
                answer=answer,
                reviews_given=[],
                received_reviews=[],
                total_score=0.0,
                average_score=0.0,
            )
        total_score = float(self_review.score)
        return RoundAgentResult(
            agent_id=agent_id,
            answer=answer,
            reviews_given=[self_review],
            received_reviews=[self_review],
            total_score=total_score,
            average_score=total_score,
        )

    def run_question(
        self,
        question_record: QuestionRecord,
        *,
        agent: MasAgent | None = None,
        experiment_config: ExperimentConfig | None = None,
        run_id: str | None = None,
        progress_callback: QuestionProgressCallback | None = None,
        question_index: int = 1,
        total_questions: int = 1,
    ) -> QuestionRunResult:
        config = experiment_config or self.experiment_config
        baseline_agent = agent or self._build_agent(config)
        run_identifier = run_id or self._default_run_id(config)
        prior_feedback = None
        rounds: list[QuestionRoundResult] = []
        total_rounds = config.runtime.num_rounds
        enable_self_reflection = self._self_reflection_enabled(config)
        for round_index in range(1, total_rounds + 1):
            if progress_callback is not None:
                progress_callback(
                    question_index,
                    total_questions,
                    question_record,
                    f"round {round_index}/{total_rounds} answer started",
                )
            answer = baseline_agent.answer(
                question_context=question_record,
                prior_feedback=prior_feedback,
                round_index=round_index,
            )
            if progress_callback is not None:
                progress_callback(
                    question_index,
                    total_questions,
                    question_record,
                    f"round {round_index}/{total_rounds} answer completed",
                )
            self_review: ReviewSubmission | None = None
            if enable_self_reflection:
                if progress_callback is not None:
                    progress_callback(
                        question_index,
                        total_questions,
                        question_record,
                        f"round {round_index}/{total_rounds} self-review started",
                    )
                self_review = self._collect_self_review(
                    agent=baseline_agent,
                    question_context=question_record,
                    answer=answer,
                    round_index=round_index,
                    experiment_config=config,
                )
                if progress_callback is not None:
                    progress_callback(
                        question_index,
                        total_questions,
                        question_record,
                        f"round {round_index}/{total_rounds} self-review completed",
                    )
            agent_result = self._build_round_agent_result(
                agent_id=baseline_agent.agent_id,
                answer=answer,
                self_review=self_review,
            )
            rounds.append(
                QuestionRoundResult(round_index=round_index, agent_results=[agent_result])
            )
            if self_review is not None:
                prior_feedback = _SingleAgentPriorFeedback(
                    previous_answer=answer,
                    self_review=self_review,
                )
        return QuestionRunResult.from_task(
            run_id=run_identifier,
            task=question_record,
            rounds=rounds,
        )

    def run_dataset(
        self,
        dataset: list[QuestionRecord],
        experiment_config: ExperimentConfig | None = None,
        progress_callback: QuestionProgressCallback | None = None,
        resume_run_path: str | Path | None = None,
    ) -> RunArtifacts:
        config = experiment_config or self.experiment_config
        assert self.store is not None
        run_path, run_id = self._resolve_run_state(
            config=config,
            resume_run_path=resume_run_path,
        )
        agent = self._build_agent(config)
        question_paths: list[str] = []
        failure_records = self._load_failure_records(run_path)
        total_questions = len(dataset)
        for index, question_record in enumerate(dataset, start=1):
            if resume_run_path is not None:
                question_path = self.store.question_result_path(
                    run_path,
                    question_record.question_key,
                )
                payload = self.store.read_question_payload(
                    run_path,
                    question_record.question_key,
                )
                if self._is_complete_question_payload(
                    payload=payload,
                    question_record=question_record,
                    num_rounds=config.runtime.num_rounds,
                    expect_self_reflection=self._self_reflection_enabled(config),
                ):
                    question_paths.append(str(question_path))
                    if self._remove_failure_record(
                        failure_records,
                        question_key=question_record.question_key,
                    ):
                        self._write_failed_question_records(run_path, failure_records.values())
                    if progress_callback is not None:
                        progress_callback(index, total_questions, question_record, "completed")
                    continue
            if progress_callback is not None:
                progress_callback(index, total_questions, question_record, "started")
            try:
                result = self.run_question(
                    question_record,
                    agent=agent,
                    experiment_config=config,
                    run_id=run_id,
                    progress_callback=progress_callback,
                    question_index=index,
                    total_questions=total_questions,
                )
            except KeyboardInterrupt:
                raise
            except BaseException as error:
                failure_records[question_record.question_key] = self._updated_failure_record(
                    existing=failure_records.get(question_record.question_key),
                    question_index=index,
                    question_id=question_record.question_id,
                    question_key=question_record.question_key,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                self._write_failed_question_records(run_path, failure_records.values())
                if progress_callback is not None:
                    progress_callback(
                        index,
                        total_questions,
                        question_record,
                        f"failed: {type(error).__name__}",
                    )
                continue
            question_paths.append(self.store.persist_question_result(run_path, result))
            if self._remove_failure_record(
                failure_records,
                question_key=question_record.question_key,
            ):
                self._write_failed_question_records(run_path, failure_records.values())
            if progress_callback is not None:
                progress_callback(index, total_questions, question_record, "completed")
        self._write_failed_question_records(run_path, failure_records.values())
        return RunArtifacts(
            run_id=run_id,
            run_path=str(run_path),
            manifest_path=str(run_path / "manifest.json"),
            question_paths=question_paths,
        )

    def _resolve_run_state(
        self,
        *,
        config: ExperimentConfig,
        resume_run_path: str | Path | None,
    ) -> tuple[Path, str]:
        assert self.store is not None
        if resume_run_path is None:
            run_id = self._default_run_id(config)
            run_path = self.store.initialize_run(
                run_id=run_id,
                experiment_config=config,
                llm_settings=self.llm_settings,
            )
            return run_path, run_id

        run_path = Path(resume_run_path)
        manifest = self.store.read_manifest(run_path)
        if manifest is None:
            raise ValueError(f"Cannot resume run without a readable manifest: {run_path}")
        run_id = str(manifest.get("run_id") or run_path.name)
        return run_path, run_id

    def _is_complete_question_payload(
        self,
        *,
        payload: dict | None,
        question_record: QuestionRecord,
        num_rounds: int,
        expect_self_reflection: bool,
    ) -> bool:
        if payload is None:
            return False
        if str(payload.get("question_key", "")) != question_record.question_key:
            return False
        rounds = payload.get("rounds")
        if not isinstance(rounds, list) or len(rounds) != num_rounds:
            return False
        for round_data in rounds:
            if not isinstance(round_data, dict):
                return False
            agent_results = round_data.get("agent_results")
            if not isinstance(agent_results, list) or len(agent_results) != 1:
                return False
            agent_result = agent_results[0]
            if not isinstance(agent_result, dict):
                return False
            if not self._has_complete_answer(agent_result):
                return False
            agent_id = str(agent_result.get("agent_id", "")).strip()
            if not agent_id:
                return False
            if not expect_self_reflection:
                reviews_given = agent_result.get("reviews_given")
                received_reviews = agent_result.get("received_reviews")
                if reviews_given not in ([], None) or received_reviews not in ([], None):
                    return False
                continue
            expected_self = {agent_id}
            if not self._reviews_cover(
                agent_result.get("reviews_given"),
                id_field="target_agent_id",
                expected_ids=expected_self,
            ):
                return False
            if not self._reviews_cover(
                agent_result.get("received_reviews"),
                id_field="reviewer_agent_id",
                expected_ids=expected_self,
            ):
                return False
        return True

    def _has_complete_answer(self, agent_result: dict) -> bool:
        answer = agent_result.get("answer")
        if not isinstance(answer, dict):
            return False
        selected_option_ids = answer.get("selected_option_ids")
        if isinstance(selected_option_ids, list) and selected_option_ids:
            return True
        final_answer = answer.get("final_answer")
        if isinstance(final_answer, str) and final_answer.strip():
            return True
        code = answer.get("code")
        return isinstance(code, str) and bool(code.strip())

    def _reviews_cover(
        self,
        reviews: object,
        *,
        id_field: str,
        expected_ids: set[str],
    ) -> bool:
        if not isinstance(reviews, list) or len(reviews) != len(expected_ids):
            return False
        actual_ids: list[str] = []
        for review in reviews:
            if not isinstance(review, dict):
                return False
            actual_ids.append(str(review.get(id_field, "")))
        return set(actual_ids) == expected_ids

    def _default_run_id(self, experiment_config: ExperimentConfig | None = None) -> str:
        config = experiment_config or self.experiment_config
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{config.single_agent.run_id_prefix}-{timestamp}"

    def _collect_self_review(
        self,
        *,
        agent: MasAgent,
        question_context: QuestionRecord,
        answer: AnswerSubmission,
        round_index: int,
        experiment_config: ExperimentConfig,
    ) -> ReviewSubmission:
        if isinstance(agent, SingleAgentModelBackedAgent):
            return agent.self_review(
                question_context=question_context,
                answer=answer,
                round_index=round_index,
            )
        if isinstance(agent, ModelBackedMasAgent):
            response = agent.client.generate_json(
                system_prompt=agent.agent_config.resolved_review_prompt(
                    experiment_config.review_prompt
                ),
                user_prompt=_build_self_review_user_prompt(
                    question_record=question_context,
                    round_index=round_index,
                    answer=answer,
                ),
                model=agent.agent_config.model or agent.default_model,
                temperature=agent.agent_config.temperature,
            )
            return _validate_self_review(
                ReviewSubmission(
                    reviewer_agent_id=agent.agent_id,
                    target_agent_id=agent.agent_id,
                    score=int(response.get("score", 0)),
                    stance=str(response.get("stance", "")),
                    main_reason=str(response.get("main_reason", "")),
                )
            )

        return _validate_self_review(
            agent.review(
                question_context=question_context,
                peer_submission=answer,
                reviewer_answer=answer,
                round_index=round_index,
            )
        )

    def _write_failed_question_records(
        self,
        run_path: str | Path,
        failures,
    ) -> None:
        assert self.store is not None
        self.store.persist_failed_questions(
            run_path,
            [dict(failure) for failure in failures],
        )

    def _load_failure_records(
        self,
        run_path: str | Path,
    ) -> dict[str, dict[str, object]]:
        assert self.store is not None
        payload = self.store.read_failed_questions(run_path)
        if not payload:
            return {}
        records: dict[str, dict[str, object]] = {}
        for item in payload:
            question_key = str(item.get("question_key", "")).strip()
            if not question_key:
                continue
            records[question_key] = dict(item)
        return records

    def _updated_failure_record(
        self,
        *,
        existing: dict[str, object] | None,
        question_index: int,
        question_id: str,
        question_key: str,
        error_type: str,
        error_message: str,
    ) -> dict[str, object]:
        record = dict(existing or {})
        record["question_index"] = record.get("question_index", question_index)
        record["question_id"] = record.get("question_id", question_id)
        record["question_key"] = question_key
        record["error_type"] = error_type
        record["error_message"] = error_message
        return record

    def _remove_failure_record(
        self,
        failure_records: dict[str, dict[str, object]],
        *,
        question_key: str,
    ) -> bool:
        if question_key not in failure_records:
            return False
        del failure_records[question_key]
        return True


@dataclass(slots=True)
class BatchSingleAgentRunner(SingleAgentRunner):
    batch_options: BatchRuntimeOptions = field(default_factory=BatchRuntimeOptions)
    _active_run_path: Path | None = field(default=None, init=False, repr=False)
    _event_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def run_dataset(
        self,
        dataset: list[QuestionRecord],
        experiment_config: ExperimentConfig | None = None,
        progress_callback: QuestionProgressCallback | None = None,
        resume_run_path: str | Path | None = None,
    ) -> RunArtifacts:
        config = experiment_config or self.experiment_config
        assert self.store is not None
        run_path, run_id = self._resolve_run_state(
            config=config,
            resume_run_path=resume_run_path,
        )
        self._active_run_path = Path(run_path)
        if not dataset:
            self._write_failed_questions([])
            write_batch_integrity_summary(run_path, [])
            return RunArtifacts(
                run_id=run_id,
                run_path=str(run_path),
                manifest_path=str(run_path / "manifest.json"),
                question_paths=[],
            )

        agent = self._build_agent(config)
        total_questions = len(dataset)
        question_index = {
            question.question_key: index
            for index, question in enumerate(dataset, 1)
        }
        rounds_by_question: dict[str, list[QuestionRoundResult]] = {
            question.question_key: [] for question in dataset
        }
        prior_feedback_by_question: dict[str, _SingleAgentPriorFeedback | None] = {
            question.question_key: None for question in dataset
        }
        failed_question_keys: set[str] = set()
        persisted_question_keys: set[str] = set()
        failures: list[BatchFailure] = []
        question_paths: list[str] = []
        if resume_run_path is not None:
            for question in dataset:
                payload = self.store.read_question_payload(run_path, question.question_key)
                if not self._is_complete_question_payload(
                    payload=payload,
                    question_record=question,
                    num_rounds=config.runtime.num_rounds,
                    expect_self_reflection=self._self_reflection_enabled(config),
                ):
                    continue
                persisted_question_keys.add(question.question_key)
                question_paths.append(
                    str(self.store.question_result_path(run_path, question.question_key))
                )
                if progress_callback is not None:
                    progress_callback(
                        question_index[question.question_key],
                        total_questions,
                        question,
                        "completed",
                    )

        for round_index in range(1, config.runtime.num_rounds + 1):
            active_questions = [
                question
                for question in dataset
                if question.question_key not in failed_question_keys
                and question.question_key not in persisted_question_keys
            ]
            if not active_questions:
                break

            answers_by_question, answer_failures = self._collect_batch_answers(
                agent=agent,
                questions=active_questions,
                prior_feedback_by_question=prior_feedback_by_question,
                round_index=round_index,
                progress_callback=progress_callback,
                question_index=question_index,
                total_questions=total_questions,
                total_rounds=config.runtime.num_rounds,
            )
            failures.extend(answer_failures)
            failed_question_keys.update(failure.question_key for failure in answer_failures)
            self._write_failed_questions(failures)

            if not self._self_reflection_enabled(config):
                for question in active_questions:
                    if (
                        question.question_key in failed_question_keys
                        or question.question_key not in answers_by_question
                    ):
                        continue
                    agent_result = self._build_round_agent_result(
                        agent_id=agent.agent_id,
                        answer=answers_by_question[question.question_key],
                        self_review=None,
                    )
                    rounds_by_question[question.question_key].append(
                        QuestionRoundResult(round_index=round_index, agent_results=[agent_result])
                    )
                    if len(rounds_by_question[question.question_key]) == config.runtime.num_rounds:
                        question_paths.append(
                            self._persist_complete_question(
                                run_path=run_path,
                                run_id=run_id,
                                question=question,
                                rounds=rounds_by_question[question.question_key],
                            )
                        )
                        persisted_question_keys.add(question.question_key)
                        del rounds_by_question[question.question_key]
                        prior_feedback_by_question.pop(question.question_key, None)
                continue

            review_questions = [
                question
                for question in active_questions
                if question.question_key not in failed_question_keys
            ]
            if not review_questions:
                continue

            reviews_by_question, review_failures = self._collect_batch_self_reviews(
                agent=agent,
                questions=review_questions,
                answers_by_question=answers_by_question,
                round_index=round_index,
                progress_callback=progress_callback,
                question_index=question_index,
                total_questions=total_questions,
                total_rounds=config.runtime.num_rounds,
                max_workers=config.runtime.max_workers,
                experiment_config=config,
                on_question_self_review=(
                    (
                        lambda question, self_review: self._persist_final_self_review_question(
                            run_path=run_path,
                            run_id=run_id,
                            agent=agent,
                            question=question,
                            answer=answers_by_question[question.question_key],
                            self_review=self_review,
                            round_index=round_index,
                            rounds_by_question=rounds_by_question,
                            prior_feedback_by_question=prior_feedback_by_question,
                            persisted_question_keys=persisted_question_keys,
                            question_paths=question_paths,
                        )
                    )
                    if round_index == config.runtime.num_rounds
                    else None
                ),
            )
            failures.extend(review_failures)
            failed_question_keys.update(failure.question_key for failure in review_failures)
            self._write_failed_questions(failures)

            for question in review_questions:
                if (
                    question.question_key in failed_question_keys
                    or question.question_key in persisted_question_keys
                    or question.question_key not in reviews_by_question
                ):
                    continue
                agent_result = self._build_round_agent_result(
                    agent_id=agent.agent_id,
                    answer=answers_by_question[question.question_key],
                    self_review=reviews_by_question[question.question_key],
                )
                rounds_by_question[question.question_key].append(
                    QuestionRoundResult(round_index=round_index, agent_results=[agent_result])
                )
                prior_feedback_by_question[question.question_key] = _SingleAgentPriorFeedback(
                    previous_answer=answers_by_question[question.question_key],
                    self_review=reviews_by_question[question.question_key],
                )
                if len(rounds_by_question[question.question_key]) == config.runtime.num_rounds:
                    question_paths.append(
                        self._persist_complete_question(
                            run_path=run_path,
                            run_id=run_id,
                            question=question,
                            rounds=rounds_by_question[question.question_key],
                        )
                    )
                    persisted_question_keys.add(question.question_key)
                    del rounds_by_question[question.question_key]
                    prior_feedback_by_question.pop(question.question_key, None)

        for question in dataset:
            if (
                question.question_key in failed_question_keys
                or question.question_key in persisted_question_keys
                or question.question_key not in rounds_by_question
            ):
                continue
            if len(rounds_by_question[question.question_key]) != config.runtime.num_rounds:
                continue
            question_paths.append(
                self._persist_complete_question(
                    run_path=run_path,
                    run_id=run_id,
                    question=question,
                    rounds=rounds_by_question[question.question_key],
                )
            )
            persisted_question_keys.add(question.question_key)
        self._write_failed_questions(failures)
        write_batch_integrity_summary(run_path, failures)

        return RunArtifacts(
            run_id=run_id,
            run_path=str(run_path),
            manifest_path=str(run_path / "manifest.json"),
            question_paths=question_paths,
        )

    def _persist_complete_question(
        self,
        *,
        run_path: str | Path,
        run_id: str,
        question: QuestionRecord,
        rounds: list[QuestionRoundResult],
    ) -> str:
        assert self.store is not None
        result = QuestionRunResult.from_task(
            run_id=run_id,
            task=question,
            rounds=rounds,
        )
        return self.store.persist_question_result(run_path, result)

    def _persist_final_self_review_question(
        self,
        *,
        run_path: str | Path,
        run_id: str,
        agent: MasAgent,
        question: QuestionRecord,
        answer: AnswerSubmission,
        self_review: ReviewSubmission,
        round_index: int,
        rounds_by_question: dict[str, list[QuestionRoundResult]],
        prior_feedback_by_question: dict[str, _SingleAgentPriorFeedback | None],
        persisted_question_keys: set[str],
        question_paths: list[str],
    ) -> None:
        if question.question_key in persisted_question_keys:
            return
        agent_result = self._build_round_agent_result(
            agent_id=agent.agent_id,
            answer=answer,
            self_review=self_review,
        )
        rounds_by_question[question.question_key].append(
            QuestionRoundResult(round_index=round_index, agent_results=[agent_result])
        )
        question_paths.append(
            self._persist_complete_question(
                run_path=run_path,
                run_id=run_id,
                question=question,
                rounds=rounds_by_question[question.question_key],
            )
        )
        persisted_question_keys.add(question.question_key)
        del rounds_by_question[question.question_key]
        prior_feedback_by_question.pop(question.question_key, None)

    def _collect_batch_answers(
        self,
        *,
        agent: MasAgent,
        questions: list[QuestionRecord],
        prior_feedback_by_question: dict[str, _SingleAgentPriorFeedback | None],
        round_index: int,
        progress_callback: QuestionProgressCallback | None,
        question_index: dict[str, int],
        total_questions: int,
        total_rounds: int,
    ) -> tuple[dict[str, AnswerSubmission], list[BatchFailure]]:
        answers_by_question: dict[str, AnswerSubmission] = {}
        failures: list[BatchFailure] = []
        batches = self._build_answer_batches(
            questions=questions,
            prior_feedback_by_question=prior_feedback_by_question,
        )
        if progress_callback is not None and questions:
            progress_callback(
                1,
                total_questions,
                questions[0],
                f"round {round_index}/{total_rounds} answer batches {len(batches)} planned",
            )
        self._record_batch_event(
            event="planned",
            stage="answer",
            round_index=round_index,
            agent_id=agent.agent_id,
            questions=questions,
            details={"batch_count": len(batches)},
        )

        for batch in batches:
            first_question = batch[0]
            if progress_callback is not None:
                progress_callback(
                    question_index[first_question.question_key],
                    total_questions,
                    first_question,
                    (
                        f"round {round_index}/{total_rounds} batch answer "
                        f"{agent.agent_id} {len(batch)}q started"
                    ),
                )
            result, result_failures = self._answer_batch_with_fallback(
                agent=agent,
                questions=batch,
                prior_feedback_by_question=prior_feedback_by_question,
                round_index=round_index,
            )
            if progress_callback is not None:
                progress_callback(
                    question_index[first_question.question_key],
                    total_questions,
                    first_question,
                    (
                        f"round {round_index}/{total_rounds} batch answer "
                        f"{agent.agent_id} {len(batch)}q completed"
                    ),
                )
            answers_by_question.update(result)
            failures.extend(result_failures)

        missing_question_keys = {
            question.question_key
            for question in questions
            if question.question_key not in answers_by_question
        }
        failed_question_keys = {failure.question_key for failure in failures}
        for question in questions:
            if question.question_key in failed_question_keys:
                continue
            if question.question_key in missing_question_keys:
                failures.extend(
                    self._build_failures(
                        questions=[question],
                        stage="answer",
                        round_index=round_index,
                        agent_id=agent.agent_id,
                        error=RuntimeError(f"Missing answer for {question.question_key}"),
                    )
                )
        return answers_by_question, failures

    def _collect_batch_self_reviews(
        self,
        *,
        agent: MasAgent,
        questions: list[QuestionRecord],
        answers_by_question: dict[str, AnswerSubmission],
        round_index: int,
        progress_callback: QuestionProgressCallback | None,
        question_index: dict[str, int],
        total_questions: int,
        total_rounds: int,
        max_workers: int,
        experiment_config: ExperimentConfig,
        on_question_self_review: Callable[[QuestionRecord, ReviewSubmission], None] | None = None,
    ) -> tuple[dict[str, ReviewSubmission], list[BatchFailure]]:
        reviews_by_question: dict[str, ReviewSubmission] = {}
        failures: list[BatchFailure] = []
        questions_by_key = {question.question_key: question for question in questions}
        handled_question_keys: set[str] = set()
        batches = self._build_self_review_batches(
            questions=questions,
            answers_by_question=answers_by_question,
        )
        if progress_callback is not None and questions:
            progress_callback(
                1,
                total_questions,
                questions[0],
                f"round {round_index}/{total_rounds} review batches {len(batches)} planned",
            )
        self._record_batch_event(
            event="planned",
            stage="self_review",
            round_index=round_index,
            agent_id=agent.agent_id,
            questions=questions,
            details={"batch_count": len(batches)},
        )

        def task(batch: list[QuestionRecord]):
            first_question = batch[0]
            if progress_callback is not None:
                progress_callback(
                    question_index[first_question.question_key],
                    total_questions,
                    first_question,
                    (
                        f"round {round_index}/{total_rounds} batch review "
                        f"{agent.agent_id} {len(batch)}q started"
                    ),
                )
            result, result_failures = self._self_review_batch_with_fallback(
                agent=agent,
                questions=batch,
                answers_by_question=answers_by_question,
                round_index=round_index,
                experiment_config=experiment_config,
            )
            if progress_callback is not None:
                progress_callback(
                    question_index[first_question.question_key],
                    total_questions,
                    first_question,
                    (
                        f"round {round_index}/{total_rounds} batch review "
                        f"{agent.agent_id} {len(batch)}q completed"
                    ),
                )
            return result, result_failures

        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(batches) or 1))
        futures = {}
        try:
            futures = {executor.submit(task, batch): batch for batch in batches}
            for future in as_completed(futures):
                result, result_failures = future.result()
                reviews_by_question.update(result)
                failures.extend(result_failures)
                failed_question_keys = {failure.question_key for failure in failures}
                if on_question_self_review is not None:
                    for question_key, self_review in result.items():
                        if (
                            question_key in failed_question_keys
                            or question_key in handled_question_keys
                        ):
                            continue
                        on_question_self_review(questions_by_key[question_key], self_review)
                        handled_question_keys.add(question_key)
        except Exception:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        failed_question_keys = {failure.question_key for failure in failures}
        for question in questions:
            if (
                question.question_key in failed_question_keys
                or question.question_key in handled_question_keys
            ):
                continue
            if question.question_key not in reviews_by_question:
                failures.extend(
                    self._build_failures(
                        questions=[question],
                        stage="self_review",
                        round_index=round_index,
                        agent_id=agent.agent_id,
                        error=RuntimeError(f"Missing self-review for {question.question_key}"),
                    )
                )
        return {
            question_key: review
            for question_key, review in reviews_by_question.items()
            if question_key not in handled_question_keys
        }, failures

    def _answer_batch_with_fallback(
        self,
        *,
        agent: MasAgent,
        questions: list[QuestionRecord],
        prior_feedback_by_question: dict[str, _SingleAgentPriorFeedback | None],
        round_index: int,
    ) -> tuple[dict[str, AnswerSubmission], list[BatchFailure]]:
        try:
            result = self._run_with_batch_timeout(
                stage="answer",
                round_index=round_index,
                agent_id=agent.agent_id,
                questions=questions,
                call=lambda: agent.answer_many(
                    question_contexts=questions,
                    prior_feedback_by_question={
                        question.question_key: prior_feedback_by_question.get(
                            question.question_key
                        )
                        for question in questions
                    },
                    round_index=round_index,
                ),
            )
            return result, []
        except Exception as error:
            if len(questions) == 1 and _is_fatal_provider_error(error):
                raise
            if len(questions) == 1:
                result = self._retry_single_question_until_success(
                    stage="answer",
                    round_index=round_index,
                    agent_id=agent.agent_id,
                    questions=questions,
                    first_error=error,
                    call=lambda: agent.answer_many(
                        question_contexts=questions,
                        prior_feedback_by_question={
                            question.question_key: prior_feedback_by_question.get(
                                question.question_key
                            )
                            for question in questions
                        },
                        round_index=round_index,
                    ),
                )
                return result, []
            if (
                not self.batch_options.fallback_to_smaller_batches
                or not _should_fallback_to_smaller_batches(error)
            ):
                raise
            left, right = self._split_in_half(questions)
            self._record_batch_event(
                event="split",
                stage="answer",
                round_index=round_index,
                agent_id=agent.agent_id,
                questions=questions,
                error=error,
                details={"left": len(left), "right": len(right)},
            )
            result, failures = self._answer_batch_with_fallback(
                agent=agent,
                questions=left,
                prior_feedback_by_question=prior_feedback_by_question,
                round_index=round_index,
            )
            right_result, right_failures = self._answer_batch_with_fallback(
                agent=agent,
                questions=right,
                prior_feedback_by_question=prior_feedback_by_question,
                round_index=round_index,
            )
            result.update(right_result)
            failures.extend(right_failures)
            return result, failures

    def _self_review_batch_with_fallback(
        self,
        *,
        agent: MasAgent,
        questions: list[QuestionRecord],
        answers_by_question: dict[str, AnswerSubmission],
        round_index: int,
        experiment_config: ExperimentConfig,
    ) -> tuple[dict[str, ReviewSubmission], list[BatchFailure]]:
        try:
            result = self._run_with_batch_timeout(
                stage="self_review",
                round_index=round_index,
                agent_id=agent.agent_id,
                questions=questions,
                call=lambda: self._self_review_many(
                    agent=agent,
                    questions=questions,
                    answers_by_question=answers_by_question,
                    round_index=round_index,
                    experiment_config=experiment_config,
                ),
            )
            return result, []
        except Exception as error:
            if len(questions) == 1 and _is_fatal_provider_error(error):
                raise
            if len(questions) == 1:
                result = self._retry_single_question_until_success(
                    stage="self_review",
                    round_index=round_index,
                    agent_id=agent.agent_id,
                    questions=questions,
                    first_error=error,
                    call=lambda: self._self_review_many(
                        agent=agent,
                        questions=questions,
                        answers_by_question=answers_by_question,
                        round_index=round_index,
                        experiment_config=experiment_config,
                    ),
                )
                return result, []
            if (
                not self.batch_options.fallback_to_smaller_batches
                or not _should_fallback_to_smaller_batches(error)
            ):
                raise
            left, right = self._split_in_half(questions)
            self._record_batch_event(
                event="split",
                stage="self_review",
                round_index=round_index,
                agent_id=agent.agent_id,
                questions=questions,
                error=error,
                details={"left": len(left), "right": len(right)},
            )
            result, failures = self._self_review_batch_with_fallback(
                agent=agent,
                questions=left,
                answers_by_question=answers_by_question,
                round_index=round_index,
                experiment_config=experiment_config,
            )
            right_result, right_failures = self._self_review_batch_with_fallback(
                agent=agent,
                questions=right,
                answers_by_question=answers_by_question,
                round_index=round_index,
                experiment_config=experiment_config,
            )
            result.update(right_result)
            failures.extend(right_failures)
            return result, failures

    def _retry_single_question_until_success(
        self,
        *,
        stage: str,
        round_index: int,
        agent_id: str,
        questions: list[QuestionRecord],
        first_error: Exception,
        call: Callable[[], object],
    ):
        attempts = 1
        last_error = first_error
        while True:
            attempts += 1
            retry_delay = self.batch_options.single_question_retry_delay_seconds
            if retry_delay > 0:
                time.sleep(retry_delay)
            self._record_batch_event(
                event="retry",
                stage=stage,
                round_index=round_index,
                agent_id=agent_id,
                questions=questions,
                error=last_error,
                details={"attempt": attempts},
            )
            try:
                return self._run_with_batch_timeout(
                    stage=stage,
                    round_index=round_index,
                    agent_id=agent_id,
                    questions=questions,
                    call=call,
                )
            except Exception as error:
                if _is_fatal_provider_error(error):
                    raise
                last_error = error

    def _can_retry_single_question(self, completed_attempts: int) -> bool:
        max_attempts = self.batch_options.single_question_max_attempts
        return max_attempts is None or completed_attempts < max_attempts

    def _self_review_many(
        self,
        *,
        agent: MasAgent,
        questions: list[QuestionRecord],
        answers_by_question: dict[str, AnswerSubmission],
        round_index: int,
        experiment_config: ExperimentConfig,
    ) -> dict[str, ReviewSubmission]:
        if isinstance(agent, SingleAgentModelBackedAgent):
            return agent.self_review_many(
                questions=questions,
                answers_by_question=answers_by_question,
                round_index=round_index,
            )
        if isinstance(agent, ModelBackedMasAgent):
            response = agent.client.generate_json(
                system_prompt=agent.agent_config.resolved_review_prompt(
                    experiment_config.review_prompt
                ),
                user_prompt=_build_batch_self_review_user_prompt(
                    question_records=questions,
                    round_index=round_index,
                    answers_by_question=answers_by_question,
                ),
                model=agent.agent_config.model or agent.default_model,
                temperature=agent.agent_config.temperature,
            )
            return _parse_batch_self_reviews(
                agent_id=agent.agent_id,
                questions=questions,
                raw_reviews=response.get("reviews", []),
            )

        raw_reviews_by_question = agent.review_many_questions(
            question_contexts=questions,
            reviewer_answers_by_question={
                question.question_key: answers_by_question[question.question_key]
                for question in questions
            },
            peer_submissions_by_question={
                question.question_key: [answers_by_question[question.question_key]]
                for question in questions
            },
            round_index=round_index,
        )
        result: dict[str, ReviewSubmission] = {}
        for question in questions:
            reviews = raw_reviews_by_question.get(question.question_key, [])
            if len(reviews) != 1:
                raise ValidationError(f"Expected one self-review for {question.question_key}.")
            result[question.question_key] = _validate_self_review(reviews[0])
        return result

    def _build_answer_batches(
        self,
        *,
        questions: list[QuestionRecord],
        prior_feedback_by_question: dict[str, _SingleAgentPriorFeedback | None],
    ) -> list[list[QuestionRecord]]:
        def prompt_for(batch: list[QuestionRecord]) -> str:
            return _build_batch_single_answer_user_prompt(
                question_records=batch,
                round_index=1,
                prior_feedback_by_question={
                    question.question_key: prior_feedback_by_question.get(question.question_key)
                    for question in batch
                },
            )

        return self._partition_by_prompt(
            questions=questions,
            prompt_for=prompt_for,
            max_input_tokens=self.batch_options.answer_max_input_tokens,
            max_questions_per_batch=self.batch_options.answer_max_questions_per_batch,
        )

    def _build_self_review_batches(
        self,
        *,
        questions: list[QuestionRecord],
        answers_by_question: dict[str, AnswerSubmission],
    ) -> list[list[QuestionRecord]]:
        def prompt_for(batch: list[QuestionRecord]) -> str:
            return _build_batch_self_review_user_prompt(
                question_records=batch,
                round_index=1,
                answers_by_question=answers_by_question,
            )

        return self._partition_by_prompt(
            questions=questions,
            prompt_for=prompt_for,
            max_input_tokens=self.batch_options.review_max_input_tokens,
            max_questions_per_batch=self.batch_options.review_max_questions_per_batch,
        )

    def _partition_by_prompt(
        self,
        *,
        questions: list[QuestionRecord],
        prompt_for: Callable[[list[QuestionRecord]], str],
        max_input_tokens: int,
        max_questions_per_batch: int,
    ) -> list[list[QuestionRecord]]:
        if max_questions_per_batch < 1:
            raise ValueError("max_questions_per_batch must be at least 1.")
        batches: list[list[QuestionRecord]] = []
        current: list[QuestionRecord] = []
        for question in questions:
            candidate = [*current, question]
            if current and (
                len(candidate) > max_questions_per_batch
                or estimate_tokens(prompt_for(candidate)) > max_input_tokens
            ):
                batches.append(current)
                current = [question]
            else:
                current = candidate
        if current:
            batches.append(current)
        return batches

    def _split_in_half(
        self,
        questions: list[QuestionRecord],
    ) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
        midpoint = max(1, len(questions) // 2)
        return questions[:midpoint], questions[midpoint:]

    def _run_with_batch_timeout(
        self,
        *,
        stage: str,
        round_index: int,
        agent_id: str,
        questions: list[QuestionRecord],
        call: Callable[[], object],
    ):
        timeout_seconds = self.batch_options.batch_timeout_seconds
        self._record_batch_event(
            event="started",
            stage=stage,
            round_index=round_index,
            agent_id=agent_id,
            questions=questions,
        )
        if timeout_seconds is None:
            try:
                result = call()
            except Exception as error:
                self._record_batch_event(
                    event="failed",
                    stage=stage,
                    round_index=round_index,
                    agent_id=agent_id,
                    questions=questions,
                    error=error,
                )
                raise
            self._record_batch_event(
                event="completed",
                stage=stage,
                round_index=round_index,
                agent_id=agent_id,
                questions=questions,
            )
            return result

        result_queue: queue.Queue[tuple[str, object | Exception]] = queue.Queue(maxsize=1)

        def run_call() -> None:
            try:
                result_queue.put(("ok", call()))
            except Exception as error:
                result_queue.put(("error", error))

        worker = threading.Thread(target=run_call, daemon=True)
        worker.start()
        worker.join(timeout_seconds)
        if worker.is_alive():
            error = BatchAttemptTimeout(
                f"{stage} batch for {agent_id} exceeded {timeout_seconds}s"
            )
            self._record_batch_event(
                event="timeout",
                stage=stage,
                round_index=round_index,
                agent_id=agent_id,
                questions=questions,
                error=error,
            )
            raise error

        status, payload = result_queue.get()
        if status == "error":
            assert isinstance(payload, Exception)
            self._record_batch_event(
                event="failed",
                stage=stage,
                round_index=round_index,
                agent_id=agent_id,
                questions=questions,
                error=payload,
            )
            raise payload

        self._record_batch_event(
            event="completed",
            stage=stage,
            round_index=round_index,
            agent_id=agent_id,
            questions=questions,
        )
        return payload

    def _build_failures(
        self,
        *,
        questions: list[QuestionRecord],
        stage: str,
        round_index: int,
        agent_id: str,
        error: Exception,
    ) -> list[BatchFailure]:
        return [
            BatchFailure(
                question_key=question.question_key,
                question_id=question.question_id,
                dataset_name=question.dataset_name,
                task_type=question.task_type,
                stage=stage,
                round_index=round_index,
                agent_id=agent_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            for question in questions
        ]

    def _record_batch_event(
        self,
        *,
        event: str,
        stage: str,
        round_index: int,
        agent_id: str | None,
        questions: list[QuestionRecord],
        error: Exception | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if self._active_run_path is None:
            return
        payload: dict[str, object] = {
            "created_at": datetime.now().isoformat(),
            "event": event,
            "stage": stage,
            "round_index": round_index,
            "agent_id": agent_id,
            "batch_size": len(questions),
            "question_keys": [question.question_key for question in questions],
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)
        if details:
            payload.update(details)
        event_path = self._active_run_path / "batch_events.jsonl"
        with self._event_lock:
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_failed_questions(self, failures: list[BatchFailure]) -> None:
        if self._active_run_path is None:
            return
        path = self._active_run_path / "failed_questions.json"
        path.write_text(
            json.dumps([failure.to_dict() for failure in failures], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
