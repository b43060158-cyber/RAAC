from __future__ import annotations

from abc import ABC, abstractmethod
import re

from autogen_mas.config import AgentConfig, ExperimentConfig
from autogen_mas.models import (
    QuestionRecord,
    AnswerSubmission,
    PriorRoundFeedback,
    ReviewSubmission,
    TaskRecord,
    ValidationError,
)
from autogen_mas.answer_consistency import extract_reasoning_decision_option_ids
from .alignment_judge import resolve_choice_answer_alignment

from .clients import StructuredLLMClient
from .prompting import (
    build_answer_user_prompt,
    build_batch_review_user_prompt,
    build_batch_answer_user_prompt,
    build_batch_questions_review_user_prompt,
    build_review_user_prompt,
    build_single_choice_answer_repair_user_prompt,
)


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


def _extract_choice_ids_from_text(text: str, valid_option_ids: list[str]) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    exact_match = next((option_id for option_id in valid_option_ids if stripped == option_id), None)
    if exact_match is not None:
        return [exact_match]

    matches: list[str] = []
    for option_id in valid_option_ids:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(option_id)}(?![A-Za-z0-9_])")
        if pattern.search(text):
            matches.append(option_id)
    return matches


def coerce_selected_option_ids(
    raw_selected_option_ids: object,
    *,
    question_context: TaskRecord,
    reasoning: str = "",
    final_answer: str = "",
) -> list[str]:
    if not isinstance(question_context, QuestionRecord):
        return []

    if isinstance(raw_selected_option_ids, list):
        values = [str(value).strip() for value in raw_selected_option_ids if str(value).strip()]
    elif isinstance(raw_selected_option_ids, tuple | set):
        values = [str(value).strip() for value in raw_selected_option_ids if str(value).strip()]
    elif isinstance(raw_selected_option_ids, str):
        values = _extract_choice_ids_from_text(raw_selected_option_ids, question_context.option_ids())
    elif raw_selected_option_ids is None:
        values = []
    else:
        values = [str(raw_selected_option_ids).strip()] if str(raw_selected_option_ids).strip() else []

    if question_context.task_type == "single_choice":
        if len(values) == 1:
            return values

        explicit_reasoning_choice = extract_reasoning_decision_option_ids(
            question_record=question_context,
            reasoning=reasoning,
        )
        if len(explicit_reasoning_choice) == 1:
            return explicit_reasoning_choice

        matched = _extract_choice_ids_from_text(final_answer, question_context.option_ids())
        if len(matched) == 1:
            return matched

        return values

    if values:
        return values

    return []


class MasAgent(ABC):
    def __init__(self, agent_config: AgentConfig) -> None:
        self.agent_config = agent_config
        self.agent_id = agent_config.agent_id

    @abstractmethod
    def answer(
        self,
        question_context: TaskRecord,
        prior_feedback: PriorRoundFeedback | None,
        round_index: int,
    ) -> AnswerSubmission:
        raise NotImplementedError

    @abstractmethod
    def review(
        self,
        question_context: TaskRecord,
        peer_submission: AnswerSubmission,
        reviewer_answer: AnswerSubmission,
        round_index: int,
    ) -> ReviewSubmission:
        raise NotImplementedError

    def review_many(
        self,
        question_context: TaskRecord,
        peer_submissions: list[AnswerSubmission],
        reviewer_answer: AnswerSubmission,
        round_index: int,
    ) -> list[ReviewSubmission]:
        return [
            self.review(
                question_context=question_context,
                peer_submission=peer_submission,
                reviewer_answer=reviewer_answer,
                round_index=round_index,
            )
            for peer_submission in peer_submissions
        ]

    def answer_many(
        self,
        question_contexts: list[TaskRecord],
        prior_feedback_by_question: dict[str, PriorRoundFeedback | None],
        round_index: int,
    ) -> dict[str, AnswerSubmission]:
        return {
            question.question_key: self.answer(
                question_context=question,
                prior_feedback=prior_feedback_by_question.get(question.question_key),
                round_index=round_index,
            )
            for question in question_contexts
        }

    def review_many_questions(
        self,
        question_contexts: list[TaskRecord],
        reviewer_answers_by_question: dict[str, AnswerSubmission],
        peer_submissions_by_question: dict[str, list[AnswerSubmission]],
        round_index: int,
    ) -> dict[str, list[ReviewSubmission]]:
        return {
            question.question_key: self.review_many(
                question_context=question,
                peer_submissions=peer_submissions_by_question[question.question_key],
                reviewer_answer=reviewer_answers_by_question[question.question_key],
                round_index=round_index,
            )
            for question in question_contexts
        }


class ModelBackedMasAgent(MasAgent):
    def __init__(
        self,
        agent_config: AgentConfig,
        experiment_config: ExperimentConfig,
        client: StructuredLLMClient,
        default_model: str,
        debug_mode: bool = False,
        enable_answer_repair: bool = False,
    ) -> None:
        super().__init__(agent_config)
        self.experiment_config = experiment_config
        self.client = client
        self.default_model = default_model
        self.debug_mode = debug_mode
        self.enable_answer_repair = enable_answer_repair

    def _normal_answer_mitigation_message(self, round_index: int) -> str | None:
        mitigation = self.experiment_config.normal_agent_answer_mitigation
        if not mitigation.enabled or round_index < mitigation.start_round:
            return None
        return mitigation.message

    def answer(
        self,
        question_context: TaskRecord,
        prior_feedback: PriorRoundFeedback | None,
        round_index: int,
    ) -> AnswerSubmission:
        response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_answer_prompt(
                self.experiment_config.answer_prompt
            ),
            user_prompt=build_answer_user_prompt(
                question_record=question_context,
                round_index=round_index,
                prior_feedback=prior_feedback,
                mitigation_message=self._normal_answer_mitigation_message(round_index),
                debug_mode=self.debug_mode,
            ),
            model=self.agent_config.model or self.default_model,
            temperature=self.agent_config.temperature,
        )
        submission = self._answer_submission_from_response(
            response=response,
            question_context=question_context,
            context=f"Agent {self.agent_id} answer",
        )
        try:
            return self._validate_answer_submission(
                submission=submission,
                question_context=question_context,
            )
        except ValidationError as error:
            if not self._should_repair_single_choice_answer(
                question_context=question_context,
                error=error,
            ):
                raise
        repair_response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_answer_prompt(
                self.experiment_config.answer_prompt
            ),
            user_prompt=build_single_choice_answer_repair_user_prompt(
                question_record=question_context,
                round_index=round_index,
                raw_response=response,
            ),
            model=self.agent_config.model or self.default_model,
            temperature=0.0,
        )
        repaired_payload = {**response, **repair_response}
        repair_submission = self._answer_submission_from_response(
            response=repaired_payload,
            question_context=question_context,
            context=f"Agent {self.agent_id} answer repair",
        )
        return self._validate_answer_submission(
            submission=repair_submission,
            question_context=question_context,
        )

    def _answer_submission_from_response(
        self,
        *,
        response: dict,
        question_context: TaskRecord,
        context: str,
    ) -> AnswerSubmission:
        return AnswerSubmission(
            agent_id=self.agent_id,
            selected_option_ids=coerce_selected_option_ids(
                response.get("selected_option_ids", []),
                question_context=question_context,
                reasoning=str(response.get("reasoning", "")),
                final_answer=str(response.get("final_answer", "")),
            ),
            reasoning=str(response.get("reasoning", "")),
            code=str(response.get("code", "")),
            final_answer=str(response.get("final_answer", "")),
            confidence=_debug_confidence(
                response,
                debug_mode=self.debug_mode,
                context=context,
            ),
            changed_answer=bool(response.get("changed_answer", False)),
            change_drivers=list(response.get("change_drivers", [])),
            change_summary=str(response.get("change_summary", "")),
            chain_of_thought=_debug_chain_of_thought(
                response,
                debug_mode=self.debug_mode,
                context=context,
            ),
        )

    def _validate_answer_submission(
        self,
        *,
        submission: AnswerSubmission,
        question_context: TaskRecord,
    ) -> AnswerSubmission:
        if isinstance(question_context, QuestionRecord):
            alignment = resolve_choice_answer_alignment(
                question_record=question_context,
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
        return submission.validate(question_context)

    def _should_repair_single_choice_answer(
        self,
        *,
        question_context: TaskRecord,
        error: ValidationError,
    ) -> bool:
        return (
            self.enable_answer_repair
            and isinstance(question_context, QuestionRecord)
            and question_context.task_type == "single_choice"
            and "must choose exactly one option for single_choice" in str(error)
        )

    def answer_many(
        self,
        question_contexts: list[TaskRecord],
        prior_feedback_by_question: dict[str, PriorRoundFeedback | None],
        round_index: int,
    ) -> dict[str, AnswerSubmission]:
        if not question_contexts:
            return {}
        response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_answer_prompt(
                self.experiment_config.answer_prompt
            ),
            user_prompt=build_batch_answer_user_prompt(
                question_records=question_contexts,
                round_index=round_index,
                prior_feedback_by_question=prior_feedback_by_question,
                mitigation_message=self._normal_answer_mitigation_message(round_index),
                debug_mode=self.debug_mode,
            ),
            model=self.agent_config.model or self.default_model,
            temperature=self.agent_config.temperature,
        )
        raw_answers = response.get("answers", [])
        if not isinstance(raw_answers, list):
            raise ValidationError("Batch answer response must contain an answers list.")

        questions_by_key = {question.question_key: question for question in question_contexts}
        answers_by_key: dict[str, AnswerSubmission] = {}
        for raw_answer in raw_answers:
            if not isinstance(raw_answer, dict):
                raise ValidationError("Each batch answer item must be an object.")
            question_key = str(raw_answer.get("question_key", ""))
            if question_key not in questions_by_key:
                raise ValidationError(f"Unexpected batch answer question_key: {question_key}")
            if question_key in answers_by_key:
                raise ValidationError(f"Duplicate batch answer question_key: {question_key}")
            question = questions_by_key[question_key]
            reasoning = str(raw_answer.get("reasoning", ""))
            final_answer = str(raw_answer.get("final_answer", ""))
            answers_by_key[question_key] = AnswerSubmission(
                agent_id=self.agent_id,
                selected_option_ids=coerce_selected_option_ids(
                    raw_answer.get("selected_option_ids", []),
                    question_context=question,
                    reasoning=reasoning,
                    final_answer=final_answer,
                ),
                reasoning=reasoning,
                code=str(raw_answer.get("code", "")),
                final_answer=final_answer,
                confidence=_debug_confidence(
                    raw_answer,
                    debug_mode=self.debug_mode,
                    context=f"Agent {self.agent_id} batch answer for {question_key}",
                ),
                changed_answer=bool(raw_answer.get("changed_answer", False)),
                change_drivers=list(raw_answer.get("change_drivers", [])),
                change_summary=str(raw_answer.get("change_summary", "")),
            )
            if isinstance(question, QuestionRecord):
                alignment = resolve_choice_answer_alignment(
                    question_record=question,
                    submitted_option_ids=list(answers_by_key[question_key].selected_option_ids),
                    reasoning=answers_by_key[question_key].reasoning,
                    alignment_judge_config=self.experiment_config.alignment_judge,
                    client=self.client,
                    default_model=self.default_model,
                )
                answers_by_key[question_key].raw_selected_option_ids = (
                    alignment.raw_selected_option_ids
                )
                answers_by_key[question_key].selected_option_ids = (
                    alignment.selected_option_ids
                )
                answers_by_key[question_key].alignment_resolution = alignment.resolution
                answers_by_key[question_key].alignment_judge_votes = alignment.judge_votes
            answers_by_key[question_key] = answers_by_key[question_key].validate(question)

        missing_keys = [
            question.question_key
            for question in question_contexts
            if question.question_key not in answers_by_key
        ]
        if missing_keys:
            raise ValidationError(f"Missing batch answer question_keys: {missing_keys}")
        return {
            question.question_key: answers_by_key[question.question_key]
            for question in question_contexts
        }

    def review(
        self,
        question_context: TaskRecord,
        peer_submission: AnswerSubmission,
        reviewer_answer: AnswerSubmission,
        round_index: int,
    ) -> ReviewSubmission:
        response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_review_prompt(
                self.experiment_config.review_prompt
            ),
            user_prompt=build_review_user_prompt(
                question_record=question_context,
                round_index=round_index,
                reviewer_answer=reviewer_answer,
                peer_submission=peer_submission,
                debug_mode=self.debug_mode,
            ),
            model=self.agent_config.model or self.default_model,
            temperature=self.agent_config.temperature,
        )
        submission = ReviewSubmission(
            reviewer_agent_id=self.agent_id,
            target_agent_id=peer_submission.agent_id,
            score=int(response.get("score", 0)),
            stance=str(response.get("stance", "")),
            main_reason=str(response.get("main_reason", "")),
            chain_of_thought=_debug_chain_of_thought(
                response,
                debug_mode=self.debug_mode,
                context=(
                    f"Review {self.agent_id}->{peer_submission.agent_id}"
                ),
            ),
        )
        return submission.validate()

    def review_many(
        self,
        question_context: TaskRecord,
        peer_submissions: list[AnswerSubmission],
        reviewer_answer: AnswerSubmission,
        round_index: int,
    ) -> list[ReviewSubmission]:
        if not peer_submissions:
            return []
        response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_review_prompt(
                self.experiment_config.review_prompt
            ),
            user_prompt=build_batch_review_user_prompt(
                question_record=question_context,
                round_index=round_index,
                reviewer_answer=reviewer_answer,
                peer_submissions=peer_submissions,
                debug_mode=self.debug_mode,
            ),
            model=self.agent_config.model or self.default_model,
            temperature=self.agent_config.temperature,
        )

        raw_reviews = response.get("reviews", [])
        if not isinstance(raw_reviews, list):
            raise ValidationError("Batch review response must contain a reviews list.")

        expected_order = [submission.agent_id for submission in peer_submissions]
        expected_targets = set(expected_order)
        reviews_by_target: dict[str, ReviewSubmission] = {}
        for raw_review in raw_reviews:
            if not isinstance(raw_review, dict):
                raise ValidationError("Each batch review item must be an object.")
            target_agent_id = str(raw_review.get("target_agent_id", ""))
            if target_agent_id not in expected_targets:
                raise ValidationError(f"Unexpected batch review target: {target_agent_id}")
            if target_agent_id in reviews_by_target:
                raise ValidationError(f"Duplicate batch review target: {target_agent_id}")
            reviews_by_target[target_agent_id] = ReviewSubmission(
                reviewer_agent_id=self.agent_id,
                target_agent_id=target_agent_id,
                score=int(raw_review.get("score", 0)),
                stance=str(raw_review.get("stance", "")),
                main_reason=str(raw_review.get("main_reason", "")),
                chain_of_thought=_debug_chain_of_thought(
                    raw_review,
                    debug_mode=self.debug_mode,
                    context=f"Review {self.agent_id}->{target_agent_id}",
                ),
            ).validate()

        missing_targets = [target for target in expected_order if target not in reviews_by_target]
        if missing_targets:
            raise ValidationError(f"Missing batch review targets: {missing_targets}")
        return [reviews_by_target[target] for target in expected_order]

    def review_many_questions(
        self,
        question_contexts: list[TaskRecord],
        reviewer_answers_by_question: dict[str, AnswerSubmission],
        peer_submissions_by_question: dict[str, list[AnswerSubmission]],
        round_index: int,
    ) -> dict[str, list[ReviewSubmission]]:
        if not question_contexts:
            return {}
        response = self.client.generate_json(
            system_prompt=self.agent_config.resolved_review_prompt(
                self.experiment_config.review_prompt
            ),
            user_prompt=build_batch_questions_review_user_prompt(
                question_records=question_contexts,
                round_index=round_index,
                reviewer_answers_by_question=reviewer_answers_by_question,
                peer_submissions_by_question=peer_submissions_by_question,
            ),
            model=self.agent_config.model or self.default_model,
            temperature=self.agent_config.temperature,
        )

        raw_reviews = response.get("reviews", [])
        if not isinstance(raw_reviews, list):
            raise ValidationError("Batch questions review response must contain a reviews list.")

        expected_targets_by_question = {
            question.question_key: [
                submission.agent_id
                for submission in peer_submissions_by_question[question.question_key]
            ]
            for question in question_contexts
        }
        expected_question_keys = set(expected_targets_by_question)
        reviews_by_pair: dict[tuple[str, str], ReviewSubmission] = {}
        for raw_review in raw_reviews:
            if not isinstance(raw_review, dict):
                raise ValidationError("Each batch questions review item must be an object.")
            question_key = str(raw_review.get("question_key", ""))
            target_agent_id = str(raw_review.get("target_agent_id", ""))
            if question_key not in expected_question_keys:
                raise ValidationError(f"Unexpected batch review question_key: {question_key}")
            if target_agent_id not in expected_targets_by_question[question_key]:
                raise ValidationError(
                    f"Unexpected batch review target for {question_key}: {target_agent_id}"
                )
            pair_key = (question_key, target_agent_id)
            if pair_key in reviews_by_pair:
                raise ValidationError(
                    f"Duplicate batch review for {question_key}/{target_agent_id}"
                )
            reviews_by_pair[pair_key] = ReviewSubmission(
                reviewer_agent_id=self.agent_id,
                target_agent_id=target_agent_id,
                score=int(raw_review.get("score", 0)),
                stance=str(raw_review.get("stance", "")),
                main_reason=str(raw_review.get("main_reason", "")),
            ).validate()

        missing_pairs = [
            (question_key, target_agent_id)
            for question_key, targets in expected_targets_by_question.items()
            for target_agent_id in targets
            if (question_key, target_agent_id) not in reviews_by_pair
        ]
        if missing_pairs:
            raise ValidationError(f"Missing batch question review pairs: {missing_pairs}")
        return {
            question.question_key: [
                reviews_by_pair[(question.question_key, target_agent_id)]
                for target_agent_id in expected_targets_by_question[question.question_key]
            ]
            for question in question_contexts
        }
