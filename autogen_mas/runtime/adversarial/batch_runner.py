from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time

from ...config import ExperimentConfig
from ...models import AnswerSubmission, PriorRoundFeedback, ReviewSubmission, TaskRecord
from ..agent import MasAgent
from ..batch_runner import (
    BatchCompetitionRunner,
    BatchFailure,
    _is_fatal_provider_error,
    _should_fallback_to_smaller_batches,
)
from ..clients import build_structured_client

from .agents import AdversarialHonestMasAgent, AdversarialMasAgent
from .parsing import ADVERSARIAL_SINGLE_QUESTION_MAX_ATTEMPTS
from .runner import materialize_adversarial_agent_configs, resolve_adversarial_agent_ids


@dataclass(slots=True)
class BatchAdversarialCompetitionRunner(BatchCompetitionRunner):
    def __post_init__(self) -> None:
        raise RuntimeError(
            "Batch mode for adversarial LLM-MAS has been removed. "
            "Use AdversarialCompetitionRunner instead."
        )

    def _answer_batch_with_fallback(
        self,
        *,
        agent: MasAgent,
        questions: list[TaskRecord],
        prior_feedback_by_question: dict[str, PriorRoundFeedback | None],
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
                try:
                    result = self._retry_single_question_with_limit(
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
                except Exception as final_error:
                    if not self.batch_options.continue_on_single_question_failure:
                        raise
                    return {}, self._build_failures(
                        questions=questions,
                        stage="answer",
                        round_index=round_index,
                        agent_id=agent.agent_id,
                        error=final_error,
                    )
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

    def _review_batch_with_fallback(
        self,
        *,
        reviewer: MasAgent,
        questions: list[TaskRecord],
        answers_by_question: dict[str, dict[str, AnswerSubmission]],
        round_index: int,
    ) -> tuple[dict[str, list[ReviewSubmission]], list[BatchFailure]]:
        try:
            result = self._run_with_batch_timeout(
                stage="review",
                round_index=round_index,
                agent_id=reviewer.agent_id,
                questions=questions,
                call=lambda: reviewer.review_many_questions(
                    question_contexts=questions,
                    reviewer_answers_by_question={
                        question.question_key: answers_by_question[question.question_key][
                            reviewer.agent_id
                        ]
                        for question in questions
                    },
                    peer_submissions_by_question={
                        question.question_key: [
                            answer
                            for agent_id, answer in answers_by_question[
                                question.question_key
                            ].items()
                            if agent_id != reviewer.agent_id
                        ]
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
                try:
                    result = self._retry_single_question_with_limit(
                        stage="review",
                        round_index=round_index,
                        agent_id=reviewer.agent_id,
                        questions=questions,
                        first_error=error,
                        call=lambda: reviewer.review_many_questions(
                            question_contexts=questions,
                            reviewer_answers_by_question={
                                question.question_key: answers_by_question[question.question_key][
                                    reviewer.agent_id
                                ]
                                for question in questions
                            },
                            peer_submissions_by_question={
                                question.question_key: [
                                    answer
                                    for agent_id, answer in answers_by_question[
                                        question.question_key
                                    ].items()
                                    if agent_id != reviewer.agent_id
                                ]
                                for question in questions
                            },
                            round_index=round_index,
                        ),
                    )
                    return result, []
                except Exception as final_error:
                    if not self.batch_options.continue_on_single_question_failure:
                        raise
                    return {}, self._build_failures(
                        questions=questions,
                        stage="review",
                        round_index=round_index,
                        agent_id=reviewer.agent_id,
                        error=final_error,
                    )
            if (
                not self.batch_options.fallback_to_smaller_batches
                or not _should_fallback_to_smaller_batches(error)
            ):
                raise
            left, right = self._split_in_half(questions)
            self._record_batch_event(
                event="split",
                stage="review",
                round_index=round_index,
                agent_id=reviewer.agent_id,
                questions=questions,
                error=error,
                details={"left": len(left), "right": len(right)},
            )
            result, failures = self._review_batch_with_fallback(
                reviewer=reviewer,
                questions=left,
                answers_by_question=answers_by_question,
                round_index=round_index,
            )
            right_result, right_failures = self._review_batch_with_fallback(
                reviewer=reviewer,
                questions=right,
                answers_by_question=answers_by_question,
                round_index=round_index,
            )
            result.update(right_result)
            failures.extend(right_failures)
            return result, failures

    def _retry_single_question_with_limit(
        self,
        *,
        stage: str,
        round_index: int,
        agent_id: str,
        questions: list[TaskRecord],
        first_error: Exception,
        call,
    ):
        attempts = 1
        last_error = first_error
        while (
            self.batch_options.retry_single_question_until_success
            and self._can_retry_single_question(attempts)
        ):
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
                result = self._run_with_batch_timeout(
                    stage=stage,
                    round_index=round_index,
                    agent_id=agent_id,
                    questions=questions,
                    call=call,
                )
                return result, []
            except Exception as error:
                if _is_fatal_provider_error(error):
                    raise
                last_error = error
        self._record_batch_event(
            event="abandoned_single_question",
            stage=stage,
            round_index=round_index,
            agent_id=agent_id,
            questions=questions,
            error=last_error,
            details={"attempts": attempts},
        )
        raise last_error

    def _build_agents(self, experiment_config: ExperimentConfig | None = None) -> list[MasAgent]:
        config = experiment_config or self.experiment_config
        if self.agent_factory is not None:
            return self.agent_factory(config, self.llm_settings)
        client = build_structured_client(self.llm_settings)
        adversarial_ids = resolve_adversarial_agent_ids(config)
        normal_default_model = (
            config.adversarial_mas.normal_agent_model or self.llm_settings.model
        )
        agents: list[MasAgent] = []
        for agent_config in materialize_adversarial_agent_configs(config):
            if agent_config.agent_id in adversarial_ids:
                agents.append(
                    AdversarialMasAgent(
                        agent_config=agent_config,
                        behavior_config=config.adversarial_mas.adversarial_agent,
                        client=client,
                        default_model=self.llm_settings.model,
                        seed=config.adversarial_mas.seed,
                        adversarial_agent_ids=adversarial_ids,
                    )
                )
            else:
                agents.append(
                    AdversarialHonestMasAgent(
                        agent_config=agent_config,
                        experiment_config=config,
                        client=client,
                        default_model=normal_default_model,
                    )
                )
        return agents

    def _default_run_id(self, experiment_config: ExperimentConfig | None = None) -> str:
        config = experiment_config or self.experiment_config
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{config.adversarial_mas.run_id_prefix}-{timestamp}"
