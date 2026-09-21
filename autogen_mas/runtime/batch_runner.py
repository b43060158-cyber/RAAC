from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
from typing import Callable, TypeVar

from autogen_mas.models import (
    AnswerSubmission,
    PriorRoundFeedback,
    QuestionRoundResult,
    QuestionRunResult,
    ReviewSubmission,
    RoundAgentResult,
    RunArtifacts,
    TaskRecord,
)
from autogen_mas.persistence import JsonRunStore

from .agent import MasAgent
from .clients import LLMGenerationError, LLMProviderError
from .prompting import build_batch_answer_user_prompt, build_batch_questions_review_user_prompt
from .runner import CompetitionRunner, QuestionProgressCallback


def estimate_tokens(text: str) -> int:
    """Cheap conservative token estimate; avoids adding tokenizer dependencies."""
    return max(1, (len(text) + 2) // 3)


@dataclass(slots=True)
class BatchRuntimeOptions:
    answer_max_input_tokens: int = 12_000
    review_max_input_tokens: int = 12_000
    answer_max_questions_per_batch: int = 8
    review_max_questions_per_batch: int = 4
    batch_timeout_seconds: int | None = None
    fallback_to_smaller_batches: bool = True
    continue_on_single_question_failure: bool = True
    retry_single_question_until_success: bool = True
    single_question_max_attempts: int | None = None
    single_question_retry_delay_seconds: float = 0.0


@dataclass(slots=True)
class BatchFailure:
    question_key: str
    question_id: str
    dataset_name: str
    task_type: str
    stage: str
    round_index: int
    agent_id: str
    error_type: str
    error_message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "question_key": self.question_key,
            "question_id": self.question_id,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "stage": self.stage,
            "round_index": self.round_index,
            "agent_id": self.agent_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class BatchAttemptTimeout(LLMGenerationError):
    """Raised when one batch attempt exceeds the configured wall-clock deadline."""


def _classify_batch_error(message: str) -> str:
    if "Missing batch answer question_keys" in message:
        return "missing_batch_answer_question_keys"
    if "Missing single-agent batch answer question_keys" in message:
        return "missing_single_agent_batch_answer_question_keys"
    if "Missing batch self-review question_keys" in message:
        return "missing_batch_self_review_question_keys"
    if "Missing batch review targets" in message:
        return "missing_batch_review_targets"
    if "Missing batch question review pairs" in message:
        return "missing_batch_question_review_pairs"
    if "Missing answer for " in message:
        return "missing_answer_after_merge"
    if "Missing self-review for " in message:
        return "missing_self_review_after_merge"
    if "Missing answers for " in message:
        return "missing_answers_after_merge"
    if "Missing reviews for " in message:
        return "missing_reviews_after_merge"
    if "Duplicate batch answer question_key" in message:
        return "duplicate_batch_answer_question_key"
    if "Duplicate single-agent batch answer question_key" in message:
        return "duplicate_single_agent_batch_answer_question_key"
    if "Duplicate batch self-review question_key" in message:
        return "duplicate_batch_self_review_question_key"
    if "Duplicate batch review for " in message:
        return "duplicate_batch_review_pair"
    if "Duplicate batch review target" in message:
        return "duplicate_batch_review_target"
    if "Unexpected batch answer question_key" in message:
        return "unexpected_batch_answer_question_key"
    if "Unexpected single-agent batch answer question_key" in message:
        return "unexpected_single_agent_batch_answer_question_key"
    if "Unexpected batch self-review question_key" in message:
        return "unexpected_batch_self_review_question_key"
    if "Unexpected batch review question_key" in message:
        return "unexpected_batch_review_question_key"
    if "Unexpected batch review target" in message:
        return "unexpected_batch_review_target"
    if "Batch answer response must contain an answers list" in message:
        return "malformed_batch_answer_response"
    if "Single-agent batch answer response must contain an answers list" in message:
        return "malformed_single_agent_batch_answer_response"
    if "Batch review response must contain a reviews list" in message:
        return "malformed_batch_review_response"
    if "Batch self-review response must contain a reviews list" in message:
        return "malformed_batch_self_review_response"
    if "Batch questions review response must contain a reviews list" in message:
        return "malformed_batch_questions_review_response"
    if "Expected one self-review" in message:
        return "missing_or_extra_self_review"
    return "other"


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def build_batch_integrity_summary(
    run_path: str | Path,
    failures: list[BatchFailure],
) -> dict[str, object]:
    path = Path(run_path)
    events_path = path / "batch_events.jsonl"
    events: list[dict[str, object]] = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                events.append({"event": "unparseable", "error_message": line})
                continue
            if isinstance(event, dict):
                events.append(event)

    event_counts = Counter(str(event.get("event", "unknown")) for event in events)
    stage_counts = Counter(str(event.get("stage", "unknown")) for event in events)
    event_stage_counts = Counter(
        f"{event.get('stage', 'unknown')}:{event.get('event', 'unknown')}"
        for event in events
    )
    error_type_counts = Counter(
        str(event.get("error_type", "unknown")) for event in events if event.get("error_type")
    )
    error_category_counts = Counter(
        _classify_batch_error(str(event.get("error_message", "")))
        for event in events
        if event.get("error_message")
    )

    final_failure_error_types = Counter(failure.error_type for failure in failures)
    final_failure_stages = Counter(failure.stage for failure in failures)
    final_failure_categories = Counter(
        _classify_batch_error(failure.error_message) for failure in failures
    )

    return {
        "run_path": str(path),
        "batch_events_path": str(events_path),
        "failed_questions_path": str(path / "failed_questions.json"),
        "total_batch_events": len(events),
        "event_counts": _counter_to_sorted_dict(event_counts),
        "stage_counts": _counter_to_sorted_dict(stage_counts),
        "event_stage_counts": _counter_to_sorted_dict(event_stage_counts),
        "error_event_count": sum(1 for event in events if event.get("error_type")),
        "error_type_counts": _counter_to_sorted_dict(error_type_counts),
        "error_category_counts": _counter_to_sorted_dict(error_category_counts),
        "split_count": event_counts.get("split", 0),
        "retry_count": event_counts.get("retry", 0),
        "timeout_count": event_counts.get("timeout", 0),
        "failed_event_count": event_counts.get("failed", 0),
        "abandoned_single_question_count": event_counts.get(
            "abandoned_single_question",
            0,
        ),
        "final_failed_question_count": len(failures),
        "final_failed_question_keys": sorted({failure.question_key for failure in failures}),
        "final_failed_by_stage": _counter_to_sorted_dict(final_failure_stages),
        "final_failed_by_error_type": _counter_to_sorted_dict(final_failure_error_types),
        "final_failed_by_error_category": _counter_to_sorted_dict(final_failure_categories),
        "has_missing_items": any(
            category.startswith("missing")
            for category in [*error_category_counts, *final_failure_categories]
        ),
        "has_duplicate_items": any(
            category.startswith("duplicate")
            for category in [*error_category_counts, *final_failure_categories]
        ),
        "has_unexpected_items": any(
            category.startswith("unexpected")
            for category in [*error_category_counts, *final_failure_categories]
        ),
    }


def write_batch_integrity_summary(
    run_path: str | Path,
    failures: list[BatchFailure],
) -> Path:
    path = Path(run_path) / "batch_integrity_summary.json"
    path.write_text(
        json.dumps(build_batch_integrity_summary(run_path, failures), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


T = TypeVar("T")


def _should_fallback_to_smaller_batches(error: Exception) -> bool:
    if isinstance(error, LLMProviderError) and error.status_code in {401, 403}:
        return False
    if isinstance(error, LLMGenerationError):
        message = str(error)
        if "HTTP Error 401" in message or "HTTP Error 403" in message:
            return False
    return True


def _is_fatal_provider_error(error: Exception) -> bool:
    if isinstance(error, LLMProviderError) and error.status_code in {401, 403}:
        return True
    if isinstance(error, LLMGenerationError):
        message = str(error)
        return "HTTP Error 401" in message or "HTTP Error 403" in message
    return False


@dataclass(slots=True)
class BatchCompetitionRunner(CompetitionRunner):
    batch_options: BatchRuntimeOptions = field(default_factory=BatchRuntimeOptions)
    _active_run_path: Path | None = field(default=None, init=False, repr=False)
    _event_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        raise RuntimeError(
            "Batch mode for LLM-MAS has been removed. Use CompetitionRunner instead."
        )

    def run_dataset(
        self,
        dataset: list[TaskRecord],
        experiment_config=None,
        progress_callback: QuestionProgressCallback | None = None,
    ) -> RunArtifacts:
        config = experiment_config or self.experiment_config
        run_id = self._default_run_id(config)
        assert self.store is not None
        run_path = self.store.initialize_run(
            run_id=run_id,
            experiment_config=config,
            llm_settings=self.llm_settings,
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

        agents = self._build_agents(config)
        total_questions = len(dataset)
        question_index = {question.question_key: index for index, question in enumerate(dataset, 1)}
        rounds_by_question: dict[str, list[QuestionRoundResult]] = {
            question.question_key: [] for question in dataset
        }
        failed_question_keys: set[str] = set()
        persisted_question_keys: set[str] = set()
        failures: list[BatchFailure] = []
        prior_feedback_by_agent: dict[str, dict[str, PriorRoundFeedback | None]] = {
            agent.agent_id: {question.question_key: None for question in dataset}
            for agent in agents
        }
        question_paths: list[str] = []

        for round_index in range(1, config.runtime.num_rounds + 1):
            active_questions = [
                question
                for question in dataset
                if question.question_key not in failed_question_keys
                and question.question_key not in persisted_question_keys
            ]
            if not active_questions:
                break
            if progress_callback is not None:
                progress_callback(
                    1,
                    total_questions,
                    active_questions[0],
                    f"round {round_index}/{config.runtime.num_rounds} started",
                )
            answers_by_question, answer_failures = self._collect_batch_answers(
                agents=agents,
                questions=active_questions,
                prior_feedback_by_agent=prior_feedback_by_agent,
                round_index=round_index,
                max_workers=config.runtime.max_workers,
                progress_callback=progress_callback,
                question_index=question_index,
                total_questions=total_questions,
                total_rounds=config.runtime.num_rounds,
            )
            failures.extend(answer_failures)
            failed_question_keys.update(failure.question_key for failure in answer_failures)
            self._write_failed_questions(failures)

            review_questions = [
                question
                for question in active_questions
                if question.question_key not in failed_question_keys
            ]
            if not review_questions:
                continue

            round_results, review_failures = self._collect_batch_reviews_and_scores(
                agents=agents,
                questions=review_questions,
                answers_by_question=answers_by_question,
                round_index=round_index,
                max_workers=config.runtime.max_workers,
                progress_callback=progress_callback,
                question_index=question_index,
                total_questions=total_questions,
                total_rounds=config.runtime.num_rounds,
                on_question_round_result=(
                    (
                        lambda question, round_result: self._persist_final_round_question(
                            run_path=run_path,
                            run_id=run_id,
                            question=question,
                            round_result=round_result,
                            rounds_by_question=rounds_by_question,
                            prior_feedback_by_agent=prior_feedback_by_agent,
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
                    or question.question_key not in round_results
                ):
                    continue
                round_result = round_results[question.question_key]
                rounds_by_question[question.question_key].append(round_result)
                for result in round_result.agent_results:
                    prior_feedback_by_agent[result.agent_id][question.question_key] = (
                        result.to_prior_feedback()
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
                    for feedback_by_question in prior_feedback_by_agent.values():
                        feedback_by_question.pop(question.question_key, None)
            if progress_callback is not None:
                progress_callback(
                    total_questions,
                    total_questions,
                    active_questions[-1],
                    f"round {round_index}/{config.runtime.num_rounds} completed",
                )

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
        question: TaskRecord,
        rounds: list[QuestionRoundResult],
    ) -> str:
        assert self.store is not None
        result = QuestionRunResult.from_task(
            run_id=run_id,
            task=question,
            rounds=rounds,
        )
        return self.store.persist_question_result(run_path, result)

    def _collect_batch_answers(
        self,
        *,
        agents: list[MasAgent],
        questions: list[TaskRecord],
        prior_feedback_by_agent: dict[str, dict[str, PriorRoundFeedback | None]],
        round_index: int,
        max_workers: int,
        progress_callback: QuestionProgressCallback | None,
        question_index: dict[str, int],
        total_questions: int,
        total_rounds: int,
    ) -> tuple[dict[str, dict[str, AnswerSubmission]], list[BatchFailure]]:
        answers_by_question: dict[str, dict[str, AnswerSubmission]] = {
            question.question_key: {} for question in questions
        }
        failures: list[BatchFailure] = []
        batches_by_agent: list[tuple[MasAgent, list[list[TaskRecord]]]] = []
        batch_counts_by_agent: dict[str, int] = {}
        for agent in agents:
            batches = self._build_answer_batches(
                questions=questions,
                prior_feedback_by_question=prior_feedback_by_agent[agent.agent_id],
            )
            batch_counts_by_agent[agent.agent_id] = len(batches)
            batches_by_agent.append((agent, batches))
        tasks = self._interleave_agent_batches(batches_by_agent)
        if progress_callback is not None and questions:
            question_batch_count = max(batch_counts_by_agent.values(), default=0)
            progress_callback(
                1,
                total_questions,
                questions[0],
                (
                    f"round {round_index}/{total_rounds} answer agent_batches {len(tasks)} "
                    f"question_batches {question_batch_count} planned"
                ),
            )
        self._record_batch_event(
            event="planned",
            stage="answer",
            round_index=round_index,
            agent_id=None,
            questions=questions,
            details={
                "agent_batch_count": len(tasks),
                "question_batch_count": max(batch_counts_by_agent.values(), default=0),
                "batch_counts_by_agent": batch_counts_by_agent,
            },
        )

        def task(item: tuple[MasAgent, list[TaskRecord]]):
            agent, batch = item
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
                prior_feedback_by_question=prior_feedback_by_agent[agent.agent_id],
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
            return result, result_failures

        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(tasks) or 1))
        futures = {}
        try:
            futures = {executor.submit(task, item): item for item in tasks}
            for future in as_completed(futures):
                result, result_failures = future.result()
                for question_key, answer in result.items():
                    answers_by_question[question_key][answer.agent_id] = answer
                failures.extend(result_failures)
        except Exception:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        expected_agent_ids = {agent.agent_id for agent in agents}
        failed_keys = {failure.question_key for failure in failures}
        for question in questions:
            if question.question_key in failed_keys:
                continue
            missing = expected_agent_ids - set(answers_by_question[question.question_key])
            if missing:
                failures.extend(
                    self._build_failures(
                        questions=[question],
                        stage="answer",
                        round_index=round_index,
                        agent_id=",".join(sorted(missing)),
                        error=RuntimeError(
                            f"Missing answers for {question.question_key}: {sorted(missing)}"
                        ),
                    )
                )
        return answers_by_question, failures

    def _collect_batch_reviews_and_scores(
        self,
        *,
        agents: list[MasAgent],
        questions: list[TaskRecord],
        answers_by_question: dict[str, dict[str, AnswerSubmission]],
        round_index: int,
        max_workers: int,
        progress_callback: QuestionProgressCallback | None,
        question_index: dict[str, int],
        total_questions: int,
        total_rounds: int,
        on_question_round_result: (
            Callable[[TaskRecord, QuestionRoundResult], None] | None
        ) = None,
    ) -> tuple[dict[str, QuestionRoundResult], list[BatchFailure]]:
        reviews_by_question: dict[str, list[ReviewSubmission]] = {
            question.question_key: [] for question in questions
        }
        failures: list[BatchFailure] = []
        expected_reviews_per_question = len(agents) * max(0, len(agents) - 1)
        questions_by_key = {question.question_key: question for question in questions}
        handled_question_keys: set[str] = set()
        batches_by_reviewer: list[tuple[MasAgent, list[list[TaskRecord]]]] = []
        batch_counts_by_reviewer: dict[str, int] = {}
        for reviewer in agents:
            batches = self._build_review_batches(
                questions=questions,
                reviewer=reviewer,
                answers_by_question=answers_by_question,
            )
            batch_counts_by_reviewer[reviewer.agent_id] = len(batches)
            batches_by_reviewer.append((reviewer, batches))
        tasks = self._interleave_agent_batches(batches_by_reviewer)
        if progress_callback is not None and questions:
            question_batch_count = max(batch_counts_by_reviewer.values(), default=0)
            progress_callback(
                1,
                total_questions,
                questions[0],
                (
                    f"round {round_index}/{total_rounds} review agent_batches {len(tasks)} "
                    f"question_batches {question_batch_count} planned"
                ),
            )
        self._record_batch_event(
            event="planned",
            stage="review",
            round_index=round_index,
            agent_id=None,
            questions=questions,
            details={
                "agent_batch_count": len(tasks),
                "question_batch_count": max(batch_counts_by_reviewer.values(), default=0),
                "batch_counts_by_reviewer": batch_counts_by_reviewer,
            },
        )

        def task(item: tuple[MasAgent, list[TaskRecord]]):
            reviewer, batch = item
            first_question = batch[0]
            if progress_callback is not None:
                progress_callback(
                    question_index[first_question.question_key],
                    total_questions,
                    first_question,
                    (
                        f"round {round_index}/{total_rounds} batch review "
                        f"{reviewer.agent_id} {len(batch)}q started"
                    ),
                )
            result, result_failures = self._review_batch_with_fallback(
                reviewer=reviewer,
                questions=batch,
                answers_by_question=answers_by_question,
                round_index=round_index,
            )
            if progress_callback is not None:
                progress_callback(
                    question_index[first_question.question_key],
                    total_questions,
                    first_question,
                    (
                        f"round {round_index}/{total_rounds} batch review "
                        f"{reviewer.agent_id} {len(batch)}q completed"
                    ),
                )
            return result, result_failures

        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(tasks) or 1))
        futures = {}
        try:
            futures = {executor.submit(task, item): item for item in tasks}
            for future in as_completed(futures):
                result, result_failures = future.result()
                updated_question_keys: set[str] = set()
                for question_key, reviews in result.items():
                    reviews_by_question[question_key].extend(reviews)
                    updated_question_keys.add(question_key)
                failures.extend(result_failures)
                failed_keys = {failure.question_key for failure in failures}
                if on_question_round_result is not None:
                    for question_key in updated_question_keys:
                        if (
                            question_key in failed_keys
                            or question_key in handled_question_keys
                            or len(reviews_by_question[question_key])
                            != expected_reviews_per_question
                        ):
                            continue
                        question = questions_by_key[question_key]
                        round_result = self._build_round_result(
                            agents=agents,
                            question=question,
                            answers=answers_by_question[question_key],
                            reviews=reviews_by_question[question_key],
                            round_index=round_index,
                        )
                        on_question_round_result(question, round_result)
                        handled_question_keys.add(question_key)
        except Exception:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        failed_keys = {failure.question_key for failure in failures}
        for question in questions:
            if (
                question.question_key in failed_keys
                or question.question_key in handled_question_keys
            ):
                continue
            if len(reviews_by_question[question.question_key]) != expected_reviews_per_question:
                failures.extend(
                    self._build_failures(
                        questions=[question],
                        stage="review",
                        round_index=round_index,
                        agent_id="batch_review",
                        error=RuntimeError(
                            "Missing reviews for "
                            f"{question.question_key}: "
                            f"{len(reviews_by_question[question.question_key])}/"
                            f"{expected_reviews_per_question}"
                        ),
                    )
                )
        failed_keys = {failure.question_key for failure in failures}
        return {
            question.question_key: self._build_round_result(
                agents=agents,
                question=question,
                answers=answers_by_question[question.question_key],
                reviews=reviews_by_question[question.question_key],
                round_index=round_index,
            )
            for question in questions
            if question.question_key not in failed_keys
            and question.question_key not in handled_question_keys
        }, failures

    def _persist_final_round_question(
        self,
        *,
        run_path: str | Path,
        run_id: str,
        question: TaskRecord,
        round_result: QuestionRoundResult,
        rounds_by_question: dict[str, list[QuestionRoundResult]],
        prior_feedback_by_agent: dict[str, dict[str, PriorRoundFeedback | None]],
        persisted_question_keys: set[str],
        question_paths: list[str],
    ) -> None:
        if question.question_key in persisted_question_keys:
            return
        rounds_by_question[question.question_key].append(round_result)
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
        for feedback_by_question in prior_feedback_by_agent.values():
            feedback_by_question.pop(question.question_key, None)

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
                        question.question_key: prior_feedback_by_question.get(question.question_key)
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
                result = self._retry_single_question_until_success(
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

    def _retry_single_question_until_success(
        self,
        *,
        stage: str,
        round_index: int,
        agent_id: str,
        questions: list[TaskRecord],
        first_error: Exception,
        call: Callable[[], T],
    ) -> T:
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

    def _build_answer_batches(
        self,
        *,
        questions: list[TaskRecord],
        prior_feedback_by_question: dict[str, PriorRoundFeedback | None],
    ) -> list[list[TaskRecord]]:
        def prompt_for(batch: list[TaskRecord]) -> str:
            return build_batch_answer_user_prompt(
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

    def _build_review_batches(
        self,
        *,
        questions: list[TaskRecord],
        reviewer: MasAgent,
        answers_by_question: dict[str, dict[str, AnswerSubmission]],
    ) -> list[list[TaskRecord]]:
        def prompt_for(batch: list[TaskRecord]) -> str:
            return build_batch_questions_review_user_prompt(
                question_records=batch,
                round_index=1,
                reviewer_answers_by_question={
                    question.question_key: answers_by_question[question.question_key][
                        reviewer.agent_id
                    ]
                    for question in batch
                },
                peer_submissions_by_question={
                    question.question_key: [
                        answer
                        for agent_id, answer in answers_by_question[question.question_key].items()
                        if agent_id != reviewer.agent_id
                    ]
                    for question in batch
                },
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
        questions: list[TaskRecord],
        prompt_for: Callable[[list[TaskRecord]], str],
        max_input_tokens: int,
        max_questions_per_batch: int,
    ) -> list[list[TaskRecord]]:
        if max_questions_per_batch < 1:
            raise ValueError("max_questions_per_batch must be at least 1.")
        batches: list[list[TaskRecord]] = []
        current: list[TaskRecord] = []
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

    def _build_round_result(
        self,
        *,
        agents: list[MasAgent],
        question: TaskRecord,
        answers: dict[str, AnswerSubmission],
        reviews: list[ReviewSubmission],
        round_index: int,
    ) -> QuestionRoundResult:
        reviews_by_reviewer: dict[str, list[ReviewSubmission]] = {
            agent.agent_id: [] for agent in agents
        }
        reviews_by_target: dict[str, list[ReviewSubmission]] = {
            agent.agent_id: [] for agent in agents
        }
        for review in reviews:
            reviews_by_reviewer[review.reviewer_agent_id].append(review)
            reviews_by_target[review.target_agent_id].append(review)

        agent_results: list[RoundAgentResult] = []
        for agent in agents:
            received_reviews = reviews_by_target[agent.agent_id]
            total_score = float(sum(review.score for review in received_reviews))
            average_score = total_score / len(received_reviews) if received_reviews else 0.0
            agent_results.append(
                RoundAgentResult(
                    agent_id=agent.agent_id,
                    answer=answers[agent.agent_id],
                    reviews_given=reviews_by_reviewer[agent.agent_id],
                    received_reviews=received_reviews,
                    total_score=total_score,
                    average_score=average_score,
                    is_adversarial_agent=bool(
                        getattr(agent, "is_adversarial_agent", False)
                    ),
                )
            )
        return QuestionRoundResult(round_index=round_index, agent_results=agent_results)

    def _split_in_half(self, questions: list[TaskRecord]) -> tuple[list[TaskRecord], list[TaskRecord]]:
        midpoint = max(1, len(questions) // 2)
        return questions[:midpoint], questions[midpoint:]

    def _interleave_agent_batches(
        self,
        batches_by_agent: list[tuple[MasAgent, list[list[TaskRecord]]]],
    ) -> list[tuple[MasAgent, list[TaskRecord]]]:
        tasks: list[tuple[MasAgent, list[TaskRecord]]] = []
        max_batch_count = max((len(batches) for _, batches in batches_by_agent), default=0)
        for batch_index in range(max_batch_count):
            for agent, batches in batches_by_agent:
                if batch_index < len(batches):
                    tasks.append((agent, batches[batch_index]))
        return tasks

    def _run_with_batch_timeout(
        self,
        *,
        stage: str,
        round_index: int,
        agent_id: str,
        questions: list[TaskRecord],
        call: Callable[[], T],
    ) -> T:
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

        result_queue: queue.Queue[tuple[str, T | Exception]] = queue.Queue(maxsize=1)

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
        return payload  # type: ignore[return-value]

    def _build_failures(
        self,
        *,
        questions: list[TaskRecord],
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
        questions: list[TaskRecord],
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
