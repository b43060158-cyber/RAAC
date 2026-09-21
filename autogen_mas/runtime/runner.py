from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from autogen_mas.answer_matching import short_answer_tie_key
from autogen_mas.config import DashScopeSettings, ExperimentConfig
from autogen_mas.models import (
    PriorRoundFeedback,
    QuestionRoundResult,
    QuestionRunResult,
    RoundAgentResult,
    RunArtifacts,
    TaskRecord,
)
from autogen_mas.persistence import JsonRunStore

from .agent import MasAgent, ModelBackedMasAgent
from .clients import build_structured_client


AgentFactory = Callable[[ExperimentConfig, DashScopeSettings], list[MasAgent]]
QuestionProgressCallback = Callable[[int, int, TaskRecord, str], None]


@dataclass(slots=True)
class QuestionFailure:
    question_index: int
    question_id: str
    question_key: str
    error_type: str
    error_message: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "question_index": self.question_index,
            "question_id": self.question_id,
            "question_key": self.question_key,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(slots=True)
class CompetitionRunner:
    experiment_config: ExperimentConfig
    llm_settings: DashScopeSettings
    store: JsonRunStore | None = None
    agent_factory: AgentFactory | None = None
    debug_mode: bool = False
    persist_feedback_summary: bool = False

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = JsonRunStore(self.experiment_config.runtime.output_dir)

    def _build_agents(self, experiment_config: ExperimentConfig | None = None) -> list[MasAgent]:
        config = experiment_config or self.experiment_config
        if self.agent_factory is not None:
            return self.agent_factory(config, self.llm_settings)
        client = build_structured_client(self.llm_settings)
        return [
            ModelBackedMasAgent(
                agent_config=agent_config,
                experiment_config=config,
                client=client,
                default_model=self.llm_settings.model,
                debug_mode=self.debug_mode,
                enable_answer_repair=True,
            )
            for agent_config in config.agents
        ]

    def run_question(
        self,
        question_record: TaskRecord,
        experiment_config: ExperimentConfig | None = None,
        run_id: str | None = None,
        progress_callback: QuestionProgressCallback | None = None,
        question_index: int = 1,
        total_questions: int = 1,
    ) -> QuestionRunResult:
        config = experiment_config or self.experiment_config
        run_identifier = run_id or self._default_run_id(config)
        agents = self._build_agents(config)
        prior_feedback_map: dict[str, PriorRoundFeedback | None] = {
            agent.agent_id: None for agent in agents
        }
        rounds: list[QuestionRoundResult] = []

        for round_index in range(1, config.runtime.num_rounds + 1):
            answers = self._collect_answers(
                agents=agents,
                question_record=question_record,
                prior_feedback_map=prior_feedback_map,
                round_index=round_index,
                max_workers=config.runtime.max_workers,
                progress_callback=progress_callback,
                question_index=question_index,
                total_questions=total_questions,
                total_rounds=config.runtime.num_rounds,
            )
            round_result = self._collect_reviews_and_scores(
                agents=agents,
                question_record=question_record,
                answers=answers,
                round_index=round_index,
                max_workers=config.runtime.max_workers,
                progress_callback=progress_callback,
                question_index=question_index,
                total_questions=total_questions,
                total_rounds=config.runtime.num_rounds,
            )
            round_result.reached_consensus = self._round_reached_consensus(
                question_record,
                round_result,
            )
            rounds.append(round_result)
            if (
                config.runtime.consensus_short_circuit
                and round_result.reached_consensus
            ):
                break
            prior_feedback_map = {
                result.agent_id: result.to_prior_feedback()
                for result in round_result.agent_results
            }

        return QuestionRunResult.from_task(
            run_id=run_identifier,
            task=question_record,
            rounds=rounds,
        )

    def run_dataset(
        self,
        dataset: list[TaskRecord],
        experiment_config: ExperimentConfig | None = None,
        progress_callback: QuestionProgressCallback | None = None,
        resume_run_path: str | Path | None = None,
    ) -> RunArtifacts:
        config = experiment_config or self.experiment_config
        assert self.store is not None
        if resume_run_path is None:
            run_id = self._default_run_id(config)
            run_path = self.store.initialize_run(
                run_id=run_id,
                experiment_config=config,
                llm_settings=self.llm_settings,
            )
        else:
            run_path = Path(resume_run_path)
            manifest = self.store.read_manifest(run_path)
            if manifest is None:
                raise ValueError(f"Cannot resume run without a readable manifest: {run_path}")
            run_id = str(manifest.get("run_id") or run_path.name)
        question_paths: list[str] = []
        failure_records = self._load_failure_records(run_path)
        total_questions = len(dataset)
        agent_ids = self._resume_agent_ids(config)
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
                    agent_ids=agent_ids,
                    num_rounds=config.runtime.num_rounds,
                    allow_consensus_short_circuit=(
                        config.runtime.consensus_short_circuit
                    ),
                ):
                    question_paths.append(str(question_path))
                    if self._remove_failure_record(
                        failure_records,
                        question_key=question_record.question_key,
                    ):
                        self._write_failed_questions(run_path, failure_records.values())
                    if progress_callback is not None:
                        progress_callback(index, total_questions, question_record, "completed")
                    continue
            if progress_callback is not None:
                progress_callback(index, total_questions, question_record, "started")
            try:
                result = self.run_question(
                    question_record,
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
                self._write_failed_questions(run_path, failure_records.values())
                if progress_callback is not None:
                    progress_callback(
                        index,
                        total_questions,
                        question_record,
                        f"failed: {type(error).__name__}",
                    )
                continue
            question_paths.append(
                self.store.persist_question_result(
                    run_path,
                    result,
                    persist_feedback_summary=(
                        self.debug_mode and self.persist_feedback_summary
                    ),
                )
            )
            if self._remove_failure_record(
                failure_records,
                question_key=question_record.question_key,
            ):
                self._write_failed_questions(run_path, failure_records.values())
            if progress_callback is not None:
                progress_callback(index, total_questions, question_record, "completed")
        self._write_failed_questions(run_path, failure_records.values())
        return RunArtifacts(
            run_id=run_id,
            run_path=str(run_path),
            manifest_path=str(run_path / "manifest.json"),
            question_paths=question_paths,
        )

    def _is_complete_question_payload(
        self,
        *,
        payload: dict | None,
        question_record: TaskRecord,
        agent_ids: list[str],
        num_rounds: int,
        allow_consensus_short_circuit: bool = True,
    ) -> bool:
        if payload is None:
            return False
        if str(payload.get("question_key", "")) != question_record.question_key:
            return False
        rounds = payload.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            return False
        if not all(isinstance(round_data, dict) for round_data in rounds):
            return False
        has_expected_round_count = len(rounds) == num_rounds
        has_consensus_completion = (
            allow_consensus_short_circuit
            and self._supports_consensus_short_circuit(question_record)
            and len(rounds) < num_rounds
            and bool(rounds[-1].get("reached_consensus", False))
        )
        if not has_expected_round_count and not has_consensus_completion:
            return False

        expected_agents = set(agent_ids)
        for round_data in rounds:
            agent_results = round_data.get("agent_results")
            if not isinstance(agent_results, list):
                return False
            if len(agent_results) != len(agent_ids):
                return False

            results_by_agent: dict[str, dict] = {}
            for agent_result in agent_results:
                if not isinstance(agent_result, dict):
                    return False
                agent_id = str(agent_result.get("agent_id", ""))
                if agent_id not in expected_agents or agent_id in results_by_agent:
                    return False
                results_by_agent[agent_id] = agent_result
            if set(results_by_agent) != expected_agents:
                return False

            for agent_id, agent_result in results_by_agent.items():
                if not self._has_complete_answer(agent_result):
                    return False
                expected_peers = expected_agents - {agent_id}
                if not self._reviews_cover(
                    agent_result.get("reviews_given"),
                    id_field="target_agent_id",
                    expected_ids=expected_peers,
                ):
                    return False
                if not self._reviews_cover(
                    agent_result.get("received_reviews"),
                    id_field="reviewer_agent_id",
                    expected_ids=expected_peers,
                ):
                    return False
        return True

    def _supports_consensus_short_circuit(self, question_record: TaskRecord) -> bool:
        return question_record.task_type in {
            "single_choice",
            "multiple_choice",
            "math_short_answer",
        }

    def _round_reached_consensus(
        self,
        question_record: TaskRecord,
        round_result: QuestionRoundResult,
    ) -> bool:
        if not self._supports_consensus_short_circuit(question_record):
            return False
        answer_keys = {
            self._consensus_answer_key(question_record, result.answer)
            for result in round_result.agent_results
        }
        return bool(answer_keys) and len(answer_keys) == 1

    def _consensus_answer_key(self, question_record: TaskRecord, answer: object) -> tuple:
        if question_record.task_type == "math_short_answer":
            final_answer = str(getattr(answer, "final_answer", "")).strip()
            return (
                "final_answer",
                short_answer_tie_key(
                    final_answer,
                    metadata=getattr(question_record, "metadata", None),
                ),
            )
        selected_option_ids = getattr(answer, "selected_option_ids", [])
        return (
            "options",
            tuple(sorted(str(item) for item in selected_option_ids)),
        )

    def _resume_agent_ids(self, experiment_config: ExperimentConfig) -> list[str]:
        return [agent.agent_id for agent in experiment_config.agents]

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

    def _write_failed_questions(
        self,
        run_path: str | Path,
        failures,
    ) -> None:
        assert self.store is not None
        self.store.persist_failed_questions(
            run_path,
            [
                failure.to_dict() if isinstance(failure, QuestionFailure) else dict(failure)
                for failure in failures
            ],
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

    def _collect_answers(
        self,
        *,
        agents: list[MasAgent],
        question_record: TaskRecord,
        prior_feedback_map: dict[str, PriorRoundFeedback | None],
        round_index: int,
        max_workers: int,
        progress_callback: QuestionProgressCallback | None = None,
        question_index: int = 1,
        total_questions: int = 1,
        total_rounds: int = 1,
    ) -> dict[str, object]:
        def task(agent: MasAgent) -> tuple[str, object]:
            if progress_callback is not None:
                progress_callback(
                    question_index,
                    total_questions,
                    question_record,
                    f"round {round_index}/{total_rounds} answer {agent.agent_id} started",
                )
            answer = agent.answer(
                question_context=question_record,
                prior_feedback=prior_feedback_map[agent.agent_id],
                round_index=round_index,
            )
            if progress_callback is not None:
                progress_callback(
                    question_index,
                    total_questions,
                    question_record,
                    f"round {round_index}/{total_rounds} answer {agent.agent_id} completed",
                )
            return (
                agent.agent_id,
                answer,
            )

        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(agents)))
        futures = []
        try:
            futures = [executor.submit(task, agent) for agent in agents]
            answers = dict(future.result() for future in futures)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        return answers

    def _collect_reviews_and_scores(
        self,
        *,
        agents: list[MasAgent],
        question_record: TaskRecord,
        answers: dict[str, object],
        round_index: int,
        max_workers: int,
        progress_callback: QuestionProgressCallback | None = None,
        question_index: int = 1,
        total_questions: int = 1,
        total_rounds: int = 1,
    ) -> QuestionRoundResult:
        review_groups = [
            (reviewer, [target for target in agents if target.agent_id != reviewer.agent_id])
            for reviewer in agents
        ]

        def task(group: tuple[MasAgent, list[MasAgent]]):
            reviewer, targets = group
            reviewer_answer = answers[reviewer.agent_id]
            target_answers = [answers[target.agent_id] for target in targets]
            if progress_callback is not None:
                progress_callback(
                    question_index,
                    total_questions,
                    question_record,
                    (
                        f"round {round_index}/{total_rounds} review "
                        f"{reviewer.agent_id}->peers started"
                    ),
                )
            reviews = reviewer.review_many(
                question_context=question_record,
                peer_submissions=target_answers,  # type: ignore[arg-type]
                reviewer_answer=reviewer_answer,  # type: ignore[arg-type]
                round_index=round_index,
            )
            if progress_callback is not None:
                progress_callback(
                    question_index,
                    total_questions,
                    question_record,
                    (
                        f"round {round_index}/{total_rounds} review "
                        f"{reviewer.agent_id}->peers completed"
                    ),
                )
            return reviews

        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(review_groups) or 1))
        futures = []
        try:
            futures = [executor.submit(task, group) for group in review_groups]
            review_batches = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        reviews = [review for batch in review_batches for review in batch]

        reviews_by_reviewer: dict[str, list] = {agent.agent_id: [] for agent in agents}
        reviews_by_target: dict[str, list] = {agent.agent_id: [] for agent in agents}
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
                    answer=answers[agent.agent_id],  # type: ignore[arg-type]
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

    def _default_run_id(self, experiment_config: ExperimentConfig | None = None) -> str:
        config = experiment_config or self.experiment_config
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{config.runtime.run_id_prefix}-{timestamp}"
