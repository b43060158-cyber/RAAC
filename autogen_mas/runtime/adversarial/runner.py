from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from autogen_mas.baseline import get_choice_baseline_spec
from ...config import AgentConfig, ExperimentConfig
from ...models import AnswerSubmission, QuestionRecord, TaskRecord
from ..agent import MasAgent
from ..clients import build_structured_client
from ..runner import CompetitionRunner, QuestionProgressCallback

from .agents import AdversarialHonestMasAgent, AdversarialMasAgent
from .few_shot import Strategy8ReviewContext, most_popular_non_reference_option
from .strategy import ATTACK_STRATEGY_FEW_SHOT_RERANK


def _agent_id_sequence(total_agents: int) -> list[str]:
    return [f"agent_{index}" for index in range(1, total_agents + 1)]


def _effective_adversarial_run_prefix(config: ExperimentConfig) -> str:
    prefix = config.adversarial_mas.run_id_prefix
    if prefix != "adversarial-competition":
        return prefix
    baseline_spec = get_choice_baseline_spec(
        config.adversarial_mas.adversarial_agent.choice_attack_strategy
    )
    if baseline_spec is None:
        return prefix
    return f"{baseline_spec.strategy_name.replace('_', '-')}-{prefix}"


def _enable_honest_answer_repair(config: ExperimentConfig) -> bool:
    return config.adversarial_mas.adversarial_agent.choice_attack_strategy in {
        "baseline_1",
        "fusion_rr_b1",
    }


def materialize_adversarial_agent_configs(config: ExperimentConfig) -> list[AgentConfig]:
    adversarial_config = config.adversarial_mas
    total_agents = adversarial_config.total_agents or len(adversarial_config.agents)
    if total_agents < 1:
        raise ValueError("adversarial_mas.total_agents must be at least 1.")
    configured_by_id = {agent.agent_id: agent for agent in adversarial_config.agents}
    agent_ids = _agent_id_sequence(total_agents)
    unexpected_ids = sorted(set(configured_by_id) - set(agent_ids))
    if unexpected_ids:
        raise ValueError(
            "adversarial_mas.agents contains ids outside total_agents sequence: "
            f"{unexpected_ids}"
        )
    return [
        configured_by_id.get(agent_id, AgentConfig(agent_id=agent_id))
        for agent_id in agent_ids
    ]


def resolve_adversarial_agent_ids(config: ExperimentConfig) -> set[str]:
    agent_configs = materialize_adversarial_agent_configs(config)
    available_ids = [agent.agent_id for agent in agent_configs]
    adversarial_config = config.adversarial_mas
    if adversarial_config.adversarial_agent_ids is not None:
        selected = set(adversarial_config.adversarial_agent_ids)
        unknown_ids = sorted(selected - set(available_ids))
        if unknown_ids:
            raise ValueError(
                "adversarial_mas.adversarial_agent_ids contains unknown ids: "
                f"{unknown_ids}"
            )
        return selected
    if adversarial_config.adversarial_count > len(available_ids):
        raise ValueError("adversarial_mas.adversarial_count must not exceed total_agents.")
    return set(available_ids[: adversarial_config.adversarial_count])


@dataclass(slots=True)
class AdversarialCompetitionRunner(CompetitionRunner):
    _strategy8_focal_by_question: dict[str, list[str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def _build_agents(self, experiment_config: ExperimentConfig | None = None) -> list[MasAgent]:
        config = experiment_config or self.experiment_config
        if self.agent_factory is not None:
            return self.agent_factory(config, self.llm_settings)
        client = build_structured_client(self.llm_settings)
        adversarial_ids = resolve_adversarial_agent_ids(config)
        normal_default_model = (
            config.adversarial_mas.normal_agent_model or self.llm_settings.model
        )
        enable_honest_answer_repair = _enable_honest_answer_repair(config)
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
                        debug_mode=self.debug_mode,
                    )
                )
            else:
                agents.append(
                    AdversarialHonestMasAgent(
                        agent_config=agent_config,
                        experiment_config=config,
                        client=client,
                        default_model=normal_default_model,
                        debug_mode=self.debug_mode,
                        enable_answer_repair=enable_honest_answer_repair,
                    )
                )
        return agents

    def _default_run_id(self, experiment_config: ExperimentConfig | None = None) -> str:
        config = experiment_config or self.experiment_config
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{_effective_adversarial_run_prefix(config)}-{timestamp}"

    def _resume_agent_ids(self, experiment_config: ExperimentConfig) -> list[str]:
        return [
            agent.agent_id
            for agent in materialize_adversarial_agent_configs(experiment_config)
        ]

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
    ):
        self._configure_strategy8_review_contexts(
            agents=agents,
            question_record=question_record,
            answers=answers,
            round_index=round_index,
        )
        return CompetitionRunner._collect_reviews_and_scores(
            self,
            agents=agents,
            question_record=question_record,
            answers=answers,
            round_index=round_index,
            max_workers=max_workers,
            progress_callback=progress_callback,
            question_index=question_index,
            total_questions=total_questions,
            total_rounds=total_rounds,
        )

    def _configure_strategy8_review_contexts(
        self,
        *,
        agents: list[MasAgent],
        question_record: TaskRecord,
        answers: dict[str, object],
        round_index: int,
    ) -> None:
        if not isinstance(question_record, QuestionRecord):
            return
        strategy8_agents = [
            agent
            for agent in agents
            if isinstance(agent, AdversarialMasAgent)
            and agent.behavior_config.choice_attack_strategy == ATTACK_STRATEGY_FEW_SHOT_RERANK
        ]
        if not strategy8_agents:
            return
        typed_answers = {
            agent_id: answer
            for agent_id, answer in answers.items()
            if isinstance(answer, AnswerSubmission)
        }
        if not typed_answers:
            return
        current_focal, current_agents = most_popular_non_reference_option(
            question_record=question_record,
            answers_by_agent=typed_answers,
        )
        prior_focal = self._strategy8_focal_by_question.get(question_record.question_key, [])
        focal = current_focal
        if round_index > 1 and prior_focal:
            prior_count = sum(
                1
                for answer in typed_answers.values()
                if sorted(answer.selected_option_ids) == sorted(prior_focal)
            )
            current_count = len(current_agents)
            if prior_count and prior_count >= current_count:
                focal = prior_focal
        if not focal:
            return
        self._strategy8_focal_by_question[question_record.question_key] = list(focal)
        anchor_supporters = [
            {
                "agent_id": agent_id,
                "selected_option_ids": answer.selected_option_ids,
                "reasoning": answer.reasoning,
            }
            for agent_id, answer in sorted(typed_answers.items())
            if sorted(answer.selected_option_ids) == sorted(focal)
        ]
        for agent in strategy8_agents:
            agent.set_strategy8_review_context(
                question_record.question_key,
                Strategy8ReviewContext(
                    focal_option_ids=list(focal),
                    anchor_supporters=[
                        supporter
                        for supporter in anchor_supporters
                        if supporter["agent_id"] != agent.agent_id
                    ],
                ),
            )
