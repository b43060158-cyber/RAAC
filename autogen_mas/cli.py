from __future__ import annotations

import argparse
from copy import deepcopy
import http.client
import json
import os
from pathlib import Path
import random
import socket
import sys
import urllib.error
import urllib.request

from autogen_mas.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DashScopeSettings,
    QuestionSelectionConfig,
    _split_api_keys,
    load_env_file,
    load_settings,
)
from autogen_mas.dataset_adapters import (
    ChessAdapter,
    CIARAdapter,
    CIARChoiceAdapter,
    FairEvalAdapter,
    HumanEvalAdapter,
    MedMCQAAdapter,
    MMLUAdapter,
    SCALRAdapter,
    TruthfulQAAdapter,
)
from autogen_mas.evaluation import Evaluator
from autogen_mas.models import TaskRecord
from autogen_mas.persistence import JsonRunStore
from autogen_mas.runtime import (
    AdversarialCompetitionRunner,
    BatchRuntimeOptions,
    BatchSingleAgentRunner,
    ChessCompetitionRunner,
    CompetitionRunner,
    SingleAgentRunner,
)
from autogen_mas.runtime.adversarial import (
    materialize_adversarial_agent_configs,
    resolve_adversarial_agent_ids,
)
from autogen_mas.runtime.clients import _provider_hint_for_model
from autogen_mas.runtime.adversarial.few_shot import strategy8_sample_question_keys
from autogen_mas.runtime.adversarial.reasoning_bank import (
    canonical_bank_dataset_name,
    reasoning_bank_question_keys,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autogen-mas")
    parser.add_argument("--config", default="config/agents.yaml")
    parser.add_argument("--env-file", default=".env")

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_models = subparsers.add_parser("list-models")
    _add_llm_overrides(list_models)

    run_question = subparsers.add_parser("run-question")
    run_question.add_argument("--dataset", default="truthfulqa")
    run_question.add_argument("--dataset-path")
    run_question.add_argument("--question-key")
    run_question.add_argument("--question-index", type=int)
    run_question.add_argument("--debug", action="store_true")
    run_question.add_argument("--persist-feedback-summary", action="store_true")
    _add_alignment_judge_override(run_question)
    _add_consensus_short_circuit_override(run_question)
    _add_llm_overrides(run_question)

    run_adversarial_question = subparsers.add_parser("run-adversarial-question")
    run_adversarial_question.add_argument("--dataset", default="truthfulqa")
    run_adversarial_question.add_argument("--dataset-path")
    run_adversarial_question.add_argument("--question-key")
    run_adversarial_question.add_argument("--question-index", type=int)
    run_adversarial_question.add_argument("--debug", action="store_true")
    run_adversarial_question.add_argument("--persist-feedback-summary", action="store_true")
    _add_alignment_judge_override(run_adversarial_question)
    _add_consensus_short_circuit_override(run_adversarial_question)
    _add_adversarial_overrides(run_adversarial_question)
    _add_llm_overrides(run_adversarial_question)

    run_dataset = subparsers.add_parser("run-dataset")
    run_dataset.add_argument("--dataset", default="truthfulqa")
    run_dataset.add_argument("--dataset-path")
    run_dataset.add_argument("--limit", type=int)
    run_dataset.add_argument("--seed", type=int)
    run_dataset.add_argument("--num-rounds", type=int)
    run_dataset.add_argument("--max-workers", type=int)
    run_dataset.add_argument("--resume-run-path")
    run_dataset.add_argument("--failed-questions-file")
    run_dataset.add_argument("--debug", action="store_true")
    run_dataset.add_argument("--persist-feedback-summary", action="store_true")
    run_dataset.add_argument(
        "--selection-rule",
        choices=["top-agent"],
        default="top-agent",
    )
    run_dataset.add_argument(
        "--skip-corpus-questions",
        "--skip-reasoning-bank-questions",
        dest="skip_corpus_questions",
        action="store_true",
        help=(
            "Exclude questions that already appear in the configured reasoning-bank corpus "
            "before applying --limit/--seed."
        ),
    )
    _add_alignment_judge_override(run_dataset)
    _add_consensus_short_circuit_override(run_dataset)
    _add_llm_overrides(run_dataset)

    run_adversarial_dataset = subparsers.add_parser("run-adversarial-dataset")
    run_adversarial_dataset.add_argument("--dataset", default="truthfulqa")
    run_adversarial_dataset.add_argument("--dataset-path")
    run_adversarial_dataset.add_argument("--limit", type=int)
    run_adversarial_dataset.add_argument("--seed", type=int)
    run_adversarial_dataset.add_argument("--num-rounds", type=int)
    run_adversarial_dataset.add_argument("--max-workers", type=int)
    run_adversarial_dataset.add_argument("--resume-run-path")
    run_adversarial_dataset.add_argument("--failed-questions-file")
    run_adversarial_dataset.add_argument("--debug", action="store_true")
    run_adversarial_dataset.add_argument("--persist-feedback-summary", action="store_true")
    run_adversarial_dataset.add_argument(
        "--selection-rule",
        choices=["top-agent"],
        default="top-agent",
    )
    run_adversarial_dataset.add_argument(
        "--skip-corpus-questions",
        "--skip-reasoning-bank-questions",
        dest="skip_corpus_questions",
        action="store_true",
        help=(
            "Exclude questions that already appear in the configured reasoning-bank corpus "
            "before applying --limit/--seed."
        ),
    )
    _add_alignment_judge_override(run_adversarial_dataset)
    _add_consensus_short_circuit_override(run_adversarial_dataset)
    _add_adversarial_overrides(run_adversarial_dataset)
    _add_llm_overrides(run_adversarial_dataset)

    run_single_agent = subparsers.add_parser("run-single-agent")
    run_single_agent.add_argument("--dataset", default="truthfulqa")
    run_single_agent.add_argument("--dataset-path")
    run_single_agent.add_argument("--limit", type=int)
    run_single_agent.add_argument("--seed", type=int)
    run_single_agent.add_argument("--num-rounds", type=int)
    run_single_agent.add_argument(
        "--self-reflection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the current single-agent iterative baseline. "
            "When disabled, single-agent runs answer once with no self-review."
        ),
    )
    run_single_agent.add_argument("--batched", action="store_true")
    run_single_agent.add_argument("--answer-batch-tokens", type=int)
    run_single_agent.add_argument("--review-batch-tokens", type=int)
    run_single_agent.add_argument("--answer-batch-size", type=int)
    run_single_agent.add_argument("--review-batch-size", type=int)
    run_single_agent.add_argument("--batch-timeout-seconds", type=int)
    run_single_agent.add_argument("--single-question-max-attempts", type=int)
    run_single_agent.add_argument("--single-question-retry-delay-seconds", type=float)
    run_single_agent.add_argument("--max-workers", type=int)
    run_single_agent.add_argument("--resume-run-path")
    _add_alignment_judge_override(run_single_agent)
    _add_llm_overrides(run_single_agent)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-path", required=True)
    evaluate.add_argument(
        "--selection-rule",
        choices=["top-agent"],
        default="top-agent",
    )

    evaluate_round_sweep = subparsers.add_parser("evaluate-round-sweep")
    evaluate_round_sweep.add_argument("--run-path", required=True)
    evaluate_round_sweep.add_argument("--max-round", type=int)
    evaluate_round_sweep.add_argument(
        "--selection-rule",
        choices=["top-agent"],
        default="top-agent",
    )

    return parser


def _add_llm_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-provider",
        choices=["dashscope", "deepseek", "anthropic", "ollama"],
        help="Override the active LLM profile for this run.",
    )
    parser.add_argument("--llm-model", help="Override the active default model.")
    parser.add_argument("--llm-base-url", help="Override the active provider base URL.")
    parser.add_argument("--llm-api-key", help="Override the active provider API key.")
    parser.add_argument("--llm-client-backend", help="Override the LLM client backend.")
    parser.add_argument("--llm-timeout-seconds", type=int, help="Override the LLM timeout.")
    parser.add_argument("--llm-max-retries", type=int, help="Override the LLM retry count.")


def _add_alignment_judge_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--alignment-judge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable the reasoning-answer alignment judge for this run. "
            "When disabled, structured answers are kept without the extra alignment-judge pass."
        ),
    )


def _add_consensus_short_circuit_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--consensus-short-circuit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable stopping a question early once all agents reach "
            "the same answer in a supported task."
        ),
    )


def _add_adversarial_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--total-agents", type=int)
    parser.add_argument("--adversarial-count", type=int)
    parser.add_argument("--adversarial-agent-ids")
    parser.add_argument("--normal-agent-model")
    parser.add_argument("--adversarial-model")
    parser.add_argument("--adversarial-answer-prompt")
    parser.add_argument("--adversarial-review-prompt")
    parser.add_argument(
        "--attack-strategy",
        "--adversarial-attack-strategy",
        dest="attack_strategy",
        help=(
            "Hot-swap adversarial method: 0=comment lines, "
            "7=cognitive manipulation (fabricated academic authority attacks), "
            "8=few-shot rerank for choice questions, "
            "11=Full RAAC for CIAR short-answer, "
            "11-1/11-2/11-3=RAAC ablations, "
            "12=RAAC + short-answer reranker best-of-n, "
            "13=fusion_rr_b1 for choice-question persuasion. "
            "Numeric 8 leaves short-answer/code strategies at their defaults."
        ),
    )
    parser.add_argument(
        "--attack-intensity",
        "--adversarial-attack-intensity",
        dest="attack_intensity",
        choices=["low", "medium", "high"],
        help="Strength of semantic code mutation for attack strategies 0 and 7.",
    )
    parser.add_argument("--adversarial-code-comment-lines", type=int)
    parser.add_argument(
        "--reasoning-bank-scorer",
        dest="reasoning_bank_scorer",
        help=(
            "Hot-swap the reasoning-bank per-record scorer for every strategy "
            "(e.g. lexical_overlap_v1, situational_transfer_v1)."
        ),
    )
    parser.add_argument(
        "--reasoning-bank-selector",
        dest="reasoning_bank_selector",
        help=(
            "Hot-swap the reasoning-bank top-k selector for every strategy "
            "(e.g. topk_sorted_v1, threshold_topk_v1, mmr_diversity_v1)."
        ),
    )
    parser.add_argument(
        "--reasoning-bank-retrieval-param",
        dest="reasoning_bank_retrieval_params",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Set a retrieval scorer/selector param (repeatable), e.g. "
            "--reasoning-bank-retrieval-param mmr_lambda=0.7 "
            "--reasoning-bank-retrieval-param min_score=0.4."
        ),
    )
    parser.add_argument(
        "--reasoning-bank-allow-cross-dataset",
        dest="reasoning_bank_allow_cross_dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Permit cross-dataset retrieval fallback (default off).",
    )
    parser.add_argument(
        "--reasoning-bank-cross-family-fallback",
        dest="reasoning_bank_allow_cross_family_fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Permit cross model-family corpus fallback (default on).",
    )


def _parse_retrieval_param(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        raise ValueError(
            f"--reasoning-bank-retrieval-param expects KEY=VALUE, got {raw!r}."
        )
    key, _, value = raw.partition("=")
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"--reasoning-bank-retrieval-param has empty key in {raw!r}.")
    lowered = value.casefold()
    if lowered in {"true", "false"}:
        return key, lowered == "true"
    for caster in (int, float):
        try:
            return key, caster(value)
        except ValueError:
            continue
    return key, value


_DATASET_ADAPTERS = {
    "chess": ChessAdapter,
    "ciar": CIARAdapter,
    "ciar_choice": CIARChoiceAdapter,
    "faireval": FairEvalAdapter,
    "humaneval": HumanEvalAdapter,
    "medmcqa": MedMCQAAdapter,
    "mmlu": MMLUAdapter,
    "scalr": SCALRAdapter,
    "truthfulqa": TruthfulQAAdapter,
}


def _load_questions(dataset_name: str, dataset_path: str) -> list[TaskRecord]:
    if dataset_name not in _DATASET_ADAPTERS:
        supported = ", ".join(sorted(_DATASET_ADAPTERS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}.")
    return _DATASET_ADAPTERS[dataset_name](dataset_path).load()


def _load_truthfulqa_questions(dataset_path: str) -> list[TaskRecord]:
    return _load_questions("truthfulqa", dataset_path)


def _competition_runner_class_for_dataset(dataset_name: str):
    if dataset_name == "chess":
        return ChessCompetitionRunner
    return CompetitionRunner


def _resolve_dataset_path(experiment_config, dataset_name: str, explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    if dataset_name not in experiment_config.datasets:
        raise ValueError(f"Dataset path for '{dataset_name}' is not configured.")
    return experiment_config.datasets[dataset_name]


def _select_questions(
    questions: list[TaskRecord],
    *,
    limit: int | None,
    seed: int | None,
) -> tuple[list[TaskRecord], QuestionSelectionConfig]:
    if limit is None:
        return questions, QuestionSelectionConfig(
            limit=None,
            seed=seed,
            strategy="all",
        )
    if limit < 0:
        raise ValueError("--limit must be non-negative.")
    if limit > len(questions):
        raise ValueError(f"--limit {limit} exceeds dataset size {len(questions)}.")
    if seed is None:
        return questions[:limit], QuestionSelectionConfig(
            limit=limit,
            seed=None,
            strategy="first_n",
        )

    rng = random.Random(seed)
    selected_indexes = sorted(rng.sample(range(len(questions)), limit))
    return [questions[index] for index in selected_indexes], QuestionSelectionConfig(
        limit=limit,
        seed=seed,
        strategy="random_sample_original_order",
    )


def _with_question_selection(
    experiment_config,
    selection: QuestionSelectionConfig,
):
    config = deepcopy(experiment_config)
    config.question_selection = selection
    return config


def _exclude_strategy8_sample_questions(
    questions: list[TaskRecord],
    experiment_config,
) -> list[TaskRecord]:
    behavior_config = experiment_config.adversarial_mas.adversarial_agent
    if behavior_config.choice_attack_strategy != "few_shot_rerank":
        return questions
    excluded_keys: set[str] = set()
    for dataset_name in {question.dataset_name for question in questions}:
        excluded_keys.update(strategy8_sample_question_keys(dataset_name))
    if not excluded_keys:
        return questions
    return [question for question in questions if question.question_key not in excluded_keys]


def _exclude_reasoning_bank_questions(
    questions: list[TaskRecord],
    experiment_config,
) -> list[TaskRecord]:
    behavior_config = experiment_config.adversarial_mas.adversarial_agent
    run_dataset_name = questions[0].dataset_name if questions else None
    bank_dataset_name = canonical_bank_dataset_name(run_dataset_name)
    excluded_keys = reasoning_bank_question_keys(
        behavior_config=behavior_config,
        dataset_name=bank_dataset_name,
    )
    if not excluded_keys:
        return questions
    # When the run-time dataset is stored in the bank under a different
    # (canonical) name, bank question keys carry the canonical prefix
    # (e.g. "ciar__0001__single_choice"). Remap the prefix back to the
    # run-time name so it matches each question's own question_key.
    if run_dataset_name and bank_dataset_name and run_dataset_name != bank_dataset_name:
        old_prefix = f"{bank_dataset_name}__"
        new_prefix = f"{run_dataset_name}__"
        excluded_keys = {
            new_prefix + key[len(old_prefix):] if key.startswith(old_prefix) else key
            for key in excluded_keys
        }
    return [question for question in questions if question.question_key not in excluded_keys]


def _load_failed_question_filters(
    path: str | os.PathLike[str],
) -> tuple[set[int], set[str], set[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    index_filters: set[int] = set()
    id_filters: set[str] = set()
    key_filters: set[str] = set()
    if not isinstance(payload, list):
        raise ValueError("--failed-questions-file must contain a JSON list.")
    for item in payload:
        if isinstance(item, int):
            index_filters.add(item)
            continue
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            if text.isdigit():
                index_filters.add(int(text))
            else:
                key_filters.add(text)
            continue
        if not isinstance(item, dict):
            continue
        raw_index = item.get("question_index")
        if isinstance(raw_index, int):
            index_filters.add(raw_index)
        elif isinstance(raw_index, str) and raw_index.strip().isdigit():
            index_filters.add(int(raw_index.strip()))
        raw_id = str(item.get("question_id", "")).strip()
        if raw_id:
            id_filters.add(raw_id)
        raw_key = str(item.get("question_key", "")).strip()
        if raw_key:
            key_filters.add(raw_key)
    return index_filters, id_filters, key_filters


def _filter_questions_by_failed_questions_file(
    questions: list[TaskRecord],
    failed_questions_file: str | None,
) -> list[TaskRecord]:
    if not failed_questions_file:
        return questions
    index_filters, id_filters, key_filters = _load_failed_question_filters(failed_questions_file)
    if not index_filters and not id_filters and not key_filters:
        return []
    # When a failed-questions payload includes stable identifiers from a prior run,
    # prefer those over question_index. Run-local indexes do not necessarily map back
    # to positions in the full dataset.
    if id_filters or key_filters:
        return [
            question
            for question in questions
            if question.question_key in key_filters or question.question_id in id_filters
        ]
    filtered = [
        question
        for index, question in enumerate(questions, start=1)
        if index in index_filters
    ]
    return filtered


def _with_max_workers(experiment_config, max_workers: int | None):
    if max_workers is None:
        return experiment_config
    if max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    config = deepcopy(experiment_config)
    config.runtime.max_workers = max_workers
    return config


def _with_num_rounds(experiment_config, num_rounds: int | None):
    if num_rounds is None:
        return experiment_config
    if num_rounds < 1:
        raise ValueError("--num-rounds must be at least 1.")
    config = deepcopy(experiment_config)
    config.runtime.num_rounds = num_rounds
    return config


def _with_single_agent_reflection(experiment_config, enabled: bool | None):
    if enabled is None:
        return experiment_config
    config = deepcopy(experiment_config)
    config.single_agent.enable_self_reflection = enabled
    if not enabled:
        config.runtime.num_rounds = 1
    return config


def _with_alignment_judge(experiment_config, enabled: bool | None):
    if enabled is None:
        return experiment_config
    config = deepcopy(experiment_config)
    config.alignment_judge.enabled = enabled
    return config


def _with_consensus_short_circuit(experiment_config, enabled: bool | None):
    if enabled is None:
        return experiment_config
    config = deepcopy(experiment_config)
    config.runtime.consensus_short_circuit = enabled
    return config


def _with_baseline_alignment_judge_default(experiment_config, enabled: bool | None):
    if enabled is not None:
        return _with_alignment_judge(experiment_config, enabled)
    if experiment_config.adversarial_mas.adversarial_agent.choice_attack_strategy != "baseline_1":
        return experiment_config
    config = deepcopy(experiment_config)
    config.alignment_judge.enabled = False
    return config


def _parse_agent_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [value.strip() for value in raw.split(",")]
    return [value for value in values if value]


def _persist_feedback_summary_enabled(args, parser: argparse.ArgumentParser) -> bool:
    enabled = bool(getattr(args, "persist_feedback_summary", False))
    if enabled and not bool(getattr(args, "debug", False)):
        parser.error("--persist-feedback-summary requires --debug.")
    return enabled


def _with_adversarial_overrides(experiment_config, args):
    config = deepcopy(experiment_config)
    adversarial_config = config.adversarial_mas
    if getattr(args, "total_agents", None) is not None:
        if args.total_agents < 1:
            raise ValueError("--total-agents must be at least 1.")
        adversarial_config.total_agents = args.total_agents
    if getattr(args, "adversarial_count", None) is not None:
        if args.adversarial_count < 0:
            raise ValueError("--adversarial-count must be non-negative.")
        adversarial_config.adversarial_count = args.adversarial_count
        if args.adversarial_count == 0:
            adversarial_config.adversarial_agent_ids = None
    if getattr(args, "adversarial_agent_ids", None) is not None:
        adversarial_config.adversarial_agent_ids = _parse_agent_ids(
            args.adversarial_agent_ids
        )
        adversarial_config.adversarial_count = len(
            adversarial_config.adversarial_agent_ids or []
        )
    if getattr(args, "normal_agent_model", None) is not None:
        adversarial_config.normal_agent_model = args.normal_agent_model
    behavior_config = adversarial_config.adversarial_agent
    if getattr(args, "adversarial_model", None) is not None:
        behavior_config.model = args.adversarial_model
    if getattr(args, "adversarial_answer_prompt", None) is not None:
        behavior_config.answer_prompt = args.adversarial_answer_prompt
    if getattr(args, "adversarial_review_prompt", None) is not None:
        behavior_config.review_prompt = args.adversarial_review_prompt
    if getattr(args, "attack_strategy", None) is not None:
        behavior_config.attack_strategy = args.attack_strategy
    if getattr(args, "attack_intensity", None) is not None:
        behavior_config.attack_intensity = args.attack_intensity
    if getattr(args, "adversarial_code_comment_lines", None) is not None:
        if args.adversarial_code_comment_lines < 0:
            raise ValueError("--adversarial-code-comment-lines must be non-negative.")
        behavior_config.code_comment_lines = args.adversarial_code_comment_lines
    if getattr(args, "reasoning_bank_scorer", None) is not None:
        behavior_config.reasoning_bank_scorer = args.reasoning_bank_scorer
    if getattr(args, "reasoning_bank_selector", None) is not None:
        behavior_config.reasoning_bank_selector = args.reasoning_bank_selector
    if getattr(args, "reasoning_bank_retrieval_params", None):
        merged = dict(behavior_config.reasoning_bank_retrieval_params)
        for raw in args.reasoning_bank_retrieval_params:
            key, value = _parse_retrieval_param(raw)
            merged[key] = value
        behavior_config.reasoning_bank_retrieval_params = merged
    if getattr(args, "reasoning_bank_allow_cross_dataset", None) is not None:
        behavior_config.reasoning_bank_allow_cross_dataset = (
            args.reasoning_bank_allow_cross_dataset
        )
    if getattr(args, "reasoning_bank_allow_cross_family_fallback", None) is not None:
        behavior_config.reasoning_bank_allow_cross_family_fallback = (
            args.reasoning_bank_allow_cross_family_fallback
        )
    adversarial_config.validate()
    return config


def _default_llm_settings_for_provider(provider: str) -> DashScopeSettings:
    if provider == "ollama":
        return DashScopeSettings(
            api_key="any",
            base_url=DEFAULT_OLLAMA_BASE_URL,
            model="llama3.1",
            client_backend="direct_http",
            provider="ollama",
        )
    raise ValueError(
        f"Provider '{provider}' is not configured in the current .env; "
        "pass explicit --llm-base-url/--llm-model values or add a profile first."
    )


def _all_llm_profiles(llm_settings) -> dict[str, DashScopeSettings]:
    profiles = {
        llm_settings.provider: deepcopy(llm_settings),
        **{provider: deepcopy(route) for provider, route in llm_settings.model_routes.items()},
    }
    for profile in profiles.values():
        profile.model_routes = {}
    for provider, profile in profiles.items():
        profile.model_routes = {
            other_provider: deepcopy(other_profile)
            for other_provider, other_profile in profiles.items()
            if other_provider != provider
        }
    return profiles


def _select_llm_profile(llm_settings, provider: str) -> DashScopeSettings:
    profiles = _all_llm_profiles(llm_settings)
    selected = profiles.get(provider)
    if selected is not None:
        return selected
    return _default_llm_settings_for_provider(provider)


def _with_llm_overrides(llm_settings, args):
    override_fields = (
        "llm_provider",
        "llm_model",
        "llm_base_url",
        "llm_api_key",
        "llm_client_backend",
        "llm_timeout_seconds",
        "llm_max_retries",
    )
    if not any(getattr(args, field, None) is not None for field in override_fields):
        return llm_settings
    settings = (
        _select_llm_profile(llm_settings, args.llm_provider)
        if getattr(args, "llm_provider", None)
        else deepcopy(llm_settings)
    )
    if any(
        getattr(args, field, None) is not None
        for field in ("llm_provider", "llm_model", "llm_base_url")
    ):
        settings.model_routes = {}
    if getattr(args, "llm_provider", None) is not None:
        settings.provider = args.llm_provider
    if getattr(args, "llm_model", None) is not None:
        settings.model = args.llm_model
    if getattr(args, "llm_base_url", None) is not None:
        settings.base_url = args.llm_base_url
    if getattr(args, "llm_api_key", None) is not None:
        override_keys = _split_api_keys(args.llm_api_key)
        settings.api_keys = override_keys or [args.llm_api_key]
        settings.api_key = settings.api_keys[0]
    if getattr(args, "llm_client_backend", None) is not None:
        settings.client_backend = args.llm_client_backend
    if getattr(args, "llm_timeout_seconds", None) is not None:
        if args.llm_timeout_seconds < 1:
            raise ValueError("--llm-timeout-seconds must be at least 1.")
        settings.timeout_seconds = args.llm_timeout_seconds
    if getattr(args, "llm_max_retries", None) is not None:
        if args.llm_max_retries < 0:
            raise ValueError("--llm-max-retries must be non-negative.")
        settings.max_retries = args.llm_max_retries
    if settings.provider == "ollama" and not settings.api_key:
        settings.api_key = "any"
    return settings


def _resolve_model_endpoint(llm_settings, model: str) -> tuple[str, str]:
    route_models = {
        provider: route.model for provider, route in llm_settings.model_routes.items()
    }
    if model == llm_settings.model:
        return llm_settings.provider, llm_settings.base_url
    for route in llm_settings.model_routes.values():
        if model == route.model:
            return route.provider, route.base_url
    provider_hint = _provider_hint_for_model(model, route_models=route_models)
    if provider_hint == llm_settings.provider:
        return llm_settings.provider, llm_settings.base_url
    route = llm_settings.model_routes.get(provider_hint or "")
    if route is not None:
        return route.provider, route.base_url
    return llm_settings.provider, llm_settings.base_url


def _models_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized + "/models"
    return normalized + "/v1/models"


def _list_models(llm_settings) -> list[str]:
    if llm_settings.provider != "anthropic":
        raise ValueError(
            "list-models currently supports Anthropic-compatible providers only."
        )
    request = urllib.request.Request(
        _models_endpoint(llm_settings.base_url),
        headers={
            "Content-Type": "application/json",
            "x-api-key": llm_settings.api_key or "any",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=llm_settings.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            detail = error.read().decode("utf-8")[:300]
        except Exception:
            detail = ""
        suffix = f": {detail}" if detail else ""
        raise ValueError(
            f"Failed to query models: HTTP {error.code} {error.reason}{suffix}"
        ) from error
    except (
        http.client.RemoteDisconnected,
        urllib.error.URLError,
        socket.timeout,
        TimeoutError,
    ) as error:
        raise ValueError(f"Failed to query models: {error}") from error

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Model list returned invalid JSON: {response_body[:200]!r}"
        ) from error

    raw_models = payload if isinstance(payload, list) else payload.get("data") or payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("Model list response is missing a model array.")
    models: list[str] = []
    for item in raw_models:
        if isinstance(item, str):
            model_id = item.strip()
        elif isinstance(item, dict):
            model_id = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
        else:
            model_id = ""
        if model_id:
            models.append(model_id)
    if not models:
        raise ValueError("Model list response did not contain any model ids.")
    return sorted(dict.fromkeys(models))


def _emit_adversarial_model_plan(experiment_config, llm_settings) -> None:
    adversarial_ids = resolve_adversarial_agent_ids(experiment_config)
    lines = ["[model-plan] adversarial LLM-MAS routing"]
    for agent_config in materialize_adversarial_agent_configs(experiment_config):
        role = "adversarial" if agent_config.agent_id in adversarial_ids else "honest"
        if role == "adversarial":
            model = (
                experiment_config.adversarial_mas.adversarial_agent.model
                or agent_config.model
                or llm_settings.model
            )
        else:
            model = (
                agent_config.model
                or experiment_config.adversarial_mas.normal_agent_model
                or llm_settings.model
            )
        provider, base_url = _resolve_model_endpoint(llm_settings, model)
        lines.append(
            f"[model-plan] {agent_config.agent_id} role={role} "
            f"model={model} provider={provider} base_url={base_url}"
        )
    print("\n".join(lines), file=sys.stderr, flush=True)


def _build_question_progress(desc: str, total_questions: int):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return None, lambda: None

    progress = tqdm(total=total_questions, desc=desc, unit="question")

    def on_progress(index, total, question_record, stage):
        if stage == "started":
            progress.set_postfix_str(
                f"{question_record.question_key} ({index}/{total})",
                refresh=True,
            )
        elif stage == "completed":
            progress.update(1)
            progress.set_postfix_str(
                f"{question_record.question_key} completed ({index}/{total})",
                refresh=True,
            )
        else:
            progress.set_postfix_str(
                f"{question_record.question_key}: {stage}",
                refresh=True,
            )

    return on_progress, progress.close


def _build_round_progress(desc: str, total_rounds: int):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return None, lambda: None

    progress = tqdm(total=total_rounds, desc=desc, unit="round")
    completed_rounds: set[str] = set()

    def on_progress(index, total, question_record, stage):
        stage_parts = stage.split()
        if len(stage_parts) == 3 and stage_parts[0] == "round" and stage_parts[2] == "completed":
            round_label = stage_parts[1]
            if round_label not in completed_rounds:
                completed_rounds.add(round_label)
                progress.update(1)
        progress.set_postfix_str(f"{question_record.question_key}: {stage}", refresh=True)

    return on_progress, progress.close


def _build_batch_progress(desc: str):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return None, lambda: None

    progress = tqdm(total=0, desc=desc, unit="call")

    def on_progress(index, total, question_record, stage):
        stage_parts = stage.split()
        if (
            len(stage_parts) == 6
            and stage_parts[0] == "round"
            and stage_parts[2] in {"answer", "review"}
            and stage_parts[3] == "batches"
            and stage_parts[5] == "planned"
        ):
            round_label = stage_parts[1]
            stage_name = stage_parts[2]
            batch_count = int(stage_parts[4])
            progress.total = batch_count
            progress.n = 0
            progress.refresh()
            progress.set_postfix_str(
                f"round={round_label} stage={stage_name} batches={batch_count}",
                refresh=True,
            )
            return
        if (
            len(stage_parts) == 8
            and stage_parts[0] == "round"
            and stage_parts[2] in {"answer", "review"}
            and stage_parts[3] == "agent_batches"
            and stage_parts[5] == "question_batches"
            and stage_parts[7] == "planned"
        ):
            round_label = stage_parts[1]
            stage_name = stage_parts[2]
            agent_batch_count = int(stage_parts[4])
            question_batch_count = int(stage_parts[6])
            progress.total = agent_batch_count
            progress.n = 0
            progress.refresh()
            progress.set_postfix_str(
                (
                    f"round={round_label} stage={stage_name} "
                    f"question_batches={question_batch_count} api_calls={agent_batch_count}"
                ),
                refresh=True,
            )
            return
        elif (
            len(stage_parts) >= 7
            and stage_parts[0] == "round"
            and stage_parts[2] == "batch"
            and stage_parts[3] in {"answer", "review"}
        ):
            round_label = stage_parts[1]
            stage_name = stage_parts[3]
            agent_id = stage_parts[4]
            batch_size = stage_parts[5]
            status = stage_parts[-1]
            if status == "completed":
                progress.update(1)
            progress.set_postfix_str(
                (
                    f"round={round_label} stage={stage_name} agent={agent_id} "
                    f"batch_size={batch_size} status={status}"
                ),
                refresh=True,
            )
            return
        progress.set_postfix_str(f"{question_record.question_key}: {stage}", refresh=True)

    return on_progress, progress.close


def _batch_runtime_options(args, env_path: str = ".env") -> BatchRuntimeOptions:
    env_values = load_env_file(env_path)
    is_single_agent = getattr(args, "command", "") == "run-single-agent"
    default_answer_tokens = "4000" if is_single_agent else "12000"
    default_review_tokens = "3000" if is_single_agent else "12000"
    answer_tokens = args.answer_batch_tokens
    if answer_tokens is None:
        single_agent_answer_tokens = (
            os.environ.get("SINGLE_AGENT_ANSWER_BATCH_TOKENS")
            or env_values.get("SINGLE_AGENT_ANSWER_BATCH_TOKENS")
            if is_single_agent
            else None
        )
        answer_tokens = int(
            single_agent_answer_tokens
            or os.environ.get("ANSWER_BATCH_TOKENS")
            or env_values.get("ANSWER_BATCH_TOKENS")
            or default_answer_tokens
        )
    review_tokens = args.review_batch_tokens
    if review_tokens is None:
        single_agent_review_tokens = (
            os.environ.get("SINGLE_AGENT_REVIEW_BATCH_TOKENS")
            or env_values.get("SINGLE_AGENT_REVIEW_BATCH_TOKENS")
            if is_single_agent
            else None
        )
        review_tokens = int(
            single_agent_review_tokens
            or os.environ.get("REVIEW_BATCH_TOKENS")
            or env_values.get("REVIEW_BATCH_TOKENS")
            or default_review_tokens
        )
    answer_batch_size = args.answer_batch_size
    if answer_batch_size is None:
        answer_batch_size = int(
            os.environ.get("ANSWER_BATCH_SIZE")
            or env_values.get("ANSWER_BATCH_SIZE")
            or "8"
        )
    review_batch_size = args.review_batch_size
    if review_batch_size is None:
        review_batch_size = int(
            os.environ.get("REVIEW_BATCH_SIZE")
            or env_values.get("REVIEW_BATCH_SIZE")
            or "4"
        )
    batch_timeout_seconds = args.batch_timeout_seconds
    if batch_timeout_seconds is None:
        raw_batch_timeout = (
            os.environ.get("BATCH_TIMEOUT_SECONDS")
            or env_values.get("BATCH_TIMEOUT_SECONDS")
        )
        batch_timeout_seconds = int(raw_batch_timeout) if raw_batch_timeout else None
    single_question_max_attempts = getattr(args, "single_question_max_attempts", None)
    if single_question_max_attempts is None:
        raw_single_question_max_attempts = (
            os.environ.get("SINGLE_QUESTION_MAX_ATTEMPTS")
            or env_values.get("SINGLE_QUESTION_MAX_ATTEMPTS")
        )
        if raw_single_question_max_attempts:
            single_question_max_attempts = int(raw_single_question_max_attempts)
    single_question_retry_delay_seconds = getattr(
        args,
        "single_question_retry_delay_seconds",
        None,
    )
    if single_question_retry_delay_seconds is None:
        single_question_retry_delay_seconds = float(
            os.environ.get("SINGLE_QUESTION_RETRY_DELAY_SECONDS")
            or env_values.get("SINGLE_QUESTION_RETRY_DELAY_SECONDS")
            or "0"
        )
    if answer_tokens < 1:
        raise ValueError("--answer-batch-tokens must be at least 1.")
    if review_tokens < 1:
        raise ValueError("--review-batch-tokens must be at least 1.")
    if answer_batch_size < 1:
        raise ValueError("--answer-batch-size must be at least 1.")
    if review_batch_size < 1:
        raise ValueError("--review-batch-size must be at least 1.")
    if batch_timeout_seconds is not None and batch_timeout_seconds < 1:
        raise ValueError("--batch-timeout-seconds must be at least 1.")
    if single_question_max_attempts is not None and single_question_max_attempts < 1:
        raise ValueError("--single-question-max-attempts must be at least 1.")
    if single_question_retry_delay_seconds < 0:
        raise ValueError("--single-question-retry-delay-seconds must be non-negative.")
    return BatchRuntimeOptions(
        answer_max_input_tokens=answer_tokens,
        review_max_input_tokens=review_tokens,
        answer_max_questions_per_batch=answer_batch_size,
        review_max_questions_per_batch=review_batch_size,
        batch_timeout_seconds=batch_timeout_seconds,
        single_question_max_attempts=single_question_max_attempts,
        single_question_retry_delay_seconds=single_question_retry_delay_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    persist_feedback_summary = _persist_feedback_summary_enabled(args, parser)

    if args.command == "evaluate":
        report = Evaluator().evaluate(args.run_path, selection_rule=args.selection_rule)
        print(json.dumps(report.summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate-round-sweep":
        summary = Evaluator().evaluate_round_sweep(
            args.run_path,
            selection_rule=args.selection_rule,
            max_round=args.max_round,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    experiment_config, llm_settings = load_settings(args.config, args.env_file)
    try:
        llm_settings = _with_llm_overrides(llm_settings, args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "list-models":
        try:
            models = _list_models(llm_settings)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "provider": llm_settings.provider,
                    "base_url": llm_settings.base_url,
                    "models": models,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        dataset_path = _resolve_dataset_path(
            experiment_config=experiment_config,
            dataset_name=args.dataset,
            explicit_path=getattr(args, "dataset_path", None),
        )
        questions = _load_questions(args.dataset, dataset_path)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        questions = _filter_questions_by_failed_questions_file(
            questions,
            getattr(args, "failed_questions_file", None),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    if getattr(args, "failed_questions_file", None) and not questions:
        parser.error("--failed-questions-file did not match any questions.")

    if args.command == "run-question":
        experiment_config = _with_alignment_judge(
            experiment_config,
            getattr(args, "alignment_judge", None),
        )
        experiment_config = _with_consensus_short_circuit(
            experiment_config,
            getattr(args, "consensus_short_circuit", None),
        )
        runner = _competition_runner_class_for_dataset(args.dataset)(
            experiment_config=experiment_config,
            llm_settings=llm_settings,
            debug_mode=args.debug,
            persist_feedback_summary=persist_feedback_summary,
        )
        if args.question_key:
            question = next(
                item for item in questions if item.question_key == args.question_key
            )
        elif args.question_index is not None:
            question = questions[args.question_index]
        else:
            question = questions[0]
        result = runner.run_question(question)
        store = JsonRunStore(experiment_config.runtime.output_dir)
        run_path = store.initialize_run(
            run_id=result.run_id,
            experiment_config=experiment_config,
            llm_settings=llm_settings,
        )
        question_path = store.persist_question_result(
            run_path,
            result,
            persist_feedback_summary=persist_feedback_summary,
        )
        print(question_path)
        return 0

    if args.command == "run-adversarial-question":
        experiment_config = _with_adversarial_overrides(experiment_config, args)
        experiment_config = _with_consensus_short_circuit(
            experiment_config,
            getattr(args, "consensus_short_circuit", None),
        )
        experiment_config = _with_baseline_alignment_judge_default(
            experiment_config,
            getattr(args, "alignment_judge", None),
        )
        _emit_adversarial_model_plan(experiment_config, llm_settings)
        runner = AdversarialCompetitionRunner(
            experiment_config=experiment_config,
            llm_settings=llm_settings,
            debug_mode=args.debug,
            persist_feedback_summary=persist_feedback_summary,
        )
        if args.question_key:
            question = next(
                item for item in questions if item.question_key == args.question_key
            )
        elif args.question_index is not None:
            question = questions[args.question_index]
        else:
            question = questions[0]
        result = runner.run_question(question)
        store = JsonRunStore(experiment_config.runtime.output_dir)
        run_path = store.initialize_run(
            run_id=result.run_id,
            experiment_config=experiment_config,
            llm_settings=llm_settings,
        )
        question_path = store.persist_question_result(
            run_path,
            result,
            persist_feedback_summary=persist_feedback_summary,
        )
        print(question_path)
        return 0

    if args.command == "run-dataset":
        if args.skip_corpus_questions:
            questions = _exclude_reasoning_bank_questions(questions, experiment_config)
        questions, selection = _select_questions(
            questions,
            limit=args.limit,
            seed=args.seed,
        )
        experiment_config = _with_question_selection(experiment_config, selection)
        experiment_config = _with_num_rounds(experiment_config, args.num_rounds)
        experiment_config = _with_max_workers(experiment_config, args.max_workers)
        experiment_config = _with_alignment_judge(
            experiment_config,
            getattr(args, "alignment_judge", None),
        )
        experiment_config = _with_consensus_short_circuit(
            experiment_config,
            getattr(args, "consensus_short_circuit", None),
        )
        runner = _competition_runner_class_for_dataset(args.dataset)(
            experiment_config=experiment_config,
            llm_settings=llm_settings,
            debug_mode=args.debug,
            persist_feedback_summary=persist_feedback_summary,
        )
        progress_callback, close_progress = _build_question_progress(
            "LLM-MAS",
            len(questions),
        )
        try:
            artifacts = runner.run_dataset(
                questions,
                progress_callback=progress_callback,
                resume_run_path=args.resume_run_path,
            )
        finally:
            close_progress()
        report = Evaluator().evaluate(
            artifacts.run_path,
            selection_rule=args.selection_rule,
        )
        summary = {
            "run_path": artifacts.run_path,
            "failed_questions_path": str(
                JsonRunStore(experiment_config.runtime.output_dir).failed_questions_path(
                    artifacts.run_path
                )
            ),
            **report.summary,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-adversarial-dataset":
        experiment_config = _with_adversarial_overrides(experiment_config, args)
        experiment_config = _with_num_rounds(experiment_config, args.num_rounds)
        experiment_config = _with_consensus_short_circuit(
            experiment_config,
            getattr(args, "consensus_short_circuit", None),
        )
        experiment_config = _with_baseline_alignment_judge_default(
            experiment_config,
            getattr(args, "alignment_judge", None),
        )
        _emit_adversarial_model_plan(experiment_config, llm_settings)
        questions = _exclude_strategy8_sample_questions(questions, experiment_config)
        if args.skip_corpus_questions:
            questions = _exclude_reasoning_bank_questions(questions, experiment_config)
        questions, selection = _select_questions(
            questions,
            limit=args.limit,
            seed=args.seed,
        )
        experiment_config = _with_question_selection(experiment_config, selection)
        experiment_config = _with_max_workers(experiment_config, args.max_workers)
        runner = AdversarialCompetitionRunner(
            experiment_config=experiment_config,
            llm_settings=llm_settings,
            debug_mode=args.debug,
            persist_feedback_summary=persist_feedback_summary,
        )
        progress_callback, close_progress = _build_question_progress(
            "Adversarial LLM-MAS",
            len(questions),
        )
        try:
            artifacts = runner.run_dataset(
                questions,
                progress_callback=progress_callback,
                resume_run_path=args.resume_run_path,
            )
        finally:
            close_progress()
        report = Evaluator().evaluate(
            artifacts.run_path,
            selection_rule=args.selection_rule,
        )
        summary = {
            "run_path": artifacts.run_path,
            "failed_questions_path": str(
                JsonRunStore(experiment_config.runtime.output_dir).failed_questions_path(
                    artifacts.run_path
                )
            ),
            **report.summary,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-single-agent":
        questions, selection = _select_questions(
            questions,
            limit=args.limit,
            seed=args.seed,
        )
        experiment_config = _with_question_selection(experiment_config, selection)
        experiment_config = _with_num_rounds(experiment_config, args.num_rounds)
        experiment_config = _with_single_agent_reflection(
            experiment_config,
            args.self_reflection,
        )
        experiment_config = _with_max_workers(experiment_config, args.max_workers)
        experiment_config = _with_alignment_judge(
            experiment_config,
            getattr(args, "alignment_judge", None),
        )
        if args.batched:
            runner = BatchSingleAgentRunner(
                experiment_config=experiment_config,
                llm_settings=llm_settings,
                batch_options=_batch_runtime_options(args, args.env_file),
            )
            progress_callback, close_progress = _build_batch_progress("Batched single agent")
        else:
            runner = SingleAgentRunner(
                experiment_config=experiment_config,
                llm_settings=llm_settings,
            )
            progress_callback, close_progress = _build_question_progress(
                "Single agent",
                len(questions),
            )
        try:
            artifacts = runner.run_dataset(
                questions,
                progress_callback=progress_callback,
                resume_run_path=args.resume_run_path,
            )
        finally:
            close_progress()
        summary = Evaluator().evaluate_single_agent(
            artifacts.run_path,
            agent_id=experiment_config.single_agent.agent_id,
            model=experiment_config.single_agent.model or llm_settings.model,
            temperature=experiment_config.single_agent.temperature,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
