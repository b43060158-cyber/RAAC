from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autogen_mas.config import (  # noqa: E402
    DashScopeSettings,
    DriverAttributionConfig,
    load_dashscope_settings,
    load_experiment_config,
)
from autogen_mas.runtime.clients import StructuredLLMClient, build_structured_client  # noqa: E402


DEFAULT_OUTPUT_NAME = "changed_answer_summary.json"
NOT_EVALUATED = "not_evaluated"
EXTERNAL_DRIVER_FOUND = "external_driver_found"
SELF_DECISION_CHANGE = "self_decision_change"
UNKNOWN_CAUSE = "unknown"


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _new_bucket() -> dict[str, Any]:
    return {
        "answer_submissions": 0,
        "change_opportunities": 0,
        "self_reported_changed_answer_count": 0,
        "changed_answer_count": 0,
        "actual_option_changed_count": 0,
        "actual_code_changed_count": 0,
        "actual_answer_changed_count": 0,
        "both_self_reported_and_actual_count": 0,
        "self_reported_only_count": 0,
        "actual_only_count": 0,
        "self_report_actual_agreement_count": 0,
        "self_report_actual_disagreement_count": 0,
        "changed_answer_rate_all_submissions": 0.0,
        "changed_answer_rate_after_prior_answer": 0.0,
        "actual_option_changed_rate_after_prior_answer": 0.0,
        "actual_code_changed_rate_after_prior_answer": 0.0,
        "actual_answer_changed_rate_after_prior_answer": 0.0,
        "self_report_actual_agreement_rate": 0.0,
    }


def _update_rates(bucket: dict[str, Any]) -> None:
    changed = int(bucket["self_reported_changed_answer_count"])
    submissions = int(bucket["answer_submissions"])
    opportunities = int(bucket["change_opportunities"])
    bucket["changed_answer_count"] = changed
    bucket["changed_answer_rate_all_submissions"] = _rate(changed, submissions)
    bucket["changed_answer_rate_after_prior_answer"] = _rate(changed, opportunities)
    bucket["actual_option_changed_rate_after_prior_answer"] = _rate(
        int(bucket["actual_option_changed_count"]),
        opportunities,
    )
    bucket["actual_code_changed_rate_after_prior_answer"] = _rate(
        int(bucket["actual_code_changed_count"]),
        opportunities,
    )
    bucket["actual_answer_changed_rate_after_prior_answer"] = _rate(
        int(bucket["actual_answer_changed_count"]),
        opportunities,
    )
    bucket["self_report_actual_agreement_rate"] = _rate(
        int(bucket["self_report_actual_agreement_count"]),
        opportunities,
    )


def _sorted_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rounds, key=lambda item: int(item.get("round_index", 0)))


def _as_option_ids(answer: dict[str, Any] | None) -> list[str]:
    if not answer:
        return []
    return [str(option_id) for option_id in answer.get("selected_option_ids", [])]


def _same_option_ids(left: list[str], right: list[str]) -> bool:
    return sorted(left) == sorted(right)


def _as_code(answer: dict[str, Any] | None) -> str:
    if not answer:
        return ""
    return str(answer.get("code", "") or "")


def _normalize_code(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def _code_hash(code: str) -> str:
    return hashlib.sha256(_normalize_code(code).encode("utf-8")).hexdigest()


def _same_code(left: str, right: str) -> bool:
    return _normalize_code(left) == _normalize_code(right)


def _code_diff(previous_code: str, current_code: str) -> str:
    previous_lines = _normalize_code(previous_code).splitlines()
    current_lines = _normalize_code(current_code).splitlines()
    return "\n".join(
        difflib.unified_diff(
            previous_lines,
            current_lines,
            fromfile="previous_code",
            tofile="current_code",
            lineterm="",
        )
    )


def _is_code_generation(payload: dict[str, Any]) -> bool:
    return str(payload.get("task_type", "")) == "code_generation"


def _record_submission(
    bucket: dict[str, Any],
    *,
    has_prior_answer: bool,
    self_reported_changed: bool,
    actual_option_changed: bool,
    actual_code_changed: bool,
    actual_answer_changed: bool,
) -> None:
    bucket["answer_submissions"] += 1
    if self_reported_changed:
        bucket["self_reported_changed_answer_count"] += 1
        bucket["changed_answer_count"] += 1

    if not has_prior_answer:
        return

    bucket["change_opportunities"] += 1
    if actual_option_changed:
        bucket["actual_option_changed_count"] += 1
    if actual_code_changed:
        bucket["actual_code_changed_count"] += 1
    if actual_answer_changed:
        bucket["actual_answer_changed_count"] += 1

    if self_reported_changed == actual_answer_changed:
        bucket["self_report_actual_agreement_count"] += 1
    else:
        bucket["self_report_actual_disagreement_count"] += 1

    if self_reported_changed and actual_answer_changed:
        bucket["both_self_reported_and_actual_count"] += 1
    elif self_reported_changed:
        bucket["self_reported_only_count"] += 1
    elif actual_answer_changed:
        bucket["actual_only_count"] += 1


def _load_question_payloads(run_path: Path) -> list[dict[str, Any]]:
    question_dir = run_path / "questions"
    if not question_dir.exists():
        raise FileNotFoundError(f"Question directory not found: {question_dir}")

    question_files = sorted(question_dir.glob("*.json"))
    if not question_files:
        raise FileNotFoundError(f"No question result JSON files found in: {question_dir}")

    return [json.loads(question_file.read_text(encoding="utf-8")) for question_file in question_files]


def _latest_run_path(runs_dir: Path) -> Path:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "questions").exists() and any((path / "questions").glob("*.json"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed runs with question results found in: {runs_dir}")
    return sorted(candidates, key=_run_sort_key)[-1]


def _run_sort_key(path: Path) -> tuple[datetime, str]:
    match = re.search(r"(\d{8}-\d{6})", path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d-%H%M%S"), path.name
    return datetime.fromtimestamp(path.stat().st_mtime), path.name


def _agent_id(agent_result: dict[str, Any]) -> str:
    answer = dict(agent_result.get("answer") or {})
    return str(agent_result.get("agent_id") or answer.get("agent_id") or "unknown")


def _round_agent_map(round_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {_agent_id(agent_result): agent_result for agent_result in round_payload.get("agent_results", [])}


def _clean_agent_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        agent_id = str(value).strip()
        if agent_id and agent_id not in seen:
            seen.add(agent_id)
            result.append(agent_id)
    return result


def _review_id(question_key: str, round_index: int, reviewer_agent_id: str, target_agent_id: str) -> str:
    return f"{question_key}:round{round_index}:{reviewer_agent_id}->{target_agent_id}"


def _previous_round_reviews(
    *,
    question_key: str,
    round_index: int,
    target_agent_id: str,
    agent_result: dict[str, Any],
) -> list[dict[str, Any]]:
    reviews = []
    for review in agent_result.get("received_reviews", []):
        reviewer_agent_id = str(review.get("reviewer_agent_id", ""))
        reviews.append(
            {
                "review_id": _review_id(
                    question_key,
                    round_index,
                    reviewer_agent_id,
                    target_agent_id,
                ),
                "reviewer_agent_id": reviewer_agent_id,
                "target_agent_id": str(review.get("target_agent_id", target_agent_id)),
                "stance": str(review.get("stance", "")),
                "score": int(review.get("score", 0)),
                "main_reason_quote": str(review.get("main_reason", "")),
            }
        )
    return reviews


def _question_context(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": str(payload.get("question_id", "")),
        "question_key": str(payload.get("question_key", "")),
        "dataset_name": str(payload.get("dataset_name", "")),
        "task_type": str(payload.get("task_type", "")),
        "question": str(payload.get("question", "")),
        "options": list(payload.get("options", [])),
    }


def _label_for_score(score: float, config: DriverAttributionConfig) -> str:
    if score >= config.strong_driver_threshold:
        return "strong_driver"
    if score >= config.plausible_driver_threshold:
        return "plausible_driver"
    if score > 0:
        return "weak_driver"
    return "no_driver"


def _step_ref(change: dict[str, Any]) -> str:
    return f"{change['from_round']}->{change['to_round']}"


def _change_id(question_key: str, agent_id: str, from_round: int, to_round: int) -> str:
    return f"{question_key}:{agent_id}:{from_round}->{to_round}"


def _build_stepwise_change(
    *,
    payload: dict[str, Any],
    target_agent_id: str,
    previous_round_index: int,
    current_round_index: int,
    previous_result: dict[str, Any],
    current_result: dict[str, Any],
) -> dict[str, Any]:
    question_key = str(payload.get("question_key") or payload.get("question_id") or "unknown")
    previous_answer = dict(previous_result.get("answer") or {})
    current_answer = dict(current_result.get("answer") or {})
    previous_code = _as_code(previous_answer)
    current_code = _as_code(current_answer)
    actual_option_changed = not _same_option_ids(
        _as_option_ids(previous_answer),
        _as_option_ids(current_answer),
    )
    actual_code_changed = _is_code_generation(payload) and not _same_code(
        previous_code,
        current_code,
    )
    actual_answer_changed = actual_option_changed or actual_code_changed
    return {
        "change_type": "stepwise",
        "question_key": question_key,
        "question_id": str(payload.get("question_id", "")),
        "task_type": str(payload.get("task_type", "unknown")),
        "target_agent_id": target_agent_id,
        "agent_id": target_agent_id,
        "from_round": previous_round_index,
        "to_round": current_round_index,
        "from_option_ids": _as_option_ids(previous_answer),
        "to_option_ids": _as_option_ids(current_answer),
        "actual_option_changed": actual_option_changed,
        "actual_code_changed": actual_code_changed,
        "actual_answer_changed": actual_answer_changed,
        "previous_code_hash": _code_hash(previous_code) if _is_code_generation(payload) else "",
        "current_code_hash": _code_hash(current_code) if _is_code_generation(payload) else "",
        "code_diff": _code_diff(previous_code, current_code) if actual_code_changed else "",
        "previous_answer_reasoning": str(previous_answer.get("reasoning", "")),
        "new_answer_reasoning": str(current_answer.get("reasoning", "")),
        "self_reported_changed_answer": bool(current_answer.get("changed_answer", False)),
        "self_reported_drivers": _clean_agent_ids(current_answer.get("change_drivers", [])),
        "change_summary": str(current_answer.get("change_summary", "")),
        "previous_round_reviews": _previous_round_reviews(
            question_key=question_key,
            round_index=previous_round_index,
            target_agent_id=target_agent_id,
            agent_result=previous_result,
        ),
        "third_party_inferred_drivers": [],
        "self_decision_change": False,
        "attribution_result": NOT_EVALUATED,
        "driver_attribution_uncertain": False,
        "driver_attribution_warnings": [],
    }


def _build_driver_prompt(change: dict[str, Any]) -> str:
    payload = {
        "stage": "driver_attribution",
        "task": (
            "Judge which previous-round reviewers, if any, plausibly drove the target "
            "agent's answer change. Use only previous_round_reviews as evidence."
        ),
        "question_context": change["question_context"],
        "change": {
            "target_agent_id": change["target_agent_id"],
            "from_round": change["from_round"],
            "to_round": change["to_round"],
            "from_option_ids": change["from_option_ids"],
            "to_option_ids": change["to_option_ids"],
            "actual_answer_changed": change.get("actual_answer_changed", True),
            "actual_option_changed": change.get("actual_option_changed", False),
            "actual_code_changed": change.get("actual_code_changed", False),
            "code_diff": change.get("code_diff", ""),
            "previous_answer_reasoning": change["previous_answer_reasoning"],
            "new_answer_reasoning": change["new_answer_reasoning"],
            "self_reported_drivers": change["self_reported_drivers"],
        },
        "previous_round_reviews": change["previous_round_reviews"],
        "response_schema": {
            "driver_scores": [
                {
                    "agent_id": "reviewer agent id from previous_round_reviews",
                    "score": "float from 0.0 to 1.0",
                    "label": "no_driver | weak_driver | plausible_driver | strong_driver",
                    "evidence_review_ids": ["review_id values from previous_round_reviews"],
                    "evidence_type": [
                        "opposed_previous_answer",
                        "recommended_new_answer",
                        "matched_new_reasoning",
                    ],
                }
            ],
            "insufficient_evidence": "boolean",
            "reason": "brief explanation without quoting or inventing evidence text",
        },
        "constraints": [
            "Return valid JSON only.",
            "Do not generate evidence text.",
            "Use evidence_review_ids only; each id must come from previous_round_reviews.",
            "Only score reviewer agent ids found in previous_round_reviews.",
            "The self_reported_drivers field is not evidence and must not be treated as fact.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _valid_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= score <= 1:
        return None
    return score


def _attribute_stepwise_change(
    *,
    change: dict[str, Any],
    config: DriverAttributionConfig,
    llm_settings: DashScopeSettings,
    client: StructuredLLMClient,
) -> None:
    candidate_reviews = change["previous_round_reviews"]
    review_by_id = {review["review_id"]: review for review in candidate_reviews}
    candidate_agent_ids = {review["reviewer_agent_id"] for review in candidate_reviews}
    if not candidate_agent_ids:
        change["self_decision_change"] = True
        change["attribution_result"] = SELF_DECISION_CHANGE
        change["driver_attribution_uncertain"] = True
        return

    warnings: list[str] = []
    judge_responses: list[dict[str, Any]] = []

    def judge_task(judge_config) -> dict[str, Any]:
        try:
            return client.generate_json(
                system_prompt=config.prompt,
                user_prompt=_build_driver_prompt(change),
                model=judge_config.model or llm_settings.model,
                temperature=judge_config.temperature,
            )
        except Exception as error:  # LLM failures should not destroy the whole offline report.
            return {
                "driver_scores": [],
                "insufficient_evidence": True,
                "_error": f"{type(error).__name__}: {error}",
            }

    with ThreadPoolExecutor(max_workers=min(config.max_workers, len(config.judges))) as executor:
        judge_responses = list(executor.map(judge_task, config.judges))

    scores_by_agent = {agent_id: [0.0 for _ in config.judges] for agent_id in candidate_agent_ids}
    evidence_ids_by_agent: dict[str, set[str]] = {agent_id: set() for agent_id in candidate_agent_ids}
    insufficient_count = 0

    for judge_index, response in enumerate(judge_responses):
        if "_error" in response:
            warnings.append(f"judge_{judge_index + 1}: {response['_error']}")
        if bool(response.get("insufficient_evidence", False)):
            insufficient_count += 1
        raw_scores = response.get("driver_scores", [])
        if not isinstance(raw_scores, list):
            warnings.append(f"judge_{judge_index + 1}: driver_scores was not a list")
            continue
        for item in raw_scores:
            if not isinstance(item, dict):
                warnings.append(f"judge_{judge_index + 1}: skipped non-object driver score")
                continue
            if "evidence" in item:
                warnings.append(
                    f"judge_{judge_index + 1}: ignored generated evidence text for "
                    f"{item.get('agent_id', 'unknown')}"
                )
            agent_id = str(item.get("agent_id", "")).strip()
            if agent_id not in candidate_agent_ids:
                warnings.append(f"judge_{judge_index + 1}: rejected unknown agent_id {agent_id!r}")
                continue
            score = _valid_score(item.get("score"))
            if score is None:
                warnings.append(f"judge_{judge_index + 1}: rejected invalid score for {agent_id}")
                continue
            raw_evidence_ids = item.get("evidence_review_ids", [])
            if not isinstance(raw_evidence_ids, list):
                warnings.append(f"judge_{judge_index + 1}: evidence_review_ids was not a list")
                continue
            valid_evidence_ids = [
                str(review_id)
                for review_id in raw_evidence_ids
                if str(review_id) in review_by_id
                and review_by_id[str(review_id)]["reviewer_agent_id"] == agent_id
            ]
            invalid_evidence_ids = set(str(review_id) for review_id in raw_evidence_ids) - set(
                valid_evidence_ids
            )
            if invalid_evidence_ids:
                warnings.append(
                    f"judge_{judge_index + 1}: rejected invalid evidence ids for "
                    f"{agent_id}: {sorted(invalid_evidence_ids)}"
                )
            if not valid_evidence_ids:
                continue
            scores_by_agent[agent_id][judge_index] = score
            evidence_ids_by_agent[agent_id].update(valid_evidence_ids)

    inferred_drivers: list[dict[str, Any]] = []
    for agent_id in sorted(candidate_agent_ids):
        evidence_ids = sorted(evidence_ids_by_agent[agent_id])
        mean_score = sum(scores_by_agent[agent_id]) / len(config.judges)
        label = _label_for_score(mean_score, config)
        if mean_score < config.plausible_driver_threshold or not evidence_ids:
            continue
        inferred_drivers.append(
            {
                "agent_id": agent_id,
                "mean_score": mean_score,
                "label": label,
                "judge_scores": scores_by_agent[agent_id],
                "evidence_review_ids": evidence_ids,
                "evidence_quotes": [
                    review_by_id[review_id]["main_reason_quote"] for review_id in evidence_ids
                ],
            }
        )

    change["third_party_inferred_drivers"] = sorted(
        inferred_drivers,
        key=lambda item: (-float(item["mean_score"]), str(item["agent_id"])),
    )
    change["driver_attribution_warnings"] = warnings
    change["driver_attribution_uncertain"] = insufficient_count > len(config.judges) / 2
    if change["third_party_inferred_drivers"]:
        change["self_decision_change"] = False
        change["attribution_result"] = EXTERNAL_DRIVER_FOUND
    elif insufficient_count == len(config.judges) and warnings:
        change["self_decision_change"] = False
        change["attribution_result"] = NOT_EVALUATED
    else:
        change["self_decision_change"] = True
        change["attribution_result"] = SELF_DECISION_CHANGE


def _aggregate_drivers(
    *,
    steps: list[dict[str, Any]],
    config: DriverAttributionConfig | None,
) -> list[dict[str, Any]]:
    scores: dict[str, list[float]] = {}
    refs: dict[str, list[str]] = {}
    evidence_ids: dict[str, set[str]] = {}
    evidence_quotes: dict[str, set[str]] = {}

    for step in steps:
        step_ref = _step_ref(step)
        for driver in step.get("third_party_inferred_drivers", []):
            agent_id = str(driver["agent_id"])
            scores.setdefault(agent_id, []).append(float(driver["mean_score"]))
            refs.setdefault(agent_id, []).append(step_ref)
            evidence_ids.setdefault(agent_id, set()).update(driver.get("evidence_review_ids", []))
            evidence_quotes.setdefault(agent_id, set()).update(driver.get("evidence_quotes", []))

    result: list[dict[str, Any]] = []
    for agent_id, agent_scores in scores.items():
        aggregate_score = sum(agent_scores) / len(agent_scores)
        if config is None:
            label = "driver"
        else:
            label = _label_for_score(aggregate_score, config)
        result.append(
            {
                "agent_id": agent_id,
                "aggregate_score": aggregate_score,
                "label": label,
                "supporting_stepwise_changes": refs[agent_id],
                "evidence_review_ids": sorted(evidence_ids.get(agent_id, set())),
                "evidence_quotes": sorted(evidence_quotes.get(agent_id, set())),
            }
        )
    return sorted(result, key=lambda item: (-float(item["aggregate_score"]), str(item["agent_id"])))


def _build_initial_to_final_change(
    *,
    payload: dict[str, Any],
    target_agent_id: str,
    first_round_index: int,
    final_round_index: int,
    first_result: dict[str, Any],
    final_result: dict[str, Any],
    stepwise_changes: list[dict[str, Any]],
    driver_attribution_config: DriverAttributionConfig | None,
) -> dict[str, Any]:
    first_answer = dict(first_result.get("answer") or {})
    final_answer = dict(final_result.get("answer") or {})
    initial_option_ids = _as_option_ids(first_answer)
    final_option_ids = _as_option_ids(final_answer)
    first_code = _as_code(first_answer)
    final_code = _as_code(final_answer)
    is_code_generation = _is_code_generation(payload)
    actual_option_changed = not _same_option_ids(initial_option_ids, final_option_ids)
    actual_code_changed = is_code_generation and not _same_code(first_code, final_code)
    changed = actual_option_changed or actual_code_changed
    final_code_hash = _code_hash(final_code) if is_code_generation else ""
    if is_code_generation:
        steps_to_final = [
            step for step in stepwise_changes if step.get("current_code_hash") == final_code_hash
        ]
    else:
        steps_to_final = [
            step
            for step in stepwise_changes
            if _same_option_ids(step["to_option_ids"], final_option_ids)
        ]
    direct_final_steps = steps_to_final[-1:] if changed and steps_to_final else []
    self_decision_steps = [
        _step_ref(step)
        for step in steps_to_final
        if step.get("self_decision_change") and step.get("attribution_result") == SELF_DECISION_CHANGE
    ]
    return {
        "change_type": "initial_to_final",
        "question_key": str(payload.get("question_key") or payload.get("question_id") or "unknown"),
        "question_id": str(payload.get("question_id", "")),
        "task_type": str(payload.get("task_type", "unknown")),
        "target_agent_id": target_agent_id,
        "agent_id": target_agent_id,
        "from_round": first_round_index,
        "to_round": final_round_index,
        "from_option_ids": initial_option_ids,
        "to_option_ids": final_option_ids,
        "actual_option_changed": actual_option_changed,
        "actual_code_changed": actual_code_changed,
        "actual_answer_changed": changed,
        "previous_code_hash": _code_hash(first_code) if is_code_generation else "",
        "current_code_hash": final_code_hash,
        "code_diff": _code_diff(first_code, final_code) if actual_code_changed else "",
        "changed": changed,
        "driver_attribution_method": "aggregated_from_stepwise_changes",
        "cumulative_path_drivers": _aggregate_drivers(
            steps=stepwise_changes,
            config=driver_attribution_config,
        ),
        "direct_final_state_drivers": (
            _aggregate_drivers(steps=direct_final_steps, config=driver_attribution_config)
            if changed
            else []
        ),
        "net_change_drivers": (
            _aggregate_drivers(steps=steps_to_final, config=driver_attribution_config)
            if changed
            else []
        ),
        "self_decision_change_steps": self_decision_steps if changed else [],
        "supporting_stepwise_changes": [_step_ref(step) for step in stepwise_changes],
    }


def _update_driver_counts(summary: dict[str, Any]) -> None:
    counter: dict[str, dict[str, int]] = {}
    for change in summary["stepwise_changes"]:
        for driver in change.get("third_party_inferred_drivers", []):
            agent_id = str(driver["agent_id"])
            bucket = counter.setdefault(agent_id, {"strong_count": 0, "plausible_count": 0})
            if driver.get("label") == "strong_driver":
                bucket["strong_count"] += 1
            elif driver.get("label") == "plausible_driver":
                bucket["plausible_count"] += 1
    summary["third_party_driver_counts"] = [
        {"agent_id": agent_id, **counts}
        for agent_id, counts in sorted(
            counter.items(),
            key=lambda item: (-(item[1]["strong_count"] + item[1]["plausible_count"]), item[0]),
        )
    ]


def _compact_evidence(driver: dict[str, Any]) -> list[dict[str, str]]:
    evidence_ids = list(driver.get("evidence_review_ids", []))
    quotes = list(driver.get("evidence_quotes", []))
    return [
        {"review_id": str(review_id), "quote": str(quotes[index]) if index < len(quotes) else ""}
        for index, review_id in enumerate(evidence_ids)
    ]


def _compact_driver(driver: dict[str, Any], *, source_step: str | None = None) -> dict[str, Any]:
    result = {
        "agent_id": str(driver["agent_id"]),
        "confidence": float(driver.get("mean_score", driver.get("aggregate_score", 0.0))),
        "evidence": _compact_evidence(driver),
    }
    if source_step is not None:
        result["source_step"] = source_step
    return result


def _step_cause(step: dict[str, Any], *, attribution_enabled: bool) -> str:
    if step.get("third_party_inferred_drivers"):
        return "external_agent"
    if not attribution_enabled:
        return UNKNOWN_CAUSE
    if step.get("attribution_result") == SELF_DECISION_CHANGE:
        return "self_decision"
    return UNKNOWN_CAUSE


def _compact_step_change(step: dict[str, Any], *, attribution_enabled: bool) -> dict[str, Any]:
    result = {
        "from_round": int(step["from_round"]),
        "to_round": int(step["to_round"]),
        "from_answer": list(step["from_option_ids"]),
        "to_answer": list(step["to_option_ids"]),
        "actual_option_changed": bool(step.get("actual_option_changed", False)),
        "actual_code_changed": bool(step.get("actual_code_changed", False)),
        "actual_answer_changed": bool(step.get("actual_answer_changed", True)),
        "cause": _step_cause(step, attribution_enabled=attribution_enabled),
        "drivers": [
            _compact_driver(driver)
            for driver in step.get("third_party_inferred_drivers", [])
        ],
    }
    if step.get("previous_code_hash") or step.get("current_code_hash"):
        result["previous_code_hash"] = str(step.get("previous_code_hash", ""))
        result["current_code_hash"] = str(step.get("current_code_hash", ""))
    if step.get("code_diff"):
        result["code_diff"] = str(step["code_diff"])
    return result


def build_change_chain_report(full_summary: dict[str, Any]) -> dict[str, Any]:
    attribution_enabled = bool(full_summary.get("driver_attribution_enabled", False))
    stepwise_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for step in full_summary.get("stepwise_changes", []):
        stepwise_by_pair.setdefault(
            (str(step["question_key"]), str(step["agent_id"])),
            [],
        ).append(step)

    changes: list[dict[str, Any]] = []
    for pair, stepwise_changes in sorted(stepwise_by_pair.items()):
        question_key, agent_id = pair
        stepwise_changes = sorted(
            stepwise_changes,
            key=lambda step: (int(step["from_round"]), int(step["to_round"])),
        )
        first_step = stepwise_changes[0]
        matching_initial = next(
            (
                item
                for item in full_summary.get("initial_to_final_changes", [])
                if item["question_key"] == question_key and item["agent_id"] == agent_id
            ),
            None,
        )
        if matching_initial is None:
            continue

        answer_by_round: dict[int, list[str]] = {
            int(matching_initial["from_round"]): list(matching_initial["from_option_ids"]),
            int(matching_initial["to_round"]): list(matching_initial["to_option_ids"]),
        }
        code_hash_by_round: dict[int, str] = {}
        if matching_initial.get("previous_code_hash") or matching_initial.get("current_code_hash"):
            code_hash_by_round[int(matching_initial["from_round"])] = str(
                matching_initial.get("previous_code_hash", "")
            )
            code_hash_by_round[int(matching_initial["to_round"])] = str(
                matching_initial.get("current_code_hash", "")
            )
        for step in stepwise_changes:
            answer_by_round[int(step["from_round"])] = list(step["from_option_ids"])
            answer_by_round[int(step["to_round"])] = list(step["to_option_ids"])
            if step.get("previous_code_hash") or step.get("current_code_hash"):
                code_hash_by_round[int(step["from_round"])] = str(
                    step.get("previous_code_hash", "")
                )
                code_hash_by_round[int(step["to_round"])] = str(step.get("current_code_hash", ""))

        answer_chain = [
            {
                "round": round_index,
                "answer": answer,
                **(
                    {"code_hash": code_hash_by_round[round_index]}
                    if round_index in code_hash_by_round
                    else {}
                ),
            }
            for round_index, answer in sorted(answer_by_round.items())
        ]
        final_answer = list(matching_initial["to_option_ids"])
        first_answer = list(matching_initial["from_option_ids"])
        final_answer_changed = bool(matching_initial["changed"])

        step_changes = [
            _compact_step_change(step, attribution_enabled=attribution_enabled)
            for step in stepwise_changes
        ]

        final_answer_drivers: list[dict[str, Any]] = []
        final_driver_method = "no_net_final_change"
        if final_answer_changed:
            if not attribution_enabled:
                final_driver_method = "driver_attribution_not_run"
            else:
                final_steps = [
                    step
                    for step in stepwise_changes
                    if _same_option_ids(list(step["to_option_ids"]), final_answer)
                ]
                final_step = final_steps[-1] if final_steps else None
                if final_step is None:
                    final_driver_method = "no_step_into_final_answer"
                elif final_step.get("third_party_inferred_drivers"):
                    source_step = _step_ref(final_step)
                    final_answer_drivers = [
                        _compact_driver(driver, source_step=source_step)
                        for driver in final_step.get("third_party_inferred_drivers", [])
                    ]
                    final_driver_method = "last_step_into_final_answer"
                elif final_step.get("attribution_result") == SELF_DECISION_CHANGE:
                    final_driver_method = "last_step_into_final_answer_self_decision"
                else:
                    final_driver_method = "driver_attribution_not_run"

        changes.append(
            {
                "question_key": question_key,
                "agent_id": agent_id,
                "answer_chain": answer_chain,
                "first_answer": first_answer,
                "final_answer": final_answer,
                "first_code_hash": str(matching_initial.get("previous_code_hash", "")),
                "final_code_hash": str(matching_initial.get("current_code_hash", "")),
                "final_answer_changed": final_answer_changed,
                "actual_option_changed": bool(matching_initial.get("actual_option_changed", False)),
                "actual_code_changed": bool(matching_initial.get("actual_code_changed", False)),
                "actual_answer_changed": bool(matching_initial.get("actual_answer_changed", False)),
                "code_diff": str(matching_initial.get("code_diff", "")),
                "step_changes": step_changes,
                "final_answer_drivers": final_answer_drivers,
                "final_driver_method": final_driver_method,
            }
        )

    external_agent_step_count = sum(
        1 for change in changes for step in change["step_changes"] if step["cause"] == "external_agent"
    )
    self_decision_step_count = sum(
        1 for change in changes for step in change["step_changes"] if step["cause"] == "self_decision"
    )
    unknown_step_count = sum(
        1 for change in changes for step in change["step_changes"] if step["cause"] == UNKNOWN_CAUSE
    )
    return {
        "run_path": full_summary["run_path"],
        "summary": {
            "total_questions": full_summary["total_questions"],
            "total_stepwise_comparisons": full_summary["change_opportunities"],
            "changed_chain_count": len(changes),
            "stepwise_change_count": full_summary["stepwise_change_count"],
            "final_answer_changed_count": sum(
                1 for change in changes if change["final_answer_changed"]
            ),
            "actual_option_changed_count": full_summary["actual_option_changed_count"],
            "actual_code_changed_count": full_summary["actual_code_changed_count"],
            "actual_answer_changed_count": full_summary["actual_answer_changed_count"],
            "questions_with_actual_option_changes": full_summary[
                "questions_with_actual_option_changes"
            ],
            "questions_with_actual_code_changes": full_summary["questions_with_actual_code_changes"],
            "questions_with_actual_answer_changes": full_summary[
                "questions_with_actual_answer_changes"
            ],
            "external_agent_step_count": external_agent_step_count,
            "self_decision_step_count": self_decision_step_count,
            "unknown_step_count": unknown_step_count,
            "driver_attribution_enabled": attribution_enabled,
        },
        "changes": changes,
    }


def evaluate_changed_answers(
    run_path: str | Path,
    *,
    driver_attribution_config: DriverAttributionConfig | None = None,
    llm_settings: DashScopeSettings | None = None,
    client: StructuredLLMClient | None = None,
    include_unchanged_initial_to_final: bool = False,
) -> dict[str, Any]:
    path = Path(run_path)
    payloads = _load_question_payloads(path)
    attribution_enabled = bool(driver_attribution_config and driver_attribution_config.enabled)
    if attribution_enabled:
        assert driver_attribution_config is not None
        driver_attribution_config.validate()
        if client is None:
            if llm_settings is None:
                raise ValueError("llm_settings is required when driver attribution is enabled.")
            client = build_structured_client(llm_settings)
        if llm_settings is None:
            llm_settings = DashScopeSettings(api_key="", model="")

    summary = {
        "run_path": str(path),
        "total_questions": len(payloads),
        "questions_with_changes": 0,
        "questions_with_self_reported_changes": 0,
        "questions_with_actual_option_changes": 0,
        "questions_with_actual_code_changes": 0,
        "questions_with_actual_answer_changes": 0,
        "question_change_rate": 0.0,
        "actual_question_change_rate": 0.0,
        "answer_submissions": 0,
        "change_opportunities": 0,
        "self_reported_changed_answer_count": 0,
        "changed_answer_count": 0,
        "actual_option_changed_count": 0,
        "actual_code_changed_count": 0,
        "actual_answer_changed_count": 0,
        "both_self_reported_and_actual_count": 0,
        "self_reported_only_count": 0,
        "actual_only_count": 0,
        "self_report_actual_agreement_count": 0,
        "self_report_actual_disagreement_count": 0,
        "changed_answer_rate_all_submissions": 0.0,
        "changed_answer_rate_after_prior_answer": 0.0,
        "actual_option_changed_rate_after_prior_answer": 0.0,
        "actual_code_changed_rate_after_prior_answer": 0.0,
        "actual_answer_changed_rate_after_prior_answer": 0.0,
        "self_report_actual_agreement_rate": 0.0,
        "driver_attribution_note": (
            "change_drivers are self-reported by the answer agent; third-party inferred "
            "drivers are evidence-constrained judgments over previous-round reviews."
        ),
        "driver_attribution_enabled": attribution_enabled,
        "by_task_type": {},
        "by_round": {},
        "by_agent": {},
        "change_drivers": [],
        "changed_answers": [],
        "actual_option_changes": [],
        "actual_code_changes": [],
        "actual_answer_changes": [],
        "stepwise_changes": [],
        "initial_to_final_changes": [],
        "initial_to_final_changed_details": [],
        "initial_to_final_unchanged_count": 0,
        "initial_to_final_total_comparisons": 0,
        "change_consistency_mismatches": [],
        "stepwise_change_count": 0,
        "initial_to_final_change_count": 0,
        "driver_attribution_evaluated_count": 0,
        "driver_attribution_uncertain_count": 0,
        "third_party_driver_counts": [],
    }

    questions_with_self_reported_changes: set[str] = set()
    questions_with_actual_option_changes: set[str] = set()
    questions_with_actual_code_changes: set[str] = set()
    questions_with_actual_answer_changes: set[str] = set()
    driver_counts: Counter[str] = Counter()

    for payload in payloads:
        question_key = str(payload.get("question_key") or payload.get("question_id") or "unknown")
        task_type = str(payload.get("task_type") or "unknown")
        prior_results_by_agent: dict[str, tuple[int, dict[str, Any]]] = {}
        question_stepwise_changes_by_agent: dict[str, list[dict[str, Any]]] = {}
        sorted_rounds = _sorted_rounds(payload.get("rounds", []))
        round_maps = [
            (int(round_payload.get("round_index", 0)), _round_agent_map(round_payload))
            for round_payload in sorted_rounds
        ]

        task_bucket = summary["by_task_type"].setdefault(task_type, _new_bucket())
        task_bucket["total_questions"] = int(task_bucket.get("total_questions", 0)) + 1

        for round_index, agent_results in round_maps:
            round_key = str(round_index)
            round_bucket = summary["by_round"].setdefault(round_key, _new_bucket())

            for agent_id, agent_result in agent_results.items():
                answer = dict(agent_result.get("answer") or {})
                changed_answer = bool(answer.get("changed_answer", False))
                previous_pair = prior_results_by_agent.get(agent_id)
                has_prior_answer = previous_pair is not None
                previous_answer = (
                    dict(previous_pair[1].get("answer") or {}) if previous_pair is not None else {}
                )
                previous_option_ids = _as_option_ids(previous_answer)
                selected_option_ids = _as_option_ids(answer)
                actual_option_changed = (
                    has_prior_answer
                    and not _same_option_ids(previous_option_ids, selected_option_ids)
                )
                previous_code = _as_code(previous_answer)
                current_code = _as_code(answer)
                actual_code_changed = (
                    has_prior_answer
                    and _is_code_generation(payload)
                    and not _same_code(previous_code, current_code)
                )
                actual_answer_changed = actual_option_changed or actual_code_changed

                agent_bucket = summary["by_agent"].setdefault(agent_id, _new_bucket())
                for bucket in (summary, task_bucket, round_bucket, agent_bucket):
                    _record_submission(
                        bucket,
                        has_prior_answer=has_prior_answer,
                        self_reported_changed=changed_answer,
                        actual_option_changed=actual_option_changed,
                        actual_code_changed=actual_code_changed,
                        actual_answer_changed=actual_answer_changed,
                    )

                if changed_answer:
                    questions_with_self_reported_changes.add(question_key)
                    change_drivers = _clean_agent_ids(answer.get("change_drivers", []))
                    driver_counts.update(change_drivers)
                    summary["changed_answers"].append(
                        {
                            "question_key": question_key,
                            "question_id": str(payload.get("question_id", "")),
                            "task_type": task_type,
                            "round_index": round_index,
                            "agent_id": agent_id,
                            "previous_selected_option_ids": previous_option_ids,
                            "selected_option_ids": selected_option_ids,
                            "actual_option_changed": actual_option_changed,
                            "actual_code_changed": actual_code_changed,
                            "actual_answer_changed": actual_answer_changed,
                            "previous_code_hash": _code_hash(previous_code)
                            if _is_code_generation(payload) and has_prior_answer
                            else "",
                            "current_code_hash": _code_hash(current_code)
                            if _is_code_generation(payload)
                            else "",
                            "code_diff": _code_diff(previous_code, current_code)
                            if actual_code_changed
                            else "",
                            "change_drivers": change_drivers,
                            "change_summary": str(answer.get("change_summary", "")),
                        }
                    )

                if actual_answer_changed and previous_pair is not None:
                    if actual_option_changed:
                        questions_with_actual_option_changes.add(question_key)
                    if actual_code_changed:
                        questions_with_actual_code_changes.add(question_key)
                    questions_with_actual_answer_changes.add(question_key)
                    previous_round_index, previous_result = previous_pair
                    stepwise_change = _build_stepwise_change(
                        payload=payload,
                        target_agent_id=agent_id,
                        previous_round_index=previous_round_index,
                        current_round_index=round_index,
                        previous_result=previous_result,
                        current_result=agent_result,
                    )
                    stepwise_change["question_context"] = _question_context(payload)
                    if attribution_enabled:
                        assert driver_attribution_config is not None
                        assert llm_settings is not None
                        assert client is not None
                        _attribute_stepwise_change(
                            change=stepwise_change,
                            config=driver_attribution_config,
                            llm_settings=llm_settings,
                            client=client,
                        )
                    stepwise_change.pop("question_context", None)
                    summary["stepwise_changes"].append(stepwise_change)
                    question_stepwise_changes_by_agent.setdefault(agent_id, []).append(stepwise_change)
                    change_record = {
                        "question_key": question_key,
                        "question_id": str(payload.get("question_id", "")),
                        "task_type": task_type,
                        "round_index": round_index,
                        "agent_id": agent_id,
                        "previous_selected_option_ids": previous_option_ids,
                        "selected_option_ids": selected_option_ids,
                        "actual_option_changed": actual_option_changed,
                        "actual_code_changed": actual_code_changed,
                        "actual_answer_changed": actual_answer_changed,
                        "previous_code_hash": _code_hash(previous_code)
                        if _is_code_generation(payload)
                        else "",
                        "current_code_hash": _code_hash(current_code)
                        if _is_code_generation(payload)
                        else "",
                        "code_diff": _code_diff(previous_code, current_code)
                        if actual_code_changed
                        else "",
                        "self_reported_changed_answer": changed_answer,
                        "change_drivers": _clean_agent_ids(answer.get("change_drivers", [])),
                        "change_summary": str(answer.get("change_summary", "")),
                    }
                    summary["actual_answer_changes"].append(change_record)
                    if actual_option_changed:
                        summary["actual_option_changes"].append(change_record)
                    if actual_code_changed:
                        summary["actual_code_changes"].append(change_record)

                if has_prior_answer and changed_answer != actual_answer_changed:
                    summary["change_consistency_mismatches"].append(
                        {
                            "question_key": question_key,
                            "question_id": str(payload.get("question_id", "")),
                            "task_type": task_type,
                            "round_index": round_index,
                            "agent_id": agent_id,
                            "previous_selected_option_ids": previous_option_ids,
                            "selected_option_ids": selected_option_ids,
                            "self_reported_changed_answer": changed_answer,
                            "actual_option_changed": actual_option_changed,
                            "actual_code_changed": actual_code_changed,
                            "actual_answer_changed": actual_answer_changed,
                            "previous_code_hash": _code_hash(previous_code)
                            if _is_code_generation(payload)
                            else "",
                            "current_code_hash": _code_hash(current_code)
                            if _is_code_generation(payload)
                            else "",
                            "code_diff": _code_diff(previous_code, current_code)
                            if actual_code_changed
                            else "",
                            "change_summary": str(answer.get("change_summary", "")),
                        }
                    )

                prior_results_by_agent[agent_id] = (round_index, agent_result)

        if round_maps:
            first_round_index, first_agent_results = round_maps[0]
            final_round_index, final_agent_results = round_maps[-1]
            for agent_id in sorted(set(first_agent_results) & set(final_agent_results)):
                initial_to_final = _build_initial_to_final_change(
                    payload=payload,
                    target_agent_id=agent_id,
                    first_round_index=first_round_index,
                    final_round_index=final_round_index,
                    first_result=first_agent_results[agent_id],
                    final_result=final_agent_results[agent_id],
                    stepwise_changes=question_stepwise_changes_by_agent.get(agent_id, []),
                    driver_attribution_config=driver_attribution_config
                    if attribution_enabled
                    else None,
                )
                summary["initial_to_final_total_comparisons"] += 1
                has_stepwise_changes = bool(question_stepwise_changes_by_agent.get(agent_id))
                if initial_to_final["changed"]:
                    summary["initial_to_final_changes"].append(initial_to_final)
                    summary["initial_to_final_changed_details"].append(initial_to_final)
                else:
                    summary["initial_to_final_unchanged_count"] += 1
                    if has_stepwise_changes or include_unchanged_initial_to_final:
                        summary["initial_to_final_changes"].append(initial_to_final)

    for bucket in [
        summary,
        *summary["by_task_type"].values(),
        *summary["by_round"].values(),
        *summary["by_agent"].values(),
    ]:
        _update_rates(bucket)

    summary["questions_with_self_reported_changes"] = len(questions_with_self_reported_changes)
    summary["questions_with_actual_option_changes"] = len(questions_with_actual_option_changes)
    summary["questions_with_actual_code_changes"] = len(questions_with_actual_code_changes)
    summary["questions_with_actual_answer_changes"] = len(questions_with_actual_answer_changes)
    summary["questions_with_changes"] = len(questions_with_self_reported_changes)
    summary["question_change_rate"] = _rate(len(questions_with_self_reported_changes), len(payloads))
    summary["actual_question_change_rate"] = _rate(
        len(questions_with_actual_answer_changes),
        len(payloads),
    )
    summary["change_drivers"] = [
        {"agent_id": agent_id, "count": count}
        for agent_id, count in sorted(driver_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    summary["by_round"] = {
        key: summary["by_round"][key]
        for key in sorted(summary["by_round"], key=lambda value: int(value))
    }
    summary["by_agent"] = {key: summary["by_agent"][key] for key in sorted(summary["by_agent"])}
    summary["stepwise_change_count"] = len(summary["stepwise_changes"])
    summary["initial_to_final_change_count"] = len(summary["initial_to_final_changed_details"])
    summary["driver_attribution_evaluated_count"] = sum(
        1 for item in summary["stepwise_changes"] if item["attribution_result"] != NOT_EVALUATED
    )
    summary["driver_attribution_uncertain_count"] = sum(
        1 for item in summary["stepwise_changes"] if item["driver_attribution_uncertain"]
    )
    _update_driver_counts(summary)
    return summary


def write_summary(run_path: Path, summary: dict[str, Any], output: str | Path | None) -> Path:
    output_path = Path(output) if output else run_path / "evaluation" / DEFAULT_OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate answer decision changes from saved run results."
    )
    parser.add_argument(
        "run_path",
        nargs="?",
        help="Path to a run directory. Defaults to the latest run under --runs-dir.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory used to find the latest run when run_path is omitted.",
    )
    parser.add_argument(
        "--output",
        help=f"Output JSON path. Defaults to <run_path>/evaluation/{DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the report without writing it to disk.",
    )
    parser.add_argument(
        "--with-driver-attribution",
        action="store_true",
        help="Call configured third-party judge LLMs to infer drivers for stepwise changes.",
    )
    parser.add_argument(
        "--include-unchanged-initial-to-final",
        action="store_true",
        help=(
            "Include every unchanged round-1-to-final comparison in initial_to_final_changes. "
            "By default only changed initial-to-final details are emitted."
        ),
    )
    parser.add_argument(
        "--debug-full",
        action="store_true",
        help="Write/print the full internal diagnostic structure instead of the simple chain report.",
    )
    parser.add_argument(
        "--config",
        default="config/agents.yaml",
        help="Experiment config path used for driver_attribution settings.",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Environment file used for LLM settings when driver attribution is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_path = Path(args.run_path) if args.run_path else _latest_run_path(Path(args.runs_dir))

    driver_attribution_config = None
    llm_settings = None
    if args.with_driver_attribution:
        experiment_config = load_experiment_config(args.config)
        driver_attribution_config = experiment_config.driver_attribution
        if not driver_attribution_config.enabled:
            raise ValueError("driver_attribution.enabled must be true when using --with-driver-attribution.")
        llm_settings = load_dashscope_settings(args.env)

    full_summary = evaluate_changed_answers(
        run_path,
        driver_attribution_config=driver_attribution_config,
        llm_settings=llm_settings,
        include_unchanged_initial_to_final=args.include_unchanged_initial_to_final,
    )
    report = full_summary if args.debug_full else build_change_chain_report(full_summary)

    if args.no_write:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        output_path = write_summary(run_path, report, args.output)
        print(f"run_path={run_path}")
        print(f"changed_answer_summary_path={output_path}")
        print(
            json.dumps(
                {
                    "total_questions": report["summary"]["total_questions"]
                    if not args.debug_full
                    else full_summary["total_questions"],
                    "changed_chain_count": report["summary"]["changed_chain_count"]
                    if not args.debug_full
                    else None,
                    "stepwise_change_count": report["summary"]["stepwise_change_count"]
                    if not args.debug_full
                    else full_summary["stepwise_change_count"],
                    "final_answer_changed_count": report["summary"]["final_answer_changed_count"]
                    if not args.debug_full
                    else full_summary["initial_to_final_change_count"],
                    "driver_attribution_enabled": report["summary"]["driver_attribution_enabled"]
                    if not args.debug_full
                    else full_summary["driver_attribution_enabled"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
