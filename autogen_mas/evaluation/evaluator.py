from __future__ import annotations

from copy import deepcopy
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from autogen_mas.answer_matching import (
    canonicalize_short_answer,
    is_short_answer_correct,
    short_answer_tie_key,
)
from autogen_mas.evaluation.chess_evaluator import ChessEvaluator
from autogen_mas.models import EvaluationQuestionResult, EvaluationReport
from autogen_mas.persistence import JsonRunStore


TOP_AGENT_SELECTION_RULE = (
    "last_round.total_score desc -> all_rounds.average_score mean desc -> "
    "agent_id asc; unresolved differing-answer ties excluded"
)
SINGLE_AGENT_SELECTION_RULE = "single_agent_direct_answer"
SELECTION_RULE_ALIASES = {
    "top_agent": "top_agent",
    "top-agent": "top_agent",
}


class Evaluator:
    def __init__(self) -> None:
        self.store = JsonRunStore("runs")

    def evaluate(
        self,
        run_path: str | Path,
        *,
        selection_rule: str = "top_agent",
    ) -> EvaluationReport:
        path = Path(run_path)
        normalized_selection_rule = self._normalize_selection_rule(selection_rule)
        question_files = sorted((path / "questions").glob("*.json"))
        question_payloads = [
            json.loads(question_file.read_text(encoding="utf-8"))
            for question_file in question_files
        ]
        question_results = [
            self._evaluate_question_payload(
                payload,
                selection_rule=normalized_selection_rule,
            )
            for payload in question_payloads
        ]
        summary = self._build_summary(question_results, question_payloads)
        summary["selection_rule_id"] = normalized_selection_rule
        summary["selection_rule"] = self._selection_rule_label(normalized_selection_rule)
        JsonRunStore(path.parent).persist_evaluation(
            path,
            question_results=[item.to_dict() for item in question_results],
            summary=summary,
            evaluation_name=normalized_selection_rule,
            write_legacy=normalized_selection_rule == "top_agent",
        )
        return EvaluationReport(
            run_path=str(path),
            question_results=question_results,
            summary=summary,
        )

    def evaluate_single_agent(
        self,
        run_path: str | Path,
        *,
        agent_id: str,
        model: str,
        temperature: float,
    ) -> dict:
        path = Path(run_path)
        report = self.evaluate(path)
        summary = {
            "run_path": str(path),
            "agent_id": agent_id,
            "model": model,
            "temperature": temperature,
            "selection_rule": SINGLE_AGENT_SELECTION_RULE,
            "total_questions": report.summary["total_questions"],
            "correct_questions": report.summary["correct_questions"],
            "accuracy": report.summary["accuracy"],
            "by_task_type": report.summary["by_task_type"],
        }
        JsonRunStore(path.parent).persist_single_agent_summary(path, summary)
        return summary

    def evaluate_round_sweep(
        self,
        run_path: str | Path,
        *,
        selection_rule: str = "top_agent",
        max_round: int | None = None,
    ) -> dict:
        path = Path(run_path)
        if max_round is not None and max_round < 1:
            raise ValueError("max_round must be at least 1.")
        normalized_selection_rule = self._normalize_selection_rule(selection_rule)
        question_files = sorted((path / "questions").glob("*.json"))
        question_payloads = [
            json.loads(question_file.read_text(encoding="utf-8"))
            for question_file in question_files
        ]
        available_rounds = [
            len(payload.get("rounds", []))
            for payload in question_payloads
            if isinstance(payload.get("rounds"), list)
        ]
        observed_max_round = max(available_rounds, default=0)
        sweep_max_round = (
            min(max_round, observed_max_round)
            if max_round is not None
            else observed_max_round
        )
        round_summaries: list[dict] = []
        for round_index in range(1, sweep_max_round + 1):
            sliced_payloads = [
                self._payload_with_round_limit(payload, round_index)
                for payload in question_payloads
            ]
            question_results = [
                self._evaluate_question_payload(
                    payload,
                    selection_rule=normalized_selection_rule,
                )
                for payload in sliced_payloads
                if payload.get("rounds")
            ]
            summary = self._build_summary(question_results, sliced_payloads)
            summary["round"] = round_index
            summary["selection_rule_id"] = normalized_selection_rule
            summary["selection_rule"] = self._selection_rule_label(
                normalized_selection_rule
            )
            summary["questions_with_observed_round"] = sum(
                1
                for payload in question_payloads
                if isinstance(payload.get("rounds"), list)
                and len(payload.get("rounds", [])) >= round_index
            )
            round_summaries.append(summary)
        sweep_summary = {
            "run_path": str(path),
            "experiment": "round_sweep",
            "selection_rule_id": normalized_selection_rule,
            "selection_rule": self._selection_rule_label(normalized_selection_rule),
            "max_round": sweep_max_round,
            "observed_max_round": observed_max_round,
            "requested_max_round": max_round,
            "total_questions": len(question_payloads),
            "rounds": round_summaries,
        }
        JsonRunStore(path.parent).persist_round_sweep_summary(path, sweep_summary)
        return sweep_summary

    def _normalize_selection_rule(self, selection_rule: str) -> str:
        try:
            return SELECTION_RULE_ALIASES[selection_rule]
        except KeyError as exc:
            supported = ", ".join(sorted(SELECTION_RULE_ALIASES))
            raise ValueError(
                f"Unsupported selection rule '{selection_rule}'. Supported: {supported}."
            ) from exc

    def _selection_rule_label(self, selection_rule: str) -> str:
        return TOP_AGENT_SELECTION_RULE

    def _payload_with_round_limit(self, payload: dict, round_limit: int) -> dict:
        limited_payload = deepcopy(payload)
        rounds = limited_payload.get("rounds", [])
        if isinstance(rounds, list):
            limited_payload["rounds"] = rounds[:round_limit]
        return limited_payload

    @staticmethod
    def _is_adversarial(agent_result: dict) -> bool:
        """Check whether an agent_result belongs to an adversarial agent."""
        return bool(
            agent_result.get("is_adversarial_agent")
            or agent_result.get("advers-agent")
        )

    def _evaluate_question_payload(
        self,
        payload: dict,
        *,
        selection_rule: str,
    ) -> EvaluationQuestionResult:
        return self._evaluate_question_payload_top_agent(payload)

    def _evaluate_question_payload_top_agent(self, payload: dict) -> EvaluationQuestionResult:
        selected, tie_candidate_ids = self._select_top_agent_with_tie(payload)
        selected_is_adversarial = self._is_adversarial(selected)
        is_tie = len(tie_candidate_ids) > 1
        if ChessEvaluator.supports_payload(payload):
            return ChessEvaluator.evaluate_question(
                payload=payload,
                selected=selected,
                tie_candidate_ids=tie_candidate_ids,
                selected_is_adversarial=selected_is_adversarial,
                is_tie=is_tie,
            )
        if payload.get("task_type") == "code_generation":
            is_correct = self._evaluate_code_generation(
                payload=payload,
                code=str(selected.get("answer", {}).get("code", "")),
            )
            return EvaluationQuestionResult(
                question_id=str(payload["question_id"]),
                question_key=str(payload["question_key"]),
                dataset_name=str(payload["dataset_name"]),
                task_type=str(payload["task_type"]),
                selected_agent_id=str(selected["agent_id"]),
                predicted_option_ids=[],
                correct_option_ids=[],
                is_correct=is_correct,
                selection_rule=TOP_AGENT_SELECTION_RULE,
                selected_is_adversarial=selected_is_adversarial,
                is_tie=is_tie,
                excluded_from_accuracy=is_tie,
                tie_candidate_ids=tie_candidate_ids,
            )
        if payload.get("task_type") == "math_short_answer":
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            predicted_final_answer = canonicalize_short_answer(
                str(selected["answer"].get("final_answer", "")),
                metadata=metadata,
            )
            acceptable_answers = [
                str(answer) for answer in payload.get("acceptable_answers", [])
            ]
            is_correct = is_short_answer_correct(
                predicted_final_answer,
                acceptable_answers,
                metadata=metadata,
            )
            return EvaluationQuestionResult(
                question_id=str(payload["question_id"]),
                question_key=str(payload["question_key"]),
                dataset_name=str(payload["dataset_name"]),
                task_type=str(payload["task_type"]),
                selected_agent_id=str(selected["agent_id"]),
                predicted_option_ids=[],
                correct_option_ids=[],
                is_correct=is_correct,
                selection_rule=TOP_AGENT_SELECTION_RULE,
                predicted_final_answer=predicted_final_answer,
                acceptable_answers=acceptable_answers,
                selected_is_adversarial=selected_is_adversarial,
                is_tie=is_tie,
                excluded_from_accuracy=is_tie,
                tie_candidate_ids=tie_candidate_ids,
            )
        predicted_option_ids = list(selected["answer"]["selected_option_ids"])
        correct_option_ids = list(payload["correct_option_ids"])
        is_correct = sorted(predicted_option_ids) == sorted(correct_option_ids)
        return EvaluationQuestionResult(
            question_id=str(payload["question_id"]),
            question_key=str(payload["question_key"]),
            dataset_name=str(payload["dataset_name"]),
            task_type=str(payload["task_type"]),
            selected_agent_id=str(selected["agent_id"]),
            predicted_option_ids=predicted_option_ids,
            correct_option_ids=correct_option_ids,
            is_correct=is_correct,
            selection_rule=TOP_AGENT_SELECTION_RULE,
            selected_is_adversarial=selected_is_adversarial,
            is_tie=is_tie,
            excluded_from_accuracy=is_tie,
            tie_candidate_ids=tie_candidate_ids,
        )

    def _select_top_agent(self, payload: dict) -> dict:
        selected, _tie_candidate_ids = self._select_top_agent_with_tie(payload)
        return selected

    def _select_top_agent_with_tie(self, payload: dict) -> tuple[dict, list[str]]:
        rounds = payload["rounds"]
        scoring_rounds, selected_round = self._top_agent_selection_rounds(payload, rounds)
        ranked_agent_results = sorted(
            selected_round["agent_results"],
            key=lambda item: (
                -float(item["total_score"]),
                -self._all_rounds_average_score_mean(
                    scoring_rounds,
                    str(item["agent_id"]),
                ),
                str(item["agent_id"]),
            ),
        )
        selected = ranked_agent_results[0]
        selected_score = float(selected["total_score"])
        selected_average = self._all_rounds_average_score_mean(
            scoring_rounds,
            str(selected["agent_id"]),
        )
        tie_candidate_ids = [
            str(item["agent_id"])
            for item in ranked_agent_results
            if float(item["total_score"]) == selected_score
            and self._all_rounds_average_score_mean(
                scoring_rounds,
                str(item["agent_id"]),
            )
            == selected_average
        ]
        if len(tie_candidate_ids) <= 1:
            return selected, []
        answer_keys = {
            self._answer_tie_key(item, payload=payload)
            for item in ranked_agent_results
            if str(item["agent_id"]) in tie_candidate_ids
        }
        if len(answer_keys) <= 1:
            return selected, []
        return selected, tie_candidate_ids

    def _top_agent_selection_rounds(
        self,
        payload: dict,
        rounds: list[dict],
    ) -> tuple[list[dict], dict]:
        if payload.get("task_type") in {
            "single_choice",
            "multiple_choice",
            "math_short_answer",
            "chess_move",
        }:
            for index, round_data in enumerate(rounds):
                if bool(round_data.get("reached_consensus", False)):
                    return rounds[: index + 1], round_data
        return rounds, rounds[-1]

    def _all_rounds_average_score_mean(self, rounds: list[dict], agent_id: str) -> float:
        values: list[float] = []
        for round_data in rounds:
            for agent_result in round_data.get("agent_results", []):
                if str(agent_result.get("agent_id", "")) == agent_id:
                    values.append(float(agent_result.get("average_score", 0.0)))
                    break
        return sum(values) / len(values) if values else 0.0

    def _answer_tie_key(self, agent_result: dict, *, payload: dict | None = None) -> tuple:
        answer = agent_result.get("answer", {})
        if not isinstance(answer, dict):
            return ("missing",)
        code = str(answer.get("code", "")).strip()
        if code:
            return ("code", code)
        final_answer = str(answer.get("final_answer", "")).strip()
        if final_answer:
            if isinstance(payload, dict) and ChessEvaluator.supports_payload(payload):
                return ChessEvaluator.tie_key(agent_result, payload=payload)
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            return (
                "final_answer",
                short_answer_tie_key(final_answer, metadata=metadata),
            )
        return (
            "options",
            tuple(sorted(str(item) for item in answer.get("selected_option_ids", []))),
        )

    def _evaluate_code_generation(self, *, payload: dict, code: str) -> bool:
        program = self._build_code_generation_program(
            prompt=str(payload["prompt"]),
            code=code,
            test=str(payload["test"]),
            entry_point=str(payload["entry_point"]),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            program_path = Path(tmp_dir) / "candidate.py"
            program_path.write_text(program, encoding="utf-8")
            try:
                result = subprocess.run(
                    [sys.executable, "-I", str(program_path)],
                    cwd=tmp_dir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False
        return result.returncode == 0

    def _build_code_generation_program(
        self,
        *,
        prompt: str,
        code: str,
        test: str,
        entry_point: str,
    ) -> str:
        candidate_code = self._strip_markdown_code_fence(code)
        return f"{prompt.rstrip()}\n{candidate_code.rstrip()}\n{test.rstrip()}\n\ncheck({entry_point})\n"

    def _strip_markdown_code_fence(self, code: str) -> str:
        stripped = code.strip()
        match = re.fullmatch(r"```(?:python|py)?\s*\n(?P<code>.*)\n```", stripped, re.DOTALL)
        if match:
            return match.group("code").strip()
        return code.strip("\r\n")

    def _build_summary(
        self,
        question_results: list[EvaluationQuestionResult],
        question_payloads: list[dict] | None = None,
    ) -> dict:
        total_questions = len(question_results)
        scored_results = [
            result for result in question_results if not result.excluded_from_accuracy
        ]
        evaluated_questions = len(scored_results)
        correct_questions = sum(1 for result in scored_results if result.is_correct)
        by_task_type: dict[str, dict[str, float | int]] = {}
        for result in question_results:
            bucket = by_task_type.setdefault(
                result.task_type,
                {
                    "total_questions": 0,
                    "evaluated_questions": 0,
                    "tie_question_count": 0,
                    "correct_questions": 0,
                    "accuracy": 0.0,
                },
            )
            bucket["total_questions"] += 1
            if result.excluded_from_accuracy:
                bucket["tie_question_count"] += 1
                continue
            bucket["evaluated_questions"] += 1
            if result.is_correct:
                bucket["correct_questions"] += 1
        for bucket in by_task_type.values():
            evaluated = int(bucket["evaluated_questions"])
            bucket["accuracy"] = (
                float(bucket["correct_questions"]) / evaluated if evaluated else 0.0
            )
        tie_question_keys = [
            result.question_key for result in question_results if result.is_tie
        ]
        summary: dict = {
            "total_questions": total_questions,
            "evaluated_questions": evaluated_questions,
            "tie_question_count": len(tie_question_keys),
            "tie_question_keys": tie_question_keys,
            "correct_questions": correct_questions,
            "accuracy": correct_questions / evaluated_questions if evaluated_questions else 0.0,
            "by_task_type": by_task_type,
        }
        if question_payloads is not None:
            answer_change_metrics = self._build_answer_change_metrics(question_payloads)
            summary["answer_change_metrics"] = answer_change_metrics
            adversarial_metrics = self._build_adversarial_metrics(
                question_results, question_payloads, answer_change_metrics
            )
            if adversarial_metrics is not None:
                summary["adversarial_metrics"] = adversarial_metrics
        return summary

    def _build_adversarial_metrics(
        self,
        question_results: list[EvaluationQuestionResult],
        question_payloads: list[dict],
        answer_change_metrics: dict | None = None,
    ) -> dict | None:
        """Compute adversarial-specific metrics from raw round data.

        Returns *None* when the run contains no adversarial agents (i.e. a
        normal, non-adversarial MAS run), so that the summary stays clean.
        """
        # ── Detect whether any adversarial agent exists in the run ──
        has_adversarial = False
        for payload in question_payloads:
            for round_data in payload.get("rounds", []):
                for agent_result in round_data.get("agent_results", []):
                    if self._is_adversarial(agent_result):
                        has_adversarial = True
                        break
                if has_adversarial:
                    break
            if has_adversarial:
                break
        if not has_adversarial:
            return None

        # ── 1. Attack effect metrics ──
        adversarial_selected_count = sum(
            1 for r in question_results if r.selected_is_adversarial
        )
        adversarial_attack_success_count = sum(
            1 for r in question_results
            if r.selected_is_adversarial and not r.is_correct
        )
        total = len(question_results) or 1

        # ── 2. Score manipulation metrics ──
        adversarial_scores_given: list[float] = []
        honest_scores_given: list[float] = []
        for payload in question_payloads:
            for round_data in payload.get("rounds", []):
                for agent_result in round_data.get("agent_results", []):
                    is_adv = self._is_adversarial(agent_result)
                    for review in agent_result.get("reviews_given", []):
                        score = float(review.get("score", 0))
                        if is_adv:
                            adversarial_scores_given.append(score)
                        else:
                            honest_scores_given.append(score)

        adversarial_avg = (
            sum(adversarial_scores_given) / len(adversarial_scores_given)
            if adversarial_scores_given
            else 0.0
        )
        honest_avg = (
            sum(honest_scores_given) / len(honest_scores_given)
            if honest_scores_given
            else 0.0
        )

        if answer_change_metrics is None:
            answer_change_metrics = self._build_answer_change_metrics(question_payloads)

        return {
            # Attack effect
            "adversarial_selected_count": adversarial_selected_count,
            "adversarial_selected_rate": adversarial_selected_count / total,
            "adversarial_attack_success_count": adversarial_attack_success_count,
            "adversarial_attack_success_rate": adversarial_attack_success_count / total,
            # Score manipulation
            "adversarial_avg_score_given": round(adversarial_avg, 2),
            "honest_avg_score_given": round(honest_avg, 2),
            "score_manipulation_gap": round(honest_avg - adversarial_avg, 2),
            # Answer influence
            **answer_change_metrics,
        }

    def _build_answer_change_metrics(self, question_payloads: list[dict]) -> dict:
        """Track non-adversarial agents changing answers across rounds."""
        honest_changed_count = 0
        changed_to_wrong_count = 0
        changed_to_correct_count = 0

        for payload in question_payloads:
            correct_ids = sorted(payload.get("correct_option_ids", []))
            is_code = payload.get("task_type") == "code_generation"
            rounds = payload.get("rounds", [])
            if len(rounds) < 2:
                continue

            # Build per-agent answer history across rounds
            for agent_id in self._honest_agent_ids(rounds):
                previous_option_ids: list[str] | None = None
                previous_code: str | None = None
                for round_data in sorted(
                    rounds, key=lambda r: int(r.get("round_index", 0))
                ):
                    agent_result = self._find_agent_result(
                        round_data, agent_id
                    )
                    if agent_result is None:
                        continue
                    answer = agent_result.get("answer", {})
                    current_option_ids = sorted(
                        answer.get("selected_option_ids", [])
                    )
                    current_code = str(answer.get("code", "")).strip()

                    if previous_option_ids is not None or previous_code is not None:
                        # Detect actual answer change
                        option_changed = (
                            previous_option_ids is not None
                            and current_option_ids != previous_option_ids
                        )
                        code_changed = (
                            is_code
                            and previous_code is not None
                            and current_code != previous_code
                        )
                        if option_changed or code_changed:
                            honest_changed_count += 1
                            if is_code:
                                # For code, we can't easily judge correctness
                                # from options alone, so skip wrong/correct
                                # bucketing for code tasks.
                                pass
                            elif current_option_ids == correct_ids:
                                changed_to_correct_count += 1
                            else:
                                changed_to_wrong_count += 1

                    previous_option_ids = current_option_ids
                    previous_code = current_code if is_code else None

        return {
            "honest_agents_changed_answer_count": honest_changed_count,
            "changed_to_wrong_count": changed_to_wrong_count,
            "changed_to_correct_count": changed_to_correct_count,
        }

    def _honest_agent_ids(self, rounds: list[dict]) -> set[str]:
        """Collect agent IDs of all honest agents across all rounds."""
        ids: set[str] = set()
        for round_data in rounds:
            for agent_result in round_data.get("agent_results", []):
                if not self._is_adversarial(agent_result):
                    ids.add(str(agent_result.get("agent_id", "")))
        return ids

    @staticmethod
    def _find_agent_result(
        round_data: dict, agent_id: str
    ) -> dict | None:
        """Find the agent_result for a specific agent in a round."""
        for agent_result in round_data.get("agent_results", []):
            if str(agent_result.get("agent_id", "")) == agent_id:
                return agent_result
        return None
