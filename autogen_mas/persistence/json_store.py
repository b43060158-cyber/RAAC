from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from autogen_mas.config import DashScopeSettings, ExperimentConfig
from autogen_mas.models import QuestionRunResult


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return slug.strip("_") or "question"


class JsonRunStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def initialize_run(
        self,
        *,
        run_id: str,
        experiment_config: ExperimentConfig,
        llm_settings: DashScopeSettings,
    ) -> Path:
        run_path = self.output_dir / run_id
        (run_path / "questions").mkdir(parents=True, exist_ok=True)
        (run_path / "evaluation").mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "experiment_config": experiment_config.snapshot(),
            "llm_settings": llm_settings.snapshot(),
        }
        self._write_json(run_path / "manifest.json", manifest)
        return run_path

    def persist_question_result(
        self,
        run_path: str | Path,
        result: QuestionRunResult,
        *,
        persist_feedback_summary: bool = False,
    ) -> str:
        path = self.question_result_path(run_path, result.question_key)
        self._write_json(path, result.to_dict())
        if persist_feedback_summary:
            self.persist_feedback_summary(run_path, result)
        return str(path)

    def question_result_path(self, run_path: str | Path, question_key: str) -> Path:
        return Path(run_path) / "questions" / f"{_slugify(question_key)}.json"

    def feedback_summary_path(self, run_path: str | Path, question_key: str) -> Path:
        return (
            Path(run_path)
            / "debug"
            / "feedback_summaries"
            / f"{_slugify(question_key)}.json"
        )

    def persist_feedback_summary(
        self,
        run_path: str | Path,
        result: QuestionRunResult,
    ) -> str:
        path = self.feedback_summary_path(run_path, result.question_key)
        entries: list[dict[str, object]] = []
        for previous_round, current_round in zip(result.rounds, result.rounds[1:]):
            previous_by_agent = previous_round.agent_map()
            for agent_result in current_round.agent_results:
                previous_result = previous_by_agent.get(agent_result.agent_id)
                if previous_result is None:
                    continue
                prior_feedback = previous_result.to_prior_feedback().to_dict()
                entries.append(
                    {
                        "round_index": current_round.round_index,
                        "previous_round_index": previous_round.round_index,
                        "agent_id": agent_result.agent_id,
                        "prior_feedback": prior_feedback,
                    }
                )
        self._write_json(
            path,
            {
                "run_id": result.run_id,
                "question_key": result.question_key,
                "items": entries,
            },
        )
        return str(path)

    def read_manifest(self, run_path: str | Path) -> dict | None:
        path = Path(run_path) / "manifest.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def read_question_payload(
        self,
        run_path: str | Path,
        question_key: str,
    ) -> dict | None:
        path = self.question_result_path(run_path, question_key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def persist_evaluation(
        self,
        run_path: str | Path,
        *,
        question_results: list[dict],
        summary: dict,
        evaluation_name: str | None = None,
        write_legacy: bool = False,
    ) -> tuple[str, str]:
        evaluation_root = Path(run_path) / "evaluation"
        evaluation_dir = (
            evaluation_root / _slugify(evaluation_name)
            if evaluation_name is not None
            else evaluation_root
        )
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        question_results_path = evaluation_dir / "question_results.json"
        summary_path = evaluation_dir / "summary.json"
        self._write_json(question_results_path, question_results)
        self._write_json(summary_path, summary)
        if write_legacy:
            self._write_json(evaluation_root / "question_results.json", question_results)
            self._write_json(evaluation_root / "summary.json", summary)
        return str(question_results_path), str(summary_path)

    def persist_single_agent_summary(
        self,
        run_path: str | Path,
        summary: dict,
    ) -> str:
        summary_path = Path(run_path) / "evaluation" / "single_agent_summary.json"
        self._write_json(summary_path, summary)
        return str(summary_path)

    def persist_round_sweep_summary(
        self,
        run_path: str | Path,
        summary: dict,
    ) -> str:
        summary_path = Path(run_path) / "evaluation" / "round_sweep" / "summary.json"
        self._write_json(summary_path, summary)
        return str(summary_path)

    def failed_questions_path(self, run_path: str | Path) -> Path:
        return Path(run_path) / "failed_questions.json"

    def persist_failed_questions(
        self,
        run_path: str | Path,
        failures: list[dict[str, object]],
    ) -> str:
        path = self.failed_questions_path(run_path)
        self._write_json(path, failures)
        return str(path)

    def read_failed_questions(self, run_path: str | Path) -> list[dict] | None:
        path = self.failed_questions_path(run_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, list):
            return None
        return [item for item in payload if isinstance(item, dict)]

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
