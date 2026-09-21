"""Stage 3: flatten enriched influence samples into attribution candidates.

The output is a JSONL file with the fields needed by the LLM-based attribution
steps that follow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent1_influence_pipeline_utils import load_json, write_jsonl
RUNS_DIR = PROJECT_ROOT / "runs"
NON_INPUT = RUNS_DIR / "agent1_choice_influence_non_adversarial_enriched.json"
ADV_INPUT = RUNS_DIR / "agent1_choice_influence_adversarial_enriched.json"
OUTPUT = RUNS_DIR / "agent1_influence_for_attribution.jsonl"


def _candidate_record(record: dict[str, Any], source_split: str) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "source_split": source_split,
        "run_id": record.get("run_id"),
        "run_type": record.get("run_type"),
        "dataset_name": record.get("dataset_name"),
        "task_type": record.get("task_type"),
        "effective_task_type": record.get("effective_task_type", record.get("task_type")),
        "question_key": record.get("question_key"),
        "question_id": record.get("question_id"),
        "target_agent_id": record.get("target_agent_id"),
        "question_text": record.get("question_text"),
        "options": record.get("options", []),
        "target_previous_selected_option_ids": record.get("target_previous_selected_option_ids", []),
        "target_previous_reasoning": record.get("target_previous_reasoning", ""),
        "agent1_review_text": record.get("agent1_review_text", ""),
        "agent1_review_stance": record.get("agent1_review_stance", ""),
        "agent1_review_score": record.get("agent1_review_score"),
        "target_new_selected_option_ids": record.get("target_new_selected_option_ids", []),
        "target_new_reasoning": record.get("target_new_reasoning", ""),
        "target_change_summary": record.get("target_change_summary", ""),
        "review_to_new_reasoning_overlap": record.get("review_to_new_reasoning_overlap", []),
        "review_targets_previous_error_evidence": record.get("review_targets_previous_error_evidence", []),
        "review_proposes_replacement_evidence": record.get("review_proposes_replacement_evidence", []),
        "change_summary_alignment_evidence": record.get("change_summary_alignment_evidence", []),
        "alignment_notes": record.get("alignment_notes", []),
    }


def build_candidates(non_payload: dict[str, Any], adv_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_split, payload in (
        ("non_adversarial", non_payload),
        ("adversarial", adv_payload),
    ):
        for record in payload.get("records", []):
            if isinstance(record, dict):
                records.append(_candidate_record(record, source_split))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build attribution candidate JSONL from enriched agent1 datasets.")
    parser.add_argument("--non-input", type=Path, default=NON_INPUT)
    parser.add_argument("--adversarial-input", type=Path, default=ADV_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = build_candidates(load_json(args.non_input), load_json(args.adversarial_input))
    write_jsonl(args.output, candidates)
    print(f"output={args.output}")
    print(f"record_count={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
