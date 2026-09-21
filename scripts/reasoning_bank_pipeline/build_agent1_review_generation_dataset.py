"""Stage 6: convert standardized attribution into generation examples.

This packages each successful persuasion case as an input/output pair for
review-generation and later malicious rewrite stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent1_influence_pipeline_utils import read_jsonl, write_jsonl
RUNS_DIR = PROJECT_ROOT / "runs"
INPUT = RUNS_DIR / "agent1_influence_standardized_attribution.jsonl"
OUTPUT = RUNS_DIR / "agent1_review_generation_dataset.jsonl"


def _generation_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "input": {
            "question_text": record.get("question_text"),
            "options": record.get("options", []),
            "target_previous_selected_option_ids": record.get("target_previous_selected_option_ids", []),
            "target_previous_reasoning": record.get("target_previous_reasoning", ""),
            "desired_target_shift": {
                "from": record.get("target_previous_selected_option_ids", []),
                "to": record.get("target_new_selected_option_ids", []),
            },
            "primary_factor_id": record.get("primary_factor_id", ""),
            "secondary_factor_ids": record.get("secondary_factor_ids", []),
            "supporting_evidence": record.get("supporting_evidence", []),
            "run_type": record.get("run_type"),
            "dataset_name": record.get("dataset_name"),
        },
        "output": {
            "review_rationale": record.get("agent1_review_text", ""),
        },
        "metadata": {
            "question_key": record.get("question_key"),
            "target_agent_id": record.get("target_agent_id"),
            "factor_mapping_confidence": record.get("factor_mapping_confidence"),
            "mapping_notes": record.get("mapping_notes", ""),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the stage-4 review generation dataset.")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.input)
    generation_records = [_generation_record(record) for record in records]
    write_jsonl(args.output, generation_records)
    print(f"output={args.output}")
    print(f"record_count={len(generation_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
