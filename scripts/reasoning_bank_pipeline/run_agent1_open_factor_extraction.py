"""Stage 4: run open-ended LLM attribution over candidate samples.

For each candidate, this script asks the model why the review likely worked
and records freeform influence factors plus evidence spans.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent1_influence_pipeline_utils import (
    append_jsonl,
    load_deepseek_client,
    options_as_text,
    read_jsonl,
)
RUNS_DIR = PROJECT_ROOT / "runs"
INPUT = RUNS_DIR / "agent1_influence_for_attribution.jsonl"
OUTPUT = RUNS_DIR / "agent1_influence_open_factor_extraction.jsonl"

SYSTEM_PROMPT = """You analyze successful review persuasion cases in a multi-agent reasoning setting.
Return only valid JSON.
Do not invent facts not present in the sample.
Use concise evidence-grounded language.
"""


def _user_prompt(record: dict[str, Any]) -> str:
    payload = {
        "task": (
            "Extract why the review likely influenced the target agent to change its selected option. "
            "Use open-ended factor names rather than a fixed label set. Cite only evidence that appears "
            "in the provided sample."
        ),
        "sample": {
            "sample_id": record["sample_id"],
            "question_text": record["question_text"],
            "options": options_as_text(record.get("options", [])),
            "target_previous_selected_option_ids": record.get("target_previous_selected_option_ids", []),
            "target_previous_reasoning": record.get("target_previous_reasoning", ""),
            "agent1_review_text": record.get("agent1_review_text", ""),
            "target_new_selected_option_ids": record.get("target_new_selected_option_ids", []),
            "target_new_reasoning": record.get("target_new_reasoning", ""),
            "target_change_summary": record.get("target_change_summary", ""),
            "review_to_new_reasoning_overlap": record.get("review_to_new_reasoning_overlap", []),
            "review_targets_previous_error_evidence": record.get("review_targets_previous_error_evidence", []),
            "review_proposes_replacement_evidence": record.get("review_proposes_replacement_evidence", []),
            "change_summary_alignment_evidence": record.get("change_summary_alignment_evidence", []),
            "alignment_notes": record.get("alignment_notes", []),
        },
        "response_schema": {
            "influence_factors_freeform": ["list of freeform factor names"],
            "most_likely_primary_factor": "single short phrase",
            "evidence_spans_from_review": ["short spans copied from review text"],
            "evidence_spans_from_new_reasoning": ["short spans copied from new reasoning"],
            "evidence_spans_from_change_summary": ["short spans copied from change summary"],
            "why_this_review_likely_worked": "2-4 sentences",
            "confidence": "float 0.0-1.0",
            "uncertainty_notes": "brief note or empty string",
        },
        "constraints": [
            "Return valid JSON only.",
            "Keep factor names open-ended and data-driven.",
            "Evidence spans must quote only text that exists in the sample.",
            "Do not use a predefined taxonomy.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek open factor extraction on attribution candidates.")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--modulus", type=int, default=1)
    parser.add_argument("--remainder", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings, client = load_deepseek_client(args.env)
    existing_ids = set()
    if args.resume and args.output.exists():
        existing_ids = {str(item.get("sample_id")) for item in read_jsonl(args.output)}

    records = read_jsonl(args.input)
    if args.modulus <= 0:
        raise ValueError("--modulus must be positive.")
    if not 0 <= args.remainder < args.modulus:
        raise ValueError("--remainder must satisfy 0 <= remainder < modulus.")
    records = [record for index, record in enumerate(records) if index % args.modulus == args.remainder]
    if args.limit is not None:
        records = records[: args.limit]
    pending = [record for record in records if str(record.get("sample_id")) not in existing_ids]

    if not args.resume and args.output.exists():
        args.output.unlink()

    processed = 0
    for record in pending:
        response = client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_user_prompt(record),
            model=settings.model,
            temperature=args.temperature,
        )
        output_record = dict(record)
        output_record.update(response)
        append_jsonl(args.output, output_record)
        processed += 1
        print(f"processed_sample_id={record['sample_id']}")

    print(f"output={args.output}")
    print(f"processed_count={processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
