"""Stage 5: map open-ended attribution output onto the shared taxonomy.

This normalizes freeform factor descriptions into stable factor ids that can
be counted, filtered, and reused downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent1_influence_pipeline_utils import append_jsonl, load_deepseek_client, load_json, read_jsonl
RUNS_DIR = PROJECT_ROOT / "runs"
INPUT = RUNS_DIR / "agent1_influence_open_factor_extraction.jsonl"
TAXONOMY = RUNS_DIR / "agent1_influence_factor_taxonomy.json"
OUTPUT = RUNS_DIR / "agent1_influence_standardized_attribution.jsonl"

SYSTEM_PROMPT = """You map open-ended persuasion analyses to a provided factor taxonomy.
Return only valid JSON.
Use only taxonomy factors that genuinely fit the evidence.
"""


def _user_prompt(record: dict, taxonomy: dict) -> str:
    payload = {
        "task": "Map this sample to the taxonomy and cite supporting evidence.",
        "taxonomy": taxonomy,
        "sample": {
            "sample_id": record.get("sample_id"),
            "question_text": record.get("question_text"),
            "target_previous_reasoning": record.get("target_previous_reasoning"),
            "agent1_review_text": record.get("agent1_review_text"),
            "target_new_reasoning": record.get("target_new_reasoning"),
            "target_change_summary": record.get("target_change_summary"),
            "open_factor_extraction": {
                "influence_factors_freeform": record.get("influence_factors_freeform", []),
                "most_likely_primary_factor": record.get("most_likely_primary_factor"),
                "why_this_review_likely_worked": record.get("why_this_review_likely_worked"),
            },
        },
        "response_schema": {
            "primary_factor_id": "factor id or empty string",
            "secondary_factor_ids": ["zero or more factor ids"],
            "supporting_evidence": ["short evidence spans from the sample"],
            "factor_mapping_confidence": "float 0.0-1.0",
            "mapping_notes": "brief explanation",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply induced factor taxonomy to open factor extraction results.")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY)
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
    taxonomy = load_json(args.taxonomy)
    records = read_jsonl(args.input)
    if args.modulus <= 0:
        raise ValueError("--modulus must be positive.")
    if not 0 <= args.remainder < args.modulus:
        raise ValueError("--remainder must satisfy 0 <= remainder < modulus.")
    records = [record for index, record in enumerate(records) if index % args.modulus == args.remainder]
    if args.limit is not None:
        records = records[: args.limit]

    existing_ids = set()
    if args.resume and args.output.exists():
        existing_ids = {str(item.get("sample_id")) for item in read_jsonl(args.output)}
    if not args.resume and args.output.exists():
        args.output.unlink()

    processed = 0
    for record in records:
        sample_id = str(record.get("sample_id"))
        if sample_id in existing_ids:
            continue
        response = client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_user_prompt(record, taxonomy),
            model=settings.model,
            temperature=args.temperature,
        )
        output_record = dict(record)
        output_record.update(response)
        append_jsonl(args.output, output_record)
        processed += 1
        print(f"processed_sample_id={sample_id}")

    print(f"output={args.output}")
    print(f"processed_count={processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
