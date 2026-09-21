from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent1_influence_pipeline_utils import chunked, load_deepseek_client, read_jsonl, write_json
RUNS_DIR = PROJECT_ROOT / "runs"
INPUT = RUNS_DIR / "agent1_influence_open_factor_extraction.jsonl"
OUTPUT = RUNS_DIR / "agent1_influence_factor_taxonomy.json"
SUMMARIES_OUTPUT = RUNS_DIR / "agent1_influence_factor_taxonomy_batch_summaries.jsonl"

SYSTEM_PROMPT = """You induce a factor taxonomy from open-ended persuasion analyses.
Return only valid JSON.
Merge semantically similar factors. Prefer compact, descriptive factor names.
"""


def _batch_prompt(records: list[dict]) -> str:
    compact = [
        {
            "sample_id": record.get("sample_id"),
            "primary": record.get("most_likely_primary_factor"),
            "factors": list(record.get("influence_factors_freeform", []))[:6],
            "summary": str(record.get("why_this_review_likely_worked", ""))[:220],
        }
        for record in records
    ]
    payload = {
        "task": "Summarize recurring influence factors within this batch and propose candidate factor groups.",
        "records": compact,
        "response_schema": {
            "candidate_factors": [
                {
                    "factor_name": "short name",
                    "definition": "1-2 sentences",
                    "surface_forms": ["freeform expressions seen in this batch"],
                    "representative_sample_ids": ["sample ids"],
                }
            ],
            "batch_notes": "short summary",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _compressed_batch_summaries(batch_summaries: list[dict]) -> list[dict]:
    compressed: list[dict] = []
    for summary in batch_summaries:
        candidate_factors = []
        for factor in summary.get("candidate_factors", []):
            if not isinstance(factor, dict):
                continue
            candidate_factors.append(
                {
                    "factor_name": factor.get("factor_name", ""),
                    "definition": factor.get("definition", ""),
                    "surface_forms": list(factor.get("surface_forms", []))[:5],
                    "representative_sample_ids": list(factor.get("representative_sample_ids", []))[:3],
                }
            )
        compressed.append(
            {
                "batch_index": summary.get("batch_index"),
                "candidate_factors": candidate_factors,
                "batch_notes": summary.get("batch_notes", ""),
            }
        )
    return compressed


def _final_prompt(batch_summaries: list[dict]) -> str:
    payload = {
        "task": (
            "Merge the batch-level candidate factors into a single reusable taxonomy for successful "
            "review influence in single-choice answer changes."
        ),
        "batch_summaries": _compressed_batch_summaries(batch_summaries),
        "response_schema": {
            "taxonomy_version": "string",
            "factors": [
                {
                    "factor_id": "factor_01 style id",
                    "factor_name": "short stable name",
                    "definition": "1-3 sentences",
                    "inclusion_criteria": ["signals that belong to this factor"],
                    "exclusion_criteria": ["signals that should not be grouped here"],
                    "surface_forms": ["typical freeform forms"],
                    "representative_sample_ids": ["sample ids"],
                }
            ],
            "global_notes": "short note",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Induce a factor taxonomy from open factor extraction results.")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summaries-output", type=Path, default=SUMMARIES_OUTPUT)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--reuse-summaries", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings, client = load_deepseek_client(args.env)
    records = read_jsonl(args.input)
    if args.reuse_summaries and args.summaries_output.exists():
        summaries = read_jsonl(args.summaries_output)
    else:
        summaries = []
        args.summaries_output.unlink(missing_ok=True)
        for index, batch in enumerate(chunked(records, args.batch_size), start=1):
            summary = client.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_batch_prompt(batch),
                model=settings.model,
                temperature=args.temperature,
            )
            summary["batch_index"] = index
            summaries.append(summary)
            with args.summaries_output.open("a", encoding="utf-8") as target:
                target.write(json.dumps(summary, ensure_ascii=False) + "\n")
            print(f"completed_batch={index}")

    taxonomy = client.generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_final_prompt(summaries),
        model=settings.model,
        temperature=args.temperature,
    )
    taxonomy["source_record_count"] = len(records)
    taxonomy["batch_summary_count"] = len(summaries)
    write_json(args.output, taxonomy)
    print(f"output={args.output}")
    print(f"factor_count={len(taxonomy.get('factors', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
