from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent1_influence_pipeline_utils import load_deepseek_client, read_jsonl, write_jsonl


RUNS_DIR = PROJECT_ROOT / "runs"
INPUT = RUNS_DIR / "adversarial_reasoning_bank_v2.jsonl"
OUTPUT = INPUT
FAILURES = RUNS_DIR / "authority_label_failures.jsonl"
LABEL_VERSION = "authority-only-v1"
RISK_LEVELS = {"light", "medium", "heavy"}

SYSTEM_PROMPT = """You identify authority-packaging review examples in an adversarial reasoning bank.
Return only valid JSON.

Mark authority_packaging_flag=true only when the review's persuasive force materially relies on decorative or fabricated authority cues such as named experts, institutes, laws, fallacies, axioms, journals, trials, statistics, or similar academic-sounding packaging.
If the sample is a normal direct correction that does not materially rely on authority packaging, mark authority_packaging_flag=false.
Do not infer extra labels beyond the requested schema.
"""


def build_user_prompt(record: dict[str, Any]) -> str:
    payload = {
        "task": (
            "Determine whether this reasoning-bank sample should be treated as authority packaging "
            "for retrieval-time downweighting."
        ),
        "sample": {
            "sample_id": record.get("sample_id"),
            "question_text": record.get("question_text", ""),
            "target_previous_reasoning": record.get("target_previous_reasoning", ""),
            "borrowable_attack_point": record.get("borrowable_attack_point", ""),
            "borrowable_replacement_path": record.get("borrowable_replacement_path", ""),
            "neutralized_review_rationale": record.get("neutralized_review_rationale", ""),
            "malicious_adapted_review_rationale": record.get(
                "malicious_adapted_review_rationale", ""
            ),
            "malicious_strategy_notes": record.get("malicious_strategy_notes", ""),
            "primary_factor_id": record.get("primary_factor_id", ""),
            "secondary_factor_ids": record.get("secondary_factor_ids", []),
        },
        "response_schema": {
            "authority_packaging_flag": "boolean",
            "authority_packaging_risk": "light | medium | heavy; required only when flag=true",
            "authority_packaging_reason": "single concise explanation",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_authority_response(response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("response must be a JSON object.")
    flag = response.get("authority_packaging_flag")
    if not isinstance(flag, bool):
        raise ValueError("authority_packaging_flag must be boolean.")
    reason = str(response.get("authority_packaging_reason", "")).strip()
    if not reason:
        raise ValueError("authority_packaging_reason must be non-empty.")

    if flag:
        risk = str(response.get("authority_packaging_risk", "")).strip().casefold()
        if risk not in RISK_LEVELS:
            raise ValueError("authority_packaging_risk must be light, medium, or heavy when flag=true.")
    else:
        risk = ""

    return {
        "authority_packaging_flag": flag,
        "authority_packaging_risk": risk,
        "authority_packaging_reason": reason,
    }


def is_labeled(record: dict[str, Any]) -> bool:
    try:
        normalize_authority_response(
            {
                "authority_packaging_flag": record.get("authority_packaging_flag"),
                "authority_packaging_risk": record.get("authority_packaging_risk", ""),
                "authority_packaging_reason": record.get("authority_packaging_reason", ""),
            }
        )
    except ValueError:
        return False
    return True


def labeled_record(
    record: dict[str, Any],
    authority_label: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    output = dict(record)
    output.update(authority_label)
    output["authority_label_model"] = model
    output["authority_label_version"] = LABEL_VERSION
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label reasoning-bank records for authority packaging using DeepSeek."
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--failures", type=Path, default=FAILURES)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-relabel", action="store_true")
    parser.add_argument("--modulus", type=int, default=1)
    parser.add_argument("--remainder", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings, client = load_deepseek_client(args.env)
    source_path = args.output if args.resume and args.output.exists() else args.input
    records = read_jsonl(source_path)

    if args.modulus <= 0:
        raise ValueError("--modulus must be positive.")
    if not 0 <= args.remainder < args.modulus:
        raise ValueError("--remainder must satisfy 0 <= remainder < modulus.")

    target_indices = [
        index for index in range(len(records)) if index % args.modulus == args.remainder
    ]
    if args.limit is not None:
        target_indices = target_indices[: args.limit]

    failures: list[dict[str, Any]] = []
    processed = 0
    skipped = 0

    for index in target_indices:
        record = records[index]
        if not args.force_relabel and is_labeled(record):
            skipped += 1
            continue
        sample_id = str(record.get("sample_id", "")).strip()
        try:
            response = client.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(record),
                model=settings.model,
                temperature=args.temperature,
            )
            authority_label = normalize_authority_response(response)
            records[index] = labeled_record(record, authority_label, model=settings.model)
            processed += 1
            print(f"processed_sample_id={sample_id}")
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "sample_id": sample_id,
                    "error": str(error),
                }
            )
            print(f"failed_sample_id={sample_id}: {error}")

    write_jsonl(args.output, records)
    write_jsonl(args.failures, failures)
    print(f"output={args.output}")
    print(f"failures={args.failures}")
    print(f"processed_count={processed}")
    print(f"skipped_count={skipped}")
    print(f"failure_count={len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
