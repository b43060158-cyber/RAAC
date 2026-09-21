"""Build and merge a strategy-9-specific subset into a target reasoning bank.

This is a specialized orchestration script for deriving a filtered subset and
then merging the reviewed result into a chosen bank file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent1_influence_pipeline_utils import read_jsonl, write_jsonl
from scripts.reasoning_bank_pipeline.reasoning_bank_backup_utils import (
    create_reasoning_bank_backup_bundle,
)

RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_MAIN_BANK = RUNS_DIR / "adversarial_reasoning_bank_v2.jsonl"
DEFAULT_BACKUP_DIR = RUNS_DIR / "reasoning_bank_backups"


def _normalize_question_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isdigit():
        return text.zfill(4)
    return text


def _run_step(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def merge_bank_records(
    existing_records: list[dict[str, Any]],
    incoming_records: list[dict[str, Any]],
    *,
    merge_tag: str | None = None,
) -> list[dict[str, Any]]:
    tag = (merge_tag or datetime.now().strftime("%Y%m%d-%H%M%S")).strip()
    merged: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    for record in existing_records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if sample_id in seen_sample_ids:
            continue
        seen_sample_ids.add(sample_id)
        merged.append(record)

    for record in incoming_records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if sample_id in seen_sample_ids:
            renamed_record = dict(record)
            renamed_record["original_sample_id"] = sample_id
            suffix_index = 1
            candidate_sample_id = f"{sample_id}::merged_{tag}"
            while candidate_sample_id in seen_sample_ids:
                suffix_index += 1
                candidate_sample_id = f"{sample_id}::merged_{tag}_{suffix_index}"
            renamed_record["sample_id"] = candidate_sample_id
            record = renamed_record
            sample_id = candidate_sample_id
        seen_sample_ids.add(sample_id)
        merged.append(record)

    return merged


def _work_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "adv_filtered": work_dir / "agent1_choice_influence_adversarial_enriched.qids.json",
        "non_filtered": work_dir / "agent1_choice_influence_non_adversarial_enriched.qids.json",
        "candidates": work_dir / "agent1_influence_for_attribution.qids.jsonl",
        "open_factors": work_dir / "agent1_influence_open_factor_extraction.qids.jsonl",
        "standardized": work_dir / "agent1_influence_standardized_attribution.qids.jsonl",
        "review_generation": work_dir / "agent1_review_generation_dataset.qids.jsonl",
        "malicious_generation": work_dir / "agent1_malicious_review_generation_dataset.qids.jsonl",
        "subset_bank": work_dir / "adversarial_reasoning_bank_v2.qids.jsonl",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strategy9 reasoning-bank subset for specific question ids and merge it "
            "into the main adversarial_reasoning_bank_v2.jsonl."
        )
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        required=True,
        help="Question ids to include, e.g. 551 868 1328 1577 1885 3799.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=[],
        help="Optional run ids to include. If omitted, keep matching question ids from all runs.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/strategy9_subset_bank"),
        help="Temporary working directory for subset artifacts.",
    )
    parser.add_argument(
        "--main-bank",
        type=Path,
        default=DEFAULT_MAIN_BANK,
        help="Main strategy9 reasoning bank file to merge into.",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip refreshing the full extracted/enriched influence datasets from runs/.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Build the subset bank only, without merging into the main bank.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=DEFAULT_BACKUP_DIR,
        help="Directory for automatic pre-merge backup bundles.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the automatic backup bundle before updating the main bank.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Environment file passed to LLM-backed stages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    question_ids = [_normalize_question_id(question_id) for question_id in args.question_ids]
    question_ids = [question_id for question_id in question_ids if question_id]
    if not question_ids:
        raise ValueError("No valid question ids provided.")
    run_ids = [str(run_id).strip() for run_id in args.run_ids if str(run_id).strip()]

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = _work_paths(work_dir)

    if not args.skip_refresh:
        _run_step([sys.executable, "scripts/reasoning_bank_pipeline/extract_agent1_choice_influence_dataset.py"])
        _run_step([sys.executable, "scripts/reasoning_bank_pipeline/enrich_agent1_choice_influence_dataset.py"])

    filter_base = [
        sys.executable,
        "scripts/filter_agent1_records_by_question_ids.py",
        "--question-ids",
        *question_ids,
    ]
    if run_ids:
        filter_base.extend(["--run-ids", *run_ids])
    _run_step(
        filter_base
        + [
            "--input",
            str(RUNS_DIR / "agent1_choice_influence_adversarial_enriched.json"),
            "--output",
            str(paths["adv_filtered"]),
        ]
    )
    _run_step(
        filter_base
        + [
            "--input",
            str(RUNS_DIR / "agent1_choice_influence_non_adversarial_enriched.json"),
            "--output",
            str(paths["non_filtered"]),
        ]
    )

    _run_step(
        [
            sys.executable,
            "scripts/reasoning_bank_pipeline/build_agent1_attribution_candidates.py",
            "--non-input",
            str(paths["non_filtered"]),
            "--adversarial-input",
            str(paths["adv_filtered"]),
            "--output",
            str(paths["candidates"]),
        ]
    )
    _run_step(
        [
            sys.executable,
            "scripts/reasoning_bank_pipeline/run_agent1_open_factor_extraction.py",
            "--input",
            str(paths["candidates"]),
            "--output",
            str(paths["open_factors"]),
            "--env",
            str(args.env),
        ]
    )
    _run_step(
        [
            sys.executable,
            "scripts/reasoning_bank_pipeline/apply_agent1_factor_taxonomy.py",
            "--input",
            str(paths["open_factors"]),
            "--output",
            str(paths["standardized"]),
            "--env",
            str(args.env),
        ]
    )
    _run_step(
        [
            sys.executable,
            "scripts/reasoning_bank_pipeline/build_agent1_review_generation_dataset.py",
            "--input",
            str(paths["standardized"]),
            "--output",
            str(paths["review_generation"]),
        ]
    )
    _run_step(
        [
            sys.executable,
            "scripts/reasoning_bank_pipeline/rewrite_agent1_malicious_generation_dataset.py",
            "--input",
            str(paths["review_generation"]),
            "--output",
            str(paths["malicious_generation"]),
            "--env",
            str(args.env),
        ]
    )
    _run_step(
        [
            sys.executable,
            "scripts/reasoning_bank_pipeline/build_adversarial_reasoning_bank_v2.py",
            "--attribution-input",
            str(paths["standardized"]),
            "--malicious-input",
            str(paths["malicious_generation"]),
            "--enriched-input",
            str(paths["adv_filtered"]),
            "--enriched-input",
            str(paths["non_filtered"]),
            "--output",
            str(paths["subset_bank"]),
        ]
    )

    print(f"subset_bank={paths['subset_bank']}")

    if args.skip_merge:
        return 0

    main_bank = args.main_bank.expanduser().resolve()
    if not args.no_backup:
        backup_path = create_reasoning_bank_backup_bundle(
            target_path=main_bank,
            source_paths=[
                RUNS_DIR / "agent1_influence_standardized_attribution.jsonl",
                RUNS_DIR / "agent1_malicious_review_generation_dataset.jsonl",
                RUNS_DIR / "agent1_choice_influence_adversarial_enriched.json",
                RUNS_DIR / "agent1_choice_influence_non_adversarial_enriched.json",
                paths["subset_bank"],
            ],
            backup_dir=args.backup_dir,
            label="merge",
        )
        if backup_path is not None:
            print(f"backup_bundle={backup_path}")
    existing_records = read_jsonl(main_bank)
    incoming_records = read_jsonl(paths["subset_bank"])
    merged_records = merge_bank_records(existing_records, incoming_records)
    write_jsonl(main_bank, merged_records)

    print(f"main_bank={main_bank}")
    print(f"existing_record_count={len(existing_records)}")
    print(f"incoming_record_count={len(incoming_records)}")
    print(f"merged_record_count={len(merged_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
