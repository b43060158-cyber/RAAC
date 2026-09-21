"""Merge a manually approved incremental candidate into the main bank.

This script creates a backup, appends reviewed records, and updates the
candidate manifest to reflect the completed merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent1_influence_pipeline_utils import read_jsonl, write_jsonl
from scripts.reasoning_bank_pipeline.build_and_merge_strategy9_subset_bank import (
    merge_bank_records,
)
from scripts.reasoning_bank_pipeline.reasoning_bank_backup_utils import (
    DEFAULT_BACKUP_DIR,
    RUNS_DIR,
    create_reasoning_bank_backup_bundle,
)

DEFAULT_MAIN_BANK = RUNS_DIR / "adversarial_reasoning_bank_v2.jsonl"


def _load_manifest(candidate_dir: Path) -> dict:
    manifest_path = candidate_dir / "candidate_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"candidate_manifest.json not found in {candidate_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a reviewed incremental reasoning-bank candidate into the main bank. "
            "This command requires explicit approval."
        )
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--main-bank", type=Path, default=DEFAULT_MAIN_BANK)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument(
        "--approved",
        action="store_true",
        help="Explicit confirmation that the candidate supplement has been manually reviewed.",
    )
    parser.add_argument("--merge-tag", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.approved:
        raise ValueError("Refusing to merge without --approved. Review the candidate artifacts first.")

    candidate_dir = args.candidate_dir.expanduser().resolve()
    manifest = _load_manifest(candidate_dir)
    candidate_bank = candidate_dir / "adversarial_reasoning_bank_v2_candidate.jsonl"
    if not candidate_bank.exists():
        raise FileNotFoundError(f"Candidate bank not found: {candidate_bank}")

    main_bank = args.main_bank.expanduser().resolve()
    backup_path = create_reasoning_bank_backup_bundle(
        target_path=main_bank,
        source_paths=[
            RUNS_DIR / "agent1_influence_standardized_attribution.jsonl",
            RUNS_DIR / "agent1_malicious_review_generation_dataset.jsonl",
            RUNS_DIR / "agent1_choice_influence_adversarial_enriched.json",
            RUNS_DIR / "agent1_choice_influence_non_adversarial_enriched.json",
            candidate_bank,
        ],
        backup_dir=args.backup_dir,
        label="reviewed_merge",
    )
    if backup_path is not None:
        print(f"backup_bundle={backup_path}")

    existing_records = read_jsonl(main_bank)
    incoming_records = read_jsonl(candidate_bank)
    merged_records = merge_bank_records(
        existing_records,
        incoming_records,
        merge_tag=args.merge_tag or None,
    )
    write_jsonl(main_bank, merged_records)

    manifest["status"] = "merged"
    manifest["merged_into"] = str(main_bank)
    candidate_manifest_path = candidate_dir / "candidate_manifest.json"
    candidate_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"main_bank={main_bank}")
    print(f"candidate_bank={candidate_bank}")
    print(f"existing_record_count={len(existing_records)}")
    print(f"incoming_record_count={len(incoming_records)}")
    print(f"merged_record_count={len(merged_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
