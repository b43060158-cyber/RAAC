"""End-to-end driver for building a review-only incremental bank candidate.

Given selected run directories and question ids, this orchestrates the full
pipeline and writes a candidate bundle that must be manually reviewed before
merge.
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

from scripts.reasoning_bank_pipeline.extract_agent1_choice_influence_dataset import (
    _load_json,
    _normalize_text,
    _run_type,
    build_dataset,
    extract_records_from_question,
)
from scripts.reasoning_bank_pipeline.reasoning_bank_backup_utils import (
    DEFAULT_BACKUP_DIR,
    RUNS_DIR,
    create_reasoning_bank_backup_bundle,
)

DEFAULT_MAIN_BANK = RUNS_DIR / "adversarial_reasoning_bank_v2.jsonl"
DEFAULT_TAXONOMY = RUNS_DIR / "agent1_influence_factor_taxonomy.json"
DEFAULT_WORK_ROOT = RUNS_DIR / "reasoning_bank_incremental_candidates"


def _normalize_question_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isdigit():
        return text.zfill(4)
    return text


def _normalize_dataset_name(value: object) -> str:
    return _normalize_text(value).lower()


def _question_id_from_payload(payload: dict[str, Any]) -> str:
    question_id = _normalize_question_id(payload.get("question_id"))
    if question_id:
        return question_id
    question_key = _normalize_text(payload.get("question_key"))
    parts = question_key.split("__")
    if len(parts) >= 3:
        return _normalize_question_id(parts[1])
    return ""


def _scan_selected_records(
    *,
    run_paths: list[Path],
    dataset_names: set[str],
    question_ids: set[str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    non_adversarial_records: list[dict[str, Any]] = []
    adversarial_records: list[dict[str, Any]] = []

    for run_path in run_paths:
        run_type = _run_type(run_path.name)
        if run_type is None:
            continue
        question_dir = run_path / "questions"
        if not question_dir.exists():
            continue

        for question_file in sorted(question_dir.glob("*.json")):
            payload = _load_json(question_file)
            if _normalize_dataset_name(payload.get("dataset_name")) not in dataset_names:
                continue
            if question_ids is not None and _question_id_from_payload(payload) not in question_ids:
                continue
            records = extract_records_from_question(
                payload,
                run_id=run_path.name,
                run_type=run_type,
            )
            if run_type == "adversarial":
                adversarial_records.extend(records)
            else:
                non_adversarial_records.extend(records)

    return build_dataset(non_adversarial_records), build_dataset(adversarial_records)


def _run_step(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_paths(candidate_dir: Path) -> dict[str, Path]:
    return {
        "non_raw": candidate_dir / "agent1_choice_influence_non_adversarial.selected.json",
        "adv_raw": candidate_dir / "agent1_choice_influence_adversarial.selected.json",
        "non_enriched": candidate_dir / "agent1_choice_influence_non_adversarial_enriched.selected.json",
        "adv_enriched": candidate_dir / "agent1_choice_influence_adversarial_enriched.selected.json",
        "candidates": candidate_dir / "agent1_influence_for_attribution.selected.jsonl",
        "open_factors": candidate_dir / "agent1_influence_open_factor_extraction.selected.jsonl",
        "standardized": candidate_dir / "agent1_influence_standardized_attribution.selected.jsonl",
        "review_generation": candidate_dir / "agent1_review_generation_dataset.selected.jsonl",
        "malicious_generation": candidate_dir / "agent1_malicious_review_generation_dataset.selected.jsonl",
        "candidate_bank": candidate_dir / "adversarial_reasoning_bank_v2_candidate.jsonl",
        "manifest": candidate_dir / "candidate_manifest.json",
        "review_notes": candidate_dir / "REVIEW_REQUIRED.md",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an incremental reasoning-bank candidate from specific run paths, dataset names, "
            "and question ids. The result is review-only and is not merged into the main bank."
        )
    )
    parser.add_argument(
        "--run-paths",
        nargs="+",
        type=Path,
        required=True,
        help="One or more concrete run directories to mine, e.g. runs/adversarial-competition-20260526-123456.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        required=True,
        help="Dataset names to keep, e.g. medmcqa.",
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Question ids to keep, e.g. 551 868 1328.",
    )
    parser.add_argument(
        "--all-questions-in-dataset",
        action="store_true",
        help="Ignore question-id filtering and include all questions for the selected dataset(s) within the provided run(s).",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=DEFAULT_WORK_ROOT,
        help="Root directory where review candidate folders are created.",
    )
    parser.add_argument(
        "--candidate-name",
        type=str,
        default="",
        help="Optional explicit candidate folder name. Defaults to a timestamped generated name.",
    )
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--main-bank", type=Path, default=DEFAULT_MAIN_BANK)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_paths = [path.expanduser().resolve() for path in args.run_paths]
    for run_path in run_paths:
        if not run_path.exists():
            raise FileNotFoundError(f"Run path not found: {run_path}")

    dataset_names = {
        _normalize_dataset_name(dataset_name)
        for dataset_name in args.dataset_names
        if _normalize_dataset_name(dataset_name)
    }
    if not dataset_names:
        raise ValueError("No valid dataset names provided.")
    question_ids: set[str] | None
    if args.all_questions_in_dataset:
        question_ids = None
    else:
        question_ids = {
            _normalize_question_id(question_id)
            for question_id in (args.question_ids or [])
            if _normalize_question_id(question_id)
        }
        if not question_ids:
            raise ValueError("Provide --question-ids or enable --all-questions-in-dataset.")

    if not args.no_backup:
        backup_path = create_reasoning_bank_backup_bundle(
            target_path=args.main_bank.expanduser().resolve(),
            source_paths=[
                RUNS_DIR / "agent1_influence_standardized_attribution.jsonl",
                RUNS_DIR / "agent1_malicious_review_generation_dataset.jsonl",
                RUNS_DIR / "agent1_choice_influence_adversarial_enriched.json",
                RUNS_DIR / "agent1_choice_influence_non_adversarial_enriched.json",
            ],
            backup_dir=args.backup_dir,
            label="incremental_candidate",
        )
        if backup_path is not None:
            print(f"backup_bundle={backup_path}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    question_suffix = "allq" if question_ids is None else "-".join(sorted(question_ids))
    candidate_name = args.candidate_name.strip() or (
        f"{'_'.join(sorted(dataset_names))}_{question_suffix}_{timestamp}"
    )
    candidate_dir = args.work_root.expanduser().resolve() / candidate_name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    paths = _manifest_paths(candidate_dir)

    non_dataset, adv_dataset = _scan_selected_records(
        run_paths=run_paths,
        dataset_names=dataset_names,
        question_ids=question_ids,
    )
    _write_json(paths["non_raw"], non_dataset)
    _write_json(paths["adv_raw"], adv_dataset)

    _run_step(
        [
            sys.executable,
            "scripts/reasoning_bank_pipeline/enrich_agent1_choice_influence_dataset.py",
            "--non-adversarial-input",
            str(paths["non_raw"]),
            "--adversarial-input",
            str(paths["adv_raw"]),
            "--non-adversarial-output",
            str(paths["non_enriched"]),
            "--adversarial-output",
            str(paths["adv_enriched"]),
        ]
    )
    _run_step(
        [
            sys.executable,
            "scripts/reasoning_bank_pipeline/build_agent1_attribution_candidates.py",
            "--non-input",
            str(paths["non_enriched"]),
            "--adversarial-input",
            str(paths["adv_enriched"]),
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
            "--taxonomy",
            str(args.taxonomy),
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
            str(paths["adv_enriched"]),
            "--enriched-input",
            str(paths["non_enriched"]),
            "--output",
            str(paths["candidate_bank"]),
            "--corpus-id",
            f"incremental_{candidate_name}",
            "--no-backup",
        ]
    )

    manifest = {
        "status": "pending_review",
        "created_at": timestamp,
        "candidate_name": candidate_name,
        "candidate_dir": str(candidate_dir),
        "run_paths": [str(path) for path in run_paths],
        "dataset_names": sorted(dataset_names),
        "selection_mode": "all_questions_in_dataset" if question_ids is None else "question_id_subset",
        "question_ids": [] if question_ids is None else sorted(question_ids),
        "main_bank": str(args.main_bank.expanduser().resolve()),
        "artifact_paths": {key: str(path) for key, path in paths.items()},
        "record_counts": {
            "non_raw": non_dataset.get("record_count", 0),
            "adv_raw": adv_dataset.get("record_count", 0),
        },
        "merge_instructions": (
            f"After review, merge with: python scripts/reasoning_bank_pipeline/merge_incremental_reasoning_bank_candidate.py "
            f"--candidate-dir {candidate_dir} --approved"
        ),
    }
    _write_json(paths["manifest"], manifest)
    paths["review_notes"].write_text(
        "\n".join(
            [
                "# Review Required",
                "",
                "This candidate supplement has not been merged into the main reasoning bank.",
                "Review the generated artifacts in this folder, especially:",
                "- adversarial_reasoning_bank_v2_candidate.jsonl",
                "- agent1_influence_standardized_attribution.selected.jsonl",
                "- agent1_malicious_review_generation_dataset.selected.jsonl",
                "",
                "Only after manual confirmation should you run the merge command from candidate_manifest.json.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"candidate_dir={candidate_dir}")
    print(f"candidate_bank={paths['candidate_bank']}")
    print(f"manifest={paths['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
