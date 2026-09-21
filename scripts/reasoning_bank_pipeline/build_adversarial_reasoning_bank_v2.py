"""Stage 8: assemble the final reasoning-bank records.

This merges standardized attribution, malicious rewrites, and enriched evidence
into the JSONL bank consumed by runtime retrieval.
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

from scripts.reasoning_bank_pipeline.reasoning_bank_backup_utils import (
    create_reasoning_bank_backup_bundle,
)

RUNS_DIR = PROJECT_ROOT / "runs"
ATTRIBUTION_INPUT = RUNS_DIR / "agent1_influence_standardized_attribution.jsonl"
MALICIOUS_INPUT = RUNS_DIR / "agent1_malicious_review_generation_dataset.jsonl"
ADV_ENRICHED_INPUT = RUNS_DIR / "agent1_choice_influence_adversarial_enriched.json"
NON_ADV_ENRICHED_INPUT = RUNS_DIR / "agent1_choice_influence_non_adversarial_enriched.json"
OUTPUT = RUNS_DIR / "adversarial_reasoning_bank_v2.jsonl"
BACKUP_DIR = RUNS_DIR / "reasoning_bank_backups"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _load_enriched_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            if isinstance(record, dict) and record.get("sample_id"):
                records[str(record["sample_id"])] = record
    return records


def _first_text(value: object) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _copy_sensitive_spans(attribution_record: dict[str, Any]) -> list[str]:
    spans: list[str] = []
    for key in (
        "evidence_spans_from_review",
        "review_targets_previous_error_evidence",
        "review_proposes_replacement_evidence",
    ):
        for item in attribution_record.get(key, []):
            text = str(item).strip()
            if text and text not in spans:
                spans.append(text)
    return spans[:4]


def _borrowable_style(attribution_record: dict[str, Any]) -> str:
    if attribution_record.get("review_proposes_replacement_evidence"):
        return "replacement_path"
    if attribution_record.get("review_targets_previous_error_evidence"):
        return "direct_refutation"
    if attribution_record.get("review_to_new_reasoning_overlap"):
        return "alignment_transfer"
    return "credible_pressure"


def _compatible_modes(attribution_record: dict[str, Any], enriched_record: dict[str, Any] | None) -> list[str]:
    modes = {"convert"}
    overlap = attribution_record.get("review_to_new_reasoning_overlap") or (
        enriched_record.get("review_to_new_reasoning_overlap", []) if enriched_record else []
    )
    target_error = attribution_record.get("review_targets_previous_error_evidence") or (
        enriched_record.get("review_targets_previous_error_evidence", []) if enriched_record else []
    )
    replacement = attribution_record.get("review_proposes_replacement_evidence") or (
        enriched_record.get("review_proposes_replacement_evidence", []) if enriched_record else []
    )
    if overlap:
        modes.add("reinforce")
    if target_error:
        modes.add("probe")
    if replacement:
        modes.add("pressure_holdout")
    return sorted(modes)


def _target_profile(attribution_record: dict[str, Any], compatible_modes: list[str]) -> str:
    previous = attribution_record.get("target_previous_selected_option_ids", [])
    new = attribution_record.get("target_new_selected_option_ids", [])
    if "reinforce" in compatible_modes and previous == new:
        return "inside_cluster"
    if attribution_record.get("run_type") == "adversarial":
        return "wrong_outsider"
    return "correct_outsider"


def _retrieval_text(
    attribution_record: dict[str, Any],
    malicious_record: dict[str, Any],
    borrowable_attack_point: str,
    borrowable_replacement_path: str,
    borrowable_style: str,
) -> str:
    desired_target_shift = malicious_record.get("input", {}).get("desired_target_shift", {})
    parts = [
        str(attribution_record.get("question_text", "")),
        " ".join(
            f"{option.get('option_id', '')}: {option.get('text', '')}"
            for option in attribution_record.get("options", [])
            if isinstance(option, dict)
        ),
        " ".join(attribution_record.get("target_previous_selected_option_ids", [])),
        str(attribution_record.get("target_previous_reasoning", "")),
        json.dumps(desired_target_shift, ensure_ascii=False),
        borrowable_attack_point,
        borrowable_replacement_path,
        borrowable_style,
        str(malicious_record.get("malicious_strategy_notes", "")),
        str(malicious_record.get("malicious_adapted_review_rationale", "")),
        str(attribution_record.get("primary_factor_id", "")),
        " ".join(attribution_record.get("secondary_factor_ids", [])),
    ]
    return "\n".join(part for part in parts if str(part).strip())


def _question_key_from_sample_id(sample_id: object) -> str:
    parts = str(sample_id or "").split("::")
    if len(parts) >= 2:
        return parts[1].strip()
    return ""


def build_reasoning_bank_v2(
    attribution_records: list[dict[str, Any]],
    malicious_records: list[dict[str, Any]],
    enriched_records: dict[str, dict[str, Any]],
    *,
    corpus_id: str = "bank_v2",
) -> list[dict[str, Any]]:
    malicious_by_id = {
        str(record.get("sample_id", "")): record
        for record in malicious_records
        if record.get("sample_id")
    }
    bank: list[dict[str, Any]] = []
    for attribution_record in attribution_records:
        sample_id = str(attribution_record.get("sample_id", ""))
        malicious_record = malicious_by_id.get(sample_id)
        if malicious_record is None:
            continue
        enriched_record = enriched_records.get(sample_id)
        desired_target_shift = (
            malicious_record.get("input", {}).get("desired_target_shift", {})
            if isinstance(malicious_record.get("input", {}), dict)
            else {}
        )
        borrowable_attack_point = _first_text(
            attribution_record.get("review_targets_previous_error_evidence")
            or (enriched_record.get("review_targets_previous_error_evidence", []) if enriched_record else [])
            or attribution_record.get("supporting_evidence")
        )
        borrowable_replacement_path = _first_text(
            attribution_record.get("review_proposes_replacement_evidence")
            or (enriched_record.get("review_proposes_replacement_evidence", []) if enriched_record else [])
            or attribution_record.get("change_summary_alignment_evidence")
        )
        borrowable_style = _borrowable_style(attribution_record)
        compatible_modes = _compatible_modes(attribution_record, enriched_record)
        bank.append(
            {
                "corpus_id": corpus_id,
                "source_bundle": "standardized_attribution+malicious_generation+enriched",
                "sample_id": sample_id,
                "question_key": _question_key_from_sample_id(sample_id),
                "question_id": attribution_record.get("question_id", ""),
                "dataset_name": attribution_record.get("dataset_name", ""),
                "run_type": attribution_record.get("run_type", ""),
                "target_agent_id": attribution_record.get("target_agent_id", ""),
                "question_text": attribution_record.get("question_text", ""),
                "options": attribution_record.get("options", []),
                "target_previous_selected_option_ids": attribution_record.get(
                    "target_previous_selected_option_ids", []
                ),
                "target_previous_reasoning": attribution_record.get(
                    "target_previous_reasoning", ""
                ),
                "desired_target_shift": desired_target_shift,
                "primary_factor_id": attribution_record.get("primary_factor_id", ""),
                "secondary_factor_ids": attribution_record.get("secondary_factor_ids", []),
                "supporting_evidence": attribution_record.get("supporting_evidence", []),
                "review_to_new_reasoning_overlap": attribution_record.get(
                    "review_to_new_reasoning_overlap",
                    [],
                ),
                "review_targets_previous_error_evidence": attribution_record.get(
                    "review_targets_previous_error_evidence", []
                ),
                "review_proposes_replacement_evidence": attribution_record.get(
                    "review_proposes_replacement_evidence", []
                ),
                "change_summary_alignment_evidence": attribution_record.get(
                    "change_summary_alignment_evidence", []
                ),
                "neutralized_review_rationale": malicious_record.get(
                    "neutralized_review_rationale", ""
                ),
                "malicious_adapted_review_rationale": malicious_record.get(
                    "malicious_adapted_review_rationale", ""
                ),
                "malicious_strategy_notes": malicious_record.get(
                    "malicious_strategy_notes", ""
                ),
                "factor_mapping_confidence": attribution_record.get(
                    "factor_mapping_confidence", 0.0
                ),
                "compatible_modes": compatible_modes,
                "target_profile": _target_profile(attribution_record, compatible_modes),
                "borrowable_attack_point": borrowable_attack_point,
                "borrowable_replacement_path": borrowable_replacement_path,
                "borrowable_style": borrowable_style,
                "copy_sensitive_span": _copy_sensitive_spans(attribution_record),
                "borrowing_notes": (
                    "Runtime borrowing sample for adversarial review generation. "
                    "Prefer structure and attack path; avoid verbatim reuse of copy_sensitive_span."
                ),
                "retrieval_text": _retrieval_text(
                    attribution_record,
                    malicious_record,
                    borrowable_attack_point,
                    borrowable_replacement_path,
                    borrowable_style,
                ),
            }
        )
    return bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the adversarial reasoning bank v2 from attribution and malicious generation artifacts."
    )
    parser.add_argument("--attribution-input", type=Path, default=ATTRIBUTION_INPUT)
    parser.add_argument("--malicious-input", type=Path, default=MALICIOUS_INPUT)
    parser.add_argument(
        "--enriched-input",
        type=Path,
        action="append",
        dest="enriched_inputs",
        help=(
            "Enriched JSON input path. Repeat to provide multiple files. "
            "Defaults to the standard adversarial and non-adversarial enriched inputs."
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--corpus-id",
        type=str,
        default="bank_v2",
        help="Corpus identifier written into every reasoning-bank record.",
    )
    parser.add_argument("--backup-dir", type=Path, default=BACKUP_DIR)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the automatic pre-rebuild archive bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enriched_inputs = args.enriched_inputs or [ADV_ENRICHED_INPUT, NON_ADV_ENRICHED_INPUT]
    if not args.no_backup:
        backup_path = create_reasoning_bank_backup_bundle(
            target_path=args.output,
            source_paths=[
                args.attribution_input,
                args.malicious_input,
                *enriched_inputs,
            ],
            backup_dir=args.backup_dir,
            label="rebuild",
        )
        if backup_path is not None:
            print(f"Backup bundle: {backup_path}")
    bank = build_reasoning_bank_v2(
        _load_jsonl(args.attribution_input),
        _load_jsonl(args.malicious_input),
        _load_enriched_records(enriched_inputs),
        corpus_id=args.corpus_id,
    )
    with args.output.open("w", encoding="utf-8") as handle:
        for record in bank:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(bank)} records to {args.output}")


if __name__ == "__main__":
    main()
