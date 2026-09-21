from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
ATTRIBUTION_INPUT = RUNS_DIR / "agent1_influence_standardized_attribution.jsonl"
MALICIOUS_INPUT = RUNS_DIR / "agent1_malicious_review_generation_dataset.jsonl"
OUTPUT = RUNS_DIR / "adversarial_reasoning_bank.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _compatible_modes(attribution_record: dict) -> list[str]:
    modes = {"convert"}
    if attribution_record.get("review_to_new_reasoning_overlap"):
        modes.add("reinforce")
    if attribution_record.get("review_targets_previous_error_evidence"):
        modes.add("probe")
    if attribution_record.get("review_proposes_replacement_evidence"):
        modes.add("pressure_holdout")
    return sorted(modes)


def _retrieval_text(attribution_record: dict, malicious_record: dict) -> str:
    parts = [
        str(attribution_record.get("question_text", "")),
        " ".join(
            f"{option.get('option_id', '')}: {option.get('text', '')}"
            for option in attribution_record.get("options", [])
            if isinstance(option, dict)
        ),
        " ".join(attribution_record.get("target_previous_selected_option_ids", [])),
        str(attribution_record.get("target_previous_reasoning", "")),
        str(malicious_record.get("malicious_strategy_notes", "")),
        str(malicious_record.get("malicious_adapted_review_rationale", "")),
        " ".join(attribution_record.get("primary_factor_id", "").split()),
        " ".join(attribution_record.get("secondary_factor_ids", [])),
    ]
    return "\n".join(part for part in parts if part.strip())


def build_reasoning_bank(
    attribution_records: list[dict],
    malicious_records: list[dict],
) -> list[dict]:
    malicious_by_id = {
        str(record.get("sample_id", "")): record
        for record in malicious_records
        if record.get("sample_id")
    }
    bank: list[dict] = []
    for attribution_record in attribution_records:
        sample_id = str(attribution_record.get("sample_id", ""))
        malicious_record = malicious_by_id.get(sample_id)
        if malicious_record is None:
            continue
        desired_target_shift = (
            malicious_record.get("input", {}).get("desired_target_shift", {})
            if isinstance(malicious_record.get("input", {}), dict)
            else {}
        )
        bank.append(
            {
                "sample_id": sample_id,
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
                    "review_to_new_reasoning_overlap", []
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
                "compatible_modes": _compatible_modes(attribution_record),
                "borrowing_notes": (
                    "Built from successful option-change persuasion samples. "
                    "convert is direct; reinforce/probe/pressure_holdout are approximate compatible modes."
                ),
                "retrieval_text": _retrieval_text(attribution_record, malicious_record),
            }
        )
    return bank


def main() -> None:
    bank = build_reasoning_bank(
        _load_jsonl(ATTRIBUTION_INPUT),
        _load_jsonl(MALICIOUS_INPUT),
    )
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in bank:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(bank)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
