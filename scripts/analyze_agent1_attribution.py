from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent1_influence_pipeline_utils import load_json, read_jsonl, write_json


RUNS_DIR = PROJECT_ROOT / "runs"
TAXONOMY_PATH = RUNS_DIR / "agent1_influence_factor_taxonomy.json"
ATTRIBUTION_PATH = RUNS_DIR / "agent1_influence_standardized_attribution.jsonl"
JSON_OUTPUT = RUNS_DIR / "agent1_attribution_analysis.json"
MARKDOWN_OUTPUT = RUNS_DIR / "agent1_attribution_analysis.md"


def _factor_meta(taxonomy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for factor in taxonomy.get("factors", []):
        if isinstance(factor, dict):
            factor_id = str(factor.get("factor_id", "")).strip()
            if factor_id:
                meta[factor_id] = factor
    return meta


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_samples(
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda item: (
            -(_safe_float(item.get("factor_mapping_confidence")) or 0.0),
            str(item.get("sample_id", "")),
        ),
    )
    result: list[dict[str, Any]] = []
    for record in ordered[:limit]:
        result.append(
            {
                "sample_id": record.get("sample_id"),
                "run_type": record.get("run_type"),
                "dataset_name": record.get("dataset_name"),
                "question_key": record.get("question_key"),
                "factor_mapping_confidence": record.get("factor_mapping_confidence"),
                "supporting_evidence": record.get("supporting_evidence", []),
                "mapping_notes": record.get("mapping_notes", ""),
                "agent1_review_text": record.get("agent1_review_text", ""),
            }
        )
    return result


def build_analysis(
    records: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    *,
    representative_limit: int,
) -> dict[str, Any]:
    factor_meta = _factor_meta(taxonomy)
    primary_counter: Counter[str] = Counter()
    secondary_counter: Counter[str] = Counter()
    by_run_type: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_dataset: defaultdict[str, Counter[str]] = defaultdict(Counter)
    factor_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    confidences_by_factor: defaultdict[str, list[float]] = defaultdict(list)

    for record in records:
        primary_factor_id = str(record.get("primary_factor_id", "")).strip()
        if not primary_factor_id:
            continue
        primary_counter[primary_factor_id] += 1
        by_run_type[str(record.get("run_type", ""))][primary_factor_id] += 1
        by_dataset[str(record.get("dataset_name", ""))][primary_factor_id] += 1
        factor_records[primary_factor_id].append(record)

        confidence = _safe_float(record.get("factor_mapping_confidence"))
        if confidence is not None:
            confidences_by_factor[primary_factor_id].append(confidence)

        for secondary_factor_id in record.get("secondary_factor_ids", []):
            secondary_counter[str(secondary_factor_id)] += 1

    factor_details: list[dict[str, Any]] = []
    for factor_id, count in primary_counter.most_common():
        meta = factor_meta.get(factor_id, {})
        confidences = confidences_by_factor.get(factor_id, [])
        factor_details.append(
            {
                "factor_id": factor_id,
                "factor_name": meta.get("factor_name", ""),
                "definition": meta.get("definition", ""),
                "primary_count": count,
                "average_factor_mapping_confidence": (
                    round(statistics.mean(confidences), 4) if confidences else None
                ),
                "run_type_distribution": dict(by_run_type["adversarial"] | Counter()) if False else None,
                "dataset_distribution": dict(),
                "representative_samples": _top_samples(
                    factor_records[factor_id],
                    limit=representative_limit,
                ),
            }
        )

    for item in factor_details:
        factor_id = item["factor_id"]
        item["run_type_distribution"] = {
            run_type: counter[factor_id]
            for run_type, counter in by_run_type.items()
            if counter[factor_id] > 0
        }
        item["dataset_distribution"] = {
            dataset_name: counter[factor_id]
            for dataset_name, counter in by_dataset.items()
            if counter[factor_id] > 0
        }

    return {
        "source_record_count": len(records),
        "factor_taxonomy_version": taxonomy.get("taxonomy_version", ""),
        "primary_factor_frequency": dict(primary_counter.most_common()),
        "secondary_factor_frequency": dict(secondary_counter.most_common()),
        "primary_factor_by_run_type": {
            run_type: dict(counter.most_common())
            for run_type, counter in sorted(by_run_type.items())
        },
        "primary_factor_by_dataset": {
            dataset_name: dict(counter.most_common())
            for dataset_name, counter in sorted(by_dataset.items())
        },
        "factor_details": factor_details,
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Agent1 Attribution Analysis")
    lines.append("")
    lines.append(f"- Source record count: {analysis['source_record_count']}")
    lines.append(f"- Taxonomy version: {analysis['factor_taxonomy_version']}")
    lines.append("")

    lines.append("## Primary Factor Frequency")
    lines.append("")
    for factor_id, count in analysis["primary_factor_frequency"].items():
        lines.append(f"- `{factor_id}`: {count}")
    lines.append("")

    lines.append("## Primary Factor By Run Type")
    lines.append("")
    for run_type, counter in analysis["primary_factor_by_run_type"].items():
        lines.append(f"### {run_type}")
        for factor_id, count in counter.items():
            lines.append(f"- `{factor_id}`: {count}")
        lines.append("")

    lines.append("## Primary Factor By Dataset")
    lines.append("")
    for dataset_name, counter in analysis["primary_factor_by_dataset"].items():
        lines.append(f"### {dataset_name}")
        for factor_id, count in counter.items():
            lines.append(f"- `{factor_id}`: {count}")
        lines.append("")

    lines.append("## Factor Details")
    lines.append("")
    for factor in analysis["factor_details"]:
        lines.append(f"### {factor['factor_id']} - {factor['factor_name']}")
        lines.append(f"- Primary count: {factor['primary_count']}")
        lines.append(f"- Average mapping confidence: {factor['average_factor_mapping_confidence']}")
        if factor["definition"]:
            lines.append(f"- Definition: {factor['definition']}")
        if factor["run_type_distribution"]:
            lines.append(f"- Run type distribution: {factor['run_type_distribution']}")
        if factor["dataset_distribution"]:
            lines.append(f"- Dataset distribution: {factor['dataset_distribution']}")
        lines.append("- Representative samples:")
        for sample in factor["representative_samples"]:
            lines.append(
                f"  - `{sample['sample_id']}` | {sample['run_type']} | {sample['dataset_name']} | "
                f"confidence={sample['factor_mapping_confidence']}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze standardized agent1 attribution results.")
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY_PATH)
    parser.add_argument("--input", type=Path, default=ATTRIBUTION_PATH)
    parser.add_argument("--json-output", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=MARKDOWN_OUTPUT)
    parser.add_argument("--representative-limit", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    taxonomy = load_json(args.taxonomy)
    records = read_jsonl(args.input)
    analysis = build_analysis(records, taxonomy, representative_limit=args.representative_limit)
    write_json(args.json_output, analysis)
    args.markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(f"json_output={args.json_output}")
    print(f"markdown_output={args.markdown_output}")
    print(f"factor_count={len(analysis['factor_details'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
