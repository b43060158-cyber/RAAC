"""Stage 2: add lexical evidence that links review text to answer changes.

This step extracts overlap phrases and supporting clauses so later attribution
steps can reason over structured evidence instead of only raw text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
NON_ADVERSARIAL_INPUT = DEFAULT_RUNS_DIR / "agent1_choice_influence_non_adversarial.json"
ADVERSARIAL_INPUT = DEFAULT_RUNS_DIR / "agent1_choice_influence_adversarial.json"
NON_ADVERSARIAL_OUTPUT = DEFAULT_RUNS_DIR / "agent1_choice_influence_non_adversarial_enriched.json"
ADVERSARIAL_OUTPUT = DEFAULT_RUNS_DIR / "agent1_choice_influence_adversarial_enriched.json"

TOKEN_RE = re.compile(r"[A-Za-z0-9_/.+-]+")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[\.;:!?])\s+|\n+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "which",
    "with",
}
NEGATIVE_MARKERS = (
    "incorrect",
    "wrong",
    "ignores",
    "missed",
    "misread",
    "not ",
    "rather than",
    "instead of",
    "fails",
    "flawed",
    "unsupported",
)
REPLACEMENT_MARKERS = (
    "correct answer",
    "correct option",
    "should be",
    "better fits",
    "the answer is",
    "the correct answer is",
    "rather than",
    "instead",
    "not ",
    "both ",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _split_clauses(text: str) -> list[str]:
    parts = [part.strip() for part in CLAUSE_SPLIT_RE.split(text) if part.strip()]
    return parts if parts else ([_normalize_text(text)] if _normalize_text(text) else [])


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokenize(text) if len(token) > 2 and token not in STOPWORDS]


def _ngrams(tokens: list[str], size: int) -> list[str]:
    return [" ".join(tokens[index : index + size]) for index in range(0, len(tokens) - size + 1)]


def _find_overlap_phrases(review_text: str, new_reasoning: str, max_phrases: int = 8) -> list[str]:
    review_tokens = _tokenize(review_text)
    new_reasoning_lower = new_reasoning.lower()
    phrases: list[str] = []
    seen: set[str] = set()

    for size in (6, 5, 4, 3, 2, 1):
        for phrase in _ngrams(review_tokens, size):
            if phrase in seen:
                continue
            content_count = len([token for token in phrase.split() if len(token) > 2 and token not in STOPWORDS])
            if len(phrase) < 4 or content_count == 0:
                continue
            if phrase in new_reasoning_lower:
                seen.add(phrase)
                phrases.append(phrase)
                if len(phrases) >= max_phrases:
                    return phrases
    return phrases


def _find_supporting_clauses(
    *,
    source_text: str,
    required_substrings: list[str] | None = None,
    marker_substrings: tuple[str, ...] = (),
    overlap_text: str = "",
    max_clauses: int = 4,
) -> list[str]:
    clauses = _split_clauses(source_text)
    overlap_tokens = set(_content_tokens(overlap_text))
    results: list[str] = []

    for clause in clauses:
        clause_lower = clause.lower()
        if required_substrings and not any(required.lower() in clause_lower for required in required_substrings):
            continue
        marker_hit = any(marker in clause_lower for marker in marker_substrings)
        overlap_hit = bool(overlap_tokens and overlap_tokens.intersection(_content_tokens(clause)))
        if marker_hit or overlap_hit or not marker_substrings:
            if clause not in results:
                results.append(clause)
        if len(results) >= max_clauses:
            break
    return results


def _option_strings(record: dict[str, Any]) -> list[str]:
    option_ids = record.get("target_new_selected_option_ids", [])
    if not isinstance(option_ids, list):
        return []
    results: list[str] = []
    for option_id in option_ids:
        option_text = _normalize_text(option_id)
        if not option_text:
            continue
        results.append(option_text.lower())
        results.append(f"option {option_text.lower()}")
    return results


def _previous_option_strings(record: dict[str, Any]) -> list[str]:
    option_ids = record.get("target_previous_selected_option_ids", [])
    if not isinstance(option_ids, list):
        return []
    results: list[str] = []
    for option_id in option_ids:
        option_text = _normalize_text(option_id)
        if not option_text:
            continue
        results.append(option_text.lower())
        results.append(f"option {option_text.lower()}")
    return results


def _extract_previous_error_evidence(record: dict[str, Any]) -> list[str]:
    review_text = _normalize_text(record.get("agent1_review_text"))
    previous_reasoning = _normalize_text(record.get("target_previous_reasoning"))
    required = _previous_option_strings(record)
    evidence = _find_supporting_clauses(
        source_text=review_text,
        required_substrings=required if required else None,
        marker_substrings=NEGATIVE_MARKERS,
        overlap_text=previous_reasoning,
    )
    if evidence:
        return evidence
    return _find_supporting_clauses(
        source_text=review_text,
        marker_substrings=NEGATIVE_MARKERS,
        overlap_text=previous_reasoning,
    )


def _extract_replacement_evidence(record: dict[str, Any]) -> list[str]:
    review_text = _normalize_text(record.get("agent1_review_text"))
    new_reasoning = _normalize_text(record.get("target_new_reasoning"))
    required = _option_strings(record)
    evidence = _find_supporting_clauses(
        source_text=review_text,
        required_substrings=required if required else None,
        marker_substrings=REPLACEMENT_MARKERS,
        overlap_text=new_reasoning,
    )
    if evidence:
        return evidence
    return _find_supporting_clauses(
        source_text=review_text,
        marker_substrings=REPLACEMENT_MARKERS,
        overlap_text=new_reasoning,
    )


def _extract_change_summary_alignment(record: dict[str, Any]) -> list[str]:
    change_summary = _normalize_text(record.get("target_change_summary"))
    review_text = _normalize_text(record.get("agent1_review_text"))
    if not change_summary or not review_text:
        return []

    overlap_phrases = _find_overlap_phrases(review_text, change_summary, max_phrases=6)
    if overlap_phrases:
        return overlap_phrases

    review_tokens = set(_content_tokens(review_text))
    change_tokens = set(_content_tokens(change_summary))
    overlap_tokens = sorted(review_tokens.intersection(change_tokens))
    return overlap_tokens[:6]


def _build_alignment_notes(record: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    overlap = record.get("review_to_new_reasoning_overlap", [])
    if overlap:
        notes.append(
            "Review and new reasoning share phrases/tokens: " + ", ".join(overlap[:4])
        )
    previous_error = record.get("review_targets_previous_error_evidence", [])
    if previous_error:
        notes.append("Review explicitly attacks the prior answer or reasoning.")
    replacement = record.get("review_proposes_replacement_evidence", [])
    if replacement:
        notes.append("Review offers a replacement option or replacement justification.")
    change_alignment = record.get("change_summary_alignment_evidence", [])
    if change_alignment:
        notes.append("Change summary echoes language from the review.")
    return notes


def enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    review_text = _normalize_text(record.get("agent1_review_text"))
    new_reasoning = _normalize_text(record.get("target_new_reasoning"))
    enriched["review_to_new_reasoning_overlap"] = _find_overlap_phrases(review_text, new_reasoning)
    enriched["review_targets_previous_error_evidence"] = _extract_previous_error_evidence(record)
    enriched["review_proposes_replacement_evidence"] = _extract_replacement_evidence(record)
    enriched["change_summary_alignment_evidence"] = _extract_change_summary_alignment(record)
    enriched["alignment_notes"] = _build_alignment_notes(enriched)
    return enriched


def enrich_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Dataset payload must contain a records list.")
    enriched_records = [enrich_record(record) for record in records if isinstance(record, dict)]
    enriched_payload = dict(payload)
    enriched_payload["record_count"] = len(enriched_records)
    enriched_payload["records"] = enriched_records
    enriched_payload["enrichment_note"] = (
        "Alignment fields are heuristic evidence aids derived from the original review, "
        "new reasoning, and change summary. They are not gold labels."
    )
    return enriched_payload


def write_dataset(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich extracted agent_1 influence datasets with evidence-alignment fields."
    )
    parser.add_argument(
        "--non-adversarial-input",
        type=Path,
        default=NON_ADVERSARIAL_INPUT,
        help="Path to the non-adversarial extracted dataset JSON.",
    )
    parser.add_argument(
        "--adversarial-input",
        type=Path,
        default=ADVERSARIAL_INPUT,
        help="Path to the adversarial extracted dataset JSON.",
    )
    parser.add_argument(
        "--non-adversarial-output",
        type=Path,
        default=NON_ADVERSARIAL_OUTPUT,
        help="Output path for the enriched non-adversarial dataset JSON.",
    )
    parser.add_argument(
        "--adversarial-output",
        type=Path,
        default=ADVERSARIAL_OUTPUT,
        help="Output path for the enriched adversarial dataset JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    non_adversarial_dataset = enrich_dataset(_load_json(args.non_adversarial_input))
    adversarial_dataset = enrich_dataset(_load_json(args.adversarial_input))
    write_dataset(args.non_adversarial_output, non_adversarial_dataset)
    write_dataset(args.adversarial_output, adversarial_dataset)
    print(f"non_adversarial_output={args.non_adversarial_output}")
    print(f"non_adversarial_record_count={non_adversarial_dataset['record_count']}")
    print(f"adversarial_output={args.adversarial_output}")
    print(f"adversarial_record_count={adversarial_dataset['record_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
