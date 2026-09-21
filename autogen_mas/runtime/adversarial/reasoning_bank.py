from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...models import QuestionRecord, ShortAnswerQuestionRecord, TaskRecord
from .retrieval import (
    RetrievalContext,
    RetrievalError,
    ScoredRecord,
    get_scorer,
    get_selector,
)
from .retrieval.features import (
    normalize_authority_flag as _normalize_authority_flag,
    normalize_authority_risk as _normalize_authority_risk,
)

if TYPE_CHECKING:
    from ...config import AdversarialAgentBehaviorConfig

logger = logging.getLogger(__name__)

# Datasets that are scored/stored in the bank under a different (canonical)
# name than the one used at run time. The CIAR single-choice variant
# (``ciar_choice``) reuses the original CIAR corpus, which was built when CIAR
# was served as single_choice and is stored under ``ciar``.
_BANK_DATASET_ALIASES = {
    "ciar_choice": "ciar",
}


def canonical_bank_dataset_name(dataset_name: str | None) -> str | None:
    """Map a run-time dataset name to the name it is stored under in the bank."""
    if dataset_name is None:
        return None
    return _BANK_DATASET_ALIASES.get(dataset_name, dataset_name)


def model_family(model_id: str | None) -> str:
    """Coarse model-family bucket used to route the reasoning bank at runtime."""
    normalized = (model_id or "").strip().casefold()
    if not normalized:
        return ""
    if "qwen" in normalized:
        return "qwen"
    if "deepseek" in normalized:
        return "deepseek"
    if "gemini" in normalized:
        return "gemini"
    if normalized.startswith("gpt") or "openai" in normalized:
        return "openai"
    if "claude" in normalized or "anthropic" in normalized:
        return "claude"
    return "other"


def _family_match(
    record: dict[str, Any],
    *,
    family_scope: str,
    runtime_family: str,
) -> bool:
    """Whether a record belongs to the requested family scope.

    ``same`` keeps records tagged with the runtime family; ``other`` keeps
    records tagged with a *different, non-empty* family (used for cross-family
    fallback). Untagged records are never matched, so a misconfigured routing
    table surfaces as an explicit ``RetrievalError`` rather than silent reuse.
    """
    record_family = str(record.get("build_model_family", "") or "").strip()
    if family_scope == "same":
        return bool(record_family) and record_family == runtime_family
    return bool(record_family) and record_family != runtime_family


def _preferred_reference_rationale(record: dict[str, Any]) -> str:
    neutralized = str(record.get("neutralized_review_rationale", "")).strip()
    malicious = str(record.get("malicious_adapted_review_rationale", "")).strip()
    if _normalize_authority_flag(record.get("authority_packaging_flag")) and neutralized:
        return neutralized
    return malicious or neutralized


class ReasoningBank:
    """Strategy-agnostic retrieval helper for adversarial review generation."""

    def __init__(
        self,
        *,
        behavior_config: "AdversarialAgentBehaviorConfig",
        runtime_model: str | None = None,
    ) -> None:
        self.behavior_config = behavior_config
        self.runtime_model = runtime_model
        self._cached_paths: tuple[Path, ...] = ()
        self._cached_records: list[dict[str, Any]] | None = None

    def enabled_for(self, *, strategy_name: str, phase: str) -> bool:
        return self.behavior_config.reasoning_bank_enabled_for(
            strategy_name=strategy_name,
            phase=phase,
        )

    def retrieve_examples(
        self,
        *,
        question_record: TaskRecord,
        target_agent_id: str,
        target_current_option_ids: list[str],
        target_current_final_answer: str = "",
        target_current_reasoning: str,
        desired_target_shift: list[str],
        desired_target_final_answer: str = "",
        review_phase_mode: str,
        strategy_name: str | None = None,
    ) -> list[dict[str, Any]]:
        records = self._load_records()
        if not records:
            if self._family_routing_active():
                raise RetrievalError(
                    "Reasoning-bank family routing is configured but no corpus records "
                    "are available; refusing corpus-free rationale generation."
                )
            return []

        ctx = RetrievalContext(
            question_record=question_record,
            dataset_name=canonical_bank_dataset_name(question_record.dataset_name),
            target_agent_id=target_agent_id,
            target_current_option_ids=list(target_current_option_ids),
            target_current_reasoning=target_current_reasoning,
            target_current_final_answer=target_current_final_answer,
            desired_target_shift=list(desired_target_shift),
            desired_target_final_answer=desired_target_final_answer,
            review_phase_mode=review_phase_mode,
        )

        require_nonempty = self._family_routing_active()
        if require_nonempty:
            candidates = self._resolve_candidate_pool(records, ctx)
        else:
            candidates = self._prefilter(records, ctx, allow_cross_dataset=False)
            if not candidates:
                return []

        scorer_id, selector_id, params = self.behavior_config.resolve_retrieval_method(
            strategy_name=strategy_name or ""
        )
        ctx = replace(ctx, params=params)
        scorer = get_scorer(scorer_id)
        selector = get_selector(selector_id)
        top_k = int(params.get("top_k", self.behavior_config.reasoning_bank_top_k) or 1)

        scored = scorer(candidates, ctx)
        if require_nonempty and not scored:
            # The configured scorer rejected every candidate (e.g. all
            # non-positive). The no-empty-generation contract forbids dropping
            # to zero, so fall back to neutral scores and let the selector pick
            # deterministically.
            scored = [ScoredRecord(score=0.0, record=record) for record in candidates]

        selected = selector(scored, ctx, top_k)
        if require_nonempty and not selected:
            raise RetrievalError(
                "Reasoning-bank retrieval produced an empty selection under family "
                "routing; refusing corpus-free rationale generation."
            )
        return selected

    def _prefilter(
        self,
        records: list[dict[str, Any]],
        ctx: RetrievalContext,
        *,
        allow_cross_dataset: bool,
    ) -> list[dict[str, Any]]:
        allowed_datasets = set(self.behavior_config.reasoning_bank_datasets)
        allowed_corpora = set(self.behavior_config.reasoning_bank_corpora)
        out: list[dict[str, Any]] = []
        for record in records:
            if allowed_datasets and record.get("dataset_name") not in allowed_datasets:
                continue
            if allowed_corpora and record.get("corpus_id") not in allowed_corpora:
                continue
            if not allow_cross_dataset and record.get("dataset_name") != ctx.dataset_name:
                continue
            compatible_modes = set(record.get("compatible_modes", []))
            if compatible_modes and ctx.review_phase_mode not in compatible_modes:
                continue
            if record.get("target_agent_id") == ctx.target_agent_id:
                continue
            if not str(record.get("retrieval_text", "")).strip():
                continue
            out.append(record)
        return out

    def _resolve_candidate_pool(
        self,
        records: list[dict[str, Any]],
        ctx: RetrievalContext,
    ) -> list[dict[str, Any]]:
        """Escalate through permitted pools, guaranteeing a non-empty result.

        Preference order (first non-empty wins):
          1. same model family, same dataset
          2. other model family, same dataset      (cross-family fallback)
          3. same model family, any dataset         (only if cross-dataset on)
          4. other model family, any dataset        (only if both switches on)
        """
        family = model_family(self.runtime_model)
        allow_fallback = self.behavior_config.reasoning_bank_allow_cross_family_fallback
        allow_cross_dataset = self.behavior_config.reasoning_bank_allow_cross_dataset

        attempts: list[tuple[str, bool]] = [("same", False)]
        if allow_fallback:
            attempts.append(("other", False))
        if allow_cross_dataset:
            attempts.append(("same", True))
            if allow_fallback:
                attempts.append(("other", True))

        for family_scope, relax_dataset in attempts:
            subset = [
                record
                for record in records
                if _family_match(record, family_scope=family_scope, runtime_family=family)
            ]
            pool = self._prefilter(subset, ctx, allow_cross_dataset=relax_dataset)
            if pool:
                return pool

        raise RetrievalError(
            "No reasoning-bank candidates available for runtime model family "
            f"{family!r} (cross_family_fallback="
            f"{allow_fallback}, cross_dataset={allow_cross_dataset}); refusing "
            "corpus-free rationale generation."
        )

    def build_borrowing_context(
        self,
        *,
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        for example in examples:
            context.append(
                {
                    "sample_id": example.get("sample_id", ""),
                    "corpus_id": example.get("corpus_id", ""),
                    "compatible_modes": example.get("compatible_modes", []),
                    "target_profile": example.get("target_profile", ""),
                    "desired_target_shift": example.get("desired_target_shift", {}),
                    "primary_factor_id": example.get("primary_factor_id", ""),
                    "secondary_factor_ids": example.get("secondary_factor_ids", []),
                    "borrowable_attack_point": example.get("borrowable_attack_point", ""),
                    "borrowable_replacement_path": example.get("borrowable_replacement_path", ""),
                    "borrowable_style": example.get("borrowable_style", ""),
                    "supporting_evidence": example.get("supporting_evidence", [])[:3],
                    "review_targets_previous_error_evidence": example.get(
                        "review_targets_previous_error_evidence", []
                    )[:2],
                    "review_proposes_replacement_evidence": example.get(
                        "review_proposes_replacement_evidence", []
                    )[:2],
                    "malicious_strategy_notes": example.get("malicious_strategy_notes", ""),
                    "reference_rationale": _preferred_reference_rationale(example),
                    "copy_sensitive_span": example.get("copy_sensitive_span", []),
                    "borrowing_notes": example.get("borrowing_notes", ""),
                    "authority_packaging_flag": _normalize_authority_flag(
                        example.get("authority_packaging_flag")
                    ),
                    "authority_packaging_risk": _normalize_authority_risk(
                        example.get("authority_packaging_risk")
                    ),
                    "authority_packaging_reason": str(
                        example.get("authority_packaging_reason", "")
                    ).strip(),
                }
            )
        return context

    def _family_routing_active(self) -> bool:
        return bool(self.behavior_config.reasoning_bank_paths_by_model_family)

    def _path_family_map(self) -> dict[str, str]:
        """Map each configured bank path to the model family it serves."""
        mapping: dict[str, str] = {}
        for family, paths in self.behavior_config.reasoning_bank_paths_by_model_family.items():
            for path in paths:
                cleaned = str(path).strip()
                if cleaned:
                    mapping[cleaned] = family
        return mapping

    def _configured_paths(self) -> tuple[Path, ...]:
        if self._family_routing_active():
            raw_paths = list(self._path_family_map())
        else:
            raw_paths = self.behavior_config.reasoning_bank_paths or [
                self.behavior_config.reasoning_bank_path
            ]
        return tuple(Path(path) for path in raw_paths if str(path).strip())

    def _load_records(self) -> list[dict[str, Any]]:
        paths = self._configured_paths()
        if self._cached_records is not None and self._cached_paths == paths:
            return self._cached_records
        path_family = self._path_family_map()
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for path in paths:
            if not path.exists():
                continue
            path_str = str(path)
            family = path_family.get(path_str)
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        continue
                    payload.setdefault("corpus_id", path.stem)
                    payload.setdefault("source_path", path_str)
                    # File-level family routing wins when configured; otherwise
                    # keep any family already recorded by the build pipeline.
                    if family is not None:
                        payload.setdefault("build_model_family", family)
                    key = (str(payload.get("corpus_id", "")), str(payload.get("sample_id", "")))
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(payload)
        self._cached_paths = paths
        self._cached_records = records
        return records


def _question_key_from_sample_id(sample_id: object) -> str | None:
    parts = str(sample_id or "").split("::")
    if len(parts) < 2:
        return None
    question_key = parts[1].strip()
    return question_key or None


def reasoning_bank_question_keys(
    *,
    behavior_config: "AdversarialAgentBehaviorConfig",
    dataset_name: str | None = None,
) -> set[str]:
    bank = ReasoningBank(behavior_config=behavior_config)
    question_keys: set[str] = set()
    allowed_corpora = set(behavior_config.reasoning_bank_corpora)
    for record in bank._load_records():
        if dataset_name and record.get("dataset_name") != dataset_name:
            continue
        corpus_id = str(record.get("corpus_id", "")).strip()
        if allowed_corpora and corpus_id not in allowed_corpora:
            continue
        question_key = str(record.get("question_key", "")).strip()
        if not question_key:
            question_key = (
                _question_key_from_sample_id(record.get("sample_id"))
                or str(record.get("source_question_key", "")).strip()
            )
        if question_key:
            question_keys.add(question_key)
    return question_keys
