"""Built-in retrieval selectors.

``topk_sorted_v1`` is a byte-for-byte port of the historical inline sort/top-k
in ``ReasoningBank.retrieve_examples`` and is the default.
"""

from __future__ import annotations

from typing import Any

from .base import RetrievalContext, ScoredRecord
from .features import overlap_score
from .registry import register_selector


def _ordered(scored: list[ScoredRecord]) -> list[ScoredRecord]:
    return sorted(
        scored,
        key=lambda item: (
            -item.score,
            item.record.get("corpus_id", ""),
            item.record.get("sample_id", ""),
        ),
    )


@register_selector("topk_sorted_v1")
def topk_sorted_v1(
    scored: list[ScoredRecord],
    ctx: RetrievalContext,
    top_k: int,
) -> list[dict[str, Any]]:
    return [item.record for item in _ordered(scored)[:top_k]]


@register_selector("threshold_topk_v1")
def threshold_topk_v1(
    scored: list[ScoredRecord],
    ctx: RetrievalContext,
    top_k: int,
) -> list[dict[str, Any]]:
    """Top-k above a *soft* relevance floor (suggestion #6).

    The floor is a preference, never a hard gate: if nothing clears
    ``min_score`` the single best candidate is still returned, honouring the
    no-empty-generation contract.
    """
    min_score = float(ctx.params.get("min_score", 0.0))
    ordered = _ordered(scored)
    above = [item for item in ordered if item.score >= min_score]
    chosen = above[:top_k] if above else ordered[:1]
    return [item.record for item in chosen]


def _diversity_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Redundancy between two candidates by rhetorical mechanism + attack point."""
    left_style = str(left.get("borrowable_style", "")).strip()
    right_style = str(right.get("borrowable_style", "")).strip()
    style_sim = 1.0 if left_style and left_style == right_style else 0.0
    point_sim = overlap_score(
        str(left.get("borrowable_attack_point", "")),
        str(right.get("borrowable_attack_point", "")),
    )
    return max(style_sim, point_sim)


@register_selector("mmr_diversity_v1")
def mmr_diversity_v1(
    scored: list[ScoredRecord],
    ctx: RetrievalContext,
    top_k: int,
) -> list[dict[str, Any]]:
    """Maximal-marginal-relevance top-k for a varied borrowing toolkit (#5).

    Greedily picks the most relevant candidate, then trades relevance against
    redundancy (same ``borrowable_style`` / overlapping attack point) for the
    rest, so the generator gets complementary angles rather than near-dupes.
    """
    if not scored:
        return []
    mmr_lambda = float(ctx.params.get("mmr_lambda", 0.7))
    remaining = list(_ordered(scored))
    selected: list[ScoredRecord] = [remaining.pop(0)]
    while remaining and len(selected) < top_k:
        best_index = 0
        best_value = float("-inf")
        for index, candidate in enumerate(remaining):
            max_sim = max(
                _diversity_similarity(candidate.record, chosen.record)
                for chosen in selected
            )
            value = mmr_lambda * candidate.score - (1.0 - mmr_lambda) * max_sim
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(remaining.pop(best_index))
    return [item.record for item in selected]
