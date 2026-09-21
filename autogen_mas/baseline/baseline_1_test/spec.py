from __future__ import annotations

from autogen_mas.baseline.baseline_1.prompts import build_review_prompt
from autogen_mas.baseline.baseline_1.spec import (
    BASELINE_1_ANSWER_SYSTEM_PROMPT,
    BASELINE_1_REVIEW_SYSTEM_PROMPT,
)
from autogen_mas.baseline.types import ChoiceBaselineSpec

from .prompts import build_answer_prompt, build_review_candidates_prompt


BASELINE_1_TEST_STRATEGY_NAME = "ablation_withrag"
BASELINE_1_TEST_ALIASES = (
    "ablation_withrag",
    "ablation-withrag",
    "baseline_1_test",
    "baseline-1-test",
)

# Identical persona to baseline-1: the only difference is that the review
# rationale is drawn from the reasoning-bank corpus (3 candidates, random pick).
BASELINE_1_TEST_ANSWER_SYSTEM_PROMPT = BASELINE_1_ANSWER_SYSTEM_PROMPT
BASELINE_1_TEST_REVIEW_SYSTEM_PROMPT = BASELINE_1_REVIEW_SYSTEM_PROMPT

BASELINE_1_TEST_SPEC = ChoiceBaselineSpec(
    strategy_name=BASELINE_1_TEST_STRATEGY_NAME,
    aliases=BASELINE_1_TEST_ALIASES,
    default_answer_system_prompt=BASELINE_1_TEST_ANSWER_SYSTEM_PROMPT,
    default_review_system_prompt=BASELINE_1_TEST_REVIEW_SYSTEM_PROMPT,
    reasoning_bank_enabled=False,
    review_candidate_reranker_enabled=False,
    use_provider_default_temperature=True,
    build_answer_prompt=build_answer_prompt,
    build_review_prompt=build_review_prompt,
    reasoning_bank_review_enabled=True,
    reasoning_bank_review_candidate_count=3,
)

__all__ = [
    "BASELINE_1_TEST_ALIASES",
    "BASELINE_1_TEST_ANSWER_SYSTEM_PROMPT",
    "BASELINE_1_TEST_REVIEW_SYSTEM_PROMPT",
    "BASELINE_1_TEST_SPEC",
    "BASELINE_1_TEST_STRATEGY_NAME",
    "build_review_candidates_prompt",
]
