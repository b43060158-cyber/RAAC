from __future__ import annotations

from autogen_mas.baseline.types import ChoiceBaselineSpec

from .baseline_1 import BASELINE_1_SPEC
from .baseline_1_test import BASELINE_1_TEST_SPEC
from .baseline_1_test_2 import BASELINE_1_TEST_2_SPEC
from .baseline_optim import BASELINE_OPTIM_SPEC


_CHOICE_BASELINE_SPECS: dict[str, ChoiceBaselineSpec] = {
    BASELINE_1_SPEC.strategy_name: BASELINE_1_SPEC,
    BASELINE_1_TEST_SPEC.strategy_name: BASELINE_1_TEST_SPEC,
    BASELINE_1_TEST_2_SPEC.strategy_name: BASELINE_1_TEST_2_SPEC,
    BASELINE_OPTIM_SPEC.strategy_name: BASELINE_OPTIM_SPEC,
}
_CHOICE_BASELINE_ALIAS_TO_NAME: dict[str, str] = {
    alias: spec.strategy_name
    for spec in _CHOICE_BASELINE_SPECS.values()
    for alias in spec.aliases
}


def choice_baseline_strategy_aliases() -> dict[str, str]:
    return dict(_CHOICE_BASELINE_ALIAS_TO_NAME)


def choice_baseline_strategy_names() -> set[str]:
    return set(_CHOICE_BASELINE_SPECS)


def normalize_choice_baseline_strategy(strategy_name: str) -> str | None:
    return _CHOICE_BASELINE_ALIAS_TO_NAME.get(str(strategy_name).strip())


def get_choice_baseline_spec(strategy_name: str) -> ChoiceBaselineSpec | None:
    normalized = normalize_choice_baseline_strategy(strategy_name)
    if normalized is None:
        return None
    return _CHOICE_BASELINE_SPECS.get(normalized)


__all__ = [
    "ChoiceBaselineSpec",
    "choice_baseline_strategy_aliases",
    "choice_baseline_strategy_names",
    "get_choice_baseline_spec",
    "normalize_choice_baseline_strategy",
]
