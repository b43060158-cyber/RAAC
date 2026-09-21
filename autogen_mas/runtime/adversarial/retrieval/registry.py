"""Registry + resolution for retrieval scorers and selectors.

Adding a new method is a pure registration: decorate a function with
``@register_scorer("my_id")`` / ``@register_selector("my_id")`` and it becomes
selectable at runtime via configuration. Unknown ids resolve to the default
implementation with a warning so a misconfiguration never crashes a run.
"""

from __future__ import annotations

import logging
from typing import Callable

from .base import Scorer, Selector

logger = logging.getLogger(__name__)

DEFAULT_SCORER_ID = "lexical_overlap_v1"
DEFAULT_SELECTOR_ID = "topk_sorted_v1"

_SCORERS: dict[str, Scorer] = {}
_SELECTORS: dict[str, Selector] = {}


def register_scorer(scorer_id: str) -> Callable[[Scorer], Scorer]:
    def decorator(func: Scorer) -> Scorer:
        if scorer_id in _SCORERS:
            raise ValueError(f"Duplicate retrieval scorer id: {scorer_id!r}")
        _SCORERS[scorer_id] = func
        return func

    return decorator


def register_selector(selector_id: str) -> Callable[[Selector], Selector]:
    def decorator(func: Selector) -> Selector:
        if selector_id in _SELECTORS:
            raise ValueError(f"Duplicate retrieval selector id: {selector_id!r}")
        _SELECTORS[selector_id] = func
        return func

    return decorator


def get_scorer(scorer_id: str | None) -> Scorer:
    if scorer_id and scorer_id in _SCORERS:
        return _SCORERS[scorer_id]
    if scorer_id:
        logger.warning(
            "Unknown reasoning-bank scorer %r; falling back to %r.",
            scorer_id,
            DEFAULT_SCORER_ID,
        )
    return _SCORERS[DEFAULT_SCORER_ID]


def get_selector(selector_id: str | None) -> Selector:
    if selector_id and selector_id in _SELECTORS:
        return _SELECTORS[selector_id]
    if selector_id:
        logger.warning(
            "Unknown reasoning-bank selector %r; falling back to %r.",
            selector_id,
            DEFAULT_SELECTOR_ID,
        )
    return _SELECTORS[DEFAULT_SELECTOR_ID]


def registered_scorer_ids() -> tuple[str, ...]:
    return tuple(sorted(_SCORERS))


def registered_selector_ids() -> tuple[str, ...]:
    return tuple(sorted(_SELECTORS))
