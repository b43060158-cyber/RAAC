"""Hot-pluggable reasoning-bank retrieval (scoring + selection).

Importing this package registers all built-in scorers and selectors.
"""

from __future__ import annotations

from .base import (
    RetrievalContext,
    RetrievalError,
    ScoredRecord,
    Scorer,
    Selector,
)
from .registry import (
    DEFAULT_SCORER_ID,
    DEFAULT_SELECTOR_ID,
    get_scorer,
    get_selector,
    register_scorer,
    register_selector,
    registered_scorer_ids,
    registered_selector_ids,
)

# Import for side-effect registration of built-ins.
from . import scorers as _scorers  # noqa: F401
from . import selectors as _selectors  # noqa: F401

__all__ = [
    "RetrievalContext",
    "RetrievalError",
    "ScoredRecord",
    "Scorer",
    "Selector",
    "DEFAULT_SCORER_ID",
    "DEFAULT_SELECTOR_ID",
    "get_scorer",
    "get_selector",
    "register_scorer",
    "register_selector",
    "registered_scorer_ids",
    "registered_selector_ids",
]
