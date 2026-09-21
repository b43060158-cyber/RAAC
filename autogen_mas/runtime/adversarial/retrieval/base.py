"""Core abstractions for hot-pluggable reasoning-bank retrieval.

The per-record *scoring* and the *top-k selection* of the adversarial
reasoning bank are factored into two independent, registrable components:

* :class:`Scorer` — takes the already pre-filtered candidate records and the
  query :class:`RetrievalContext` and returns scored records. Receiving the
  whole candidate set (rather than one record at a time) lets a scorer perform
  corpus-level normalisation (IDF, calibration, …) when it wants to.
* :class:`Selector` — takes the scored records and decides which ones survive
  into the borrowing context (sorting, top-k, diversity, soft thresholds).

A strategy chooses its scorer/selector/params at runtime via configuration, so
new methods can be added by registering a function without touching the call
site. See :mod:`.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ....models import TaskRecord


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot satisfy the no-empty-generation contract.

    The adversarial review path must never generate a rationale without
    referencing corpus material. When family routing is active and every
    permitted candidate pool is empty, this is raised rather than silently
    falling back to corpus-free generation.
    """


@dataclass(frozen=True)
class RetrievalContext:
    """Immutable per-query features shared by scorer and selector."""

    question_record: "TaskRecord"
    dataset_name: str | None
    target_agent_id: str
    target_current_option_ids: list[str]
    target_current_reasoning: str
    target_current_final_answer: str
    desired_target_shift: list[str]
    desired_target_final_answer: str
    review_phase_mode: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredRecord:
    """A candidate record paired with its computed relevance score."""

    score: float
    record: dict[str, Any]


class Scorer(Protocol):
    def __call__(
        self,
        records: list[dict[str, Any]],
        ctx: RetrievalContext,
    ) -> list[ScoredRecord]: ...


class Selector(Protocol):
    def __call__(
        self,
        scored: list[ScoredRecord],
        ctx: RetrievalContext,
        top_k: int,
    ) -> list[dict[str, Any]]: ...
