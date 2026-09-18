"""The data types that cross the boundary to Jev, and the seam they cross it through.

Nothing here imports `typesafe_sdk`. These questions and answers mirror Jev's three
primitives in plain dataclasses, so the rest of Jueding — and anything a test puts in
Jev's place — never handles an SDK object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

__all__ = [
    "Answer",
    "Backend",
    "BackendResponse",
    "BooleanAnswer",
    "BooleanQuestion",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "JSONContent",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "Usage",
]

JSONContent: TypeAlias = str | Mapping[str, Any] | Sequence[Any]
"""Text, a JSON object, or an array — the shapes Jev accepts as content."""


@dataclass(frozen=True, kw_only=True)
class ChoiceQuestion:
    """Select one of several named labels."""

    instructions: str
    criteria: Mapping[str, JSONContent | None]
    """Label names mapped to descriptions; `None` leaves a label described by its name."""


@dataclass(frozen=True, kw_only=True)
class BooleanQuestion:
    """Judge a yes/no statement."""

    instructions: str
    true: JSONContent | None = None
    """What counts as a yes answer."""
    false: JSONContent | None = None
    """What counts as a no answer."""


@dataclass(frozen=True, kw_only=True)
class ScoreQuestion:
    """Rate content against an ordered rubric, scored from zero."""

    instructions: str
    criteria: Sequence[JSONContent]
    """Ordered level descriptions; position is the score."""


Question: TypeAlias = ChoiceQuestion | BooleanQuestion | ScoreQuestion
"""A question a backend can be asked."""


@dataclass(frozen=True, kw_only=True)
class ChoiceAnswer:
    """The selected label, Jev's confidence in it, and the full distribution."""

    label: str
    confidence: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True, kw_only=True)
class BooleanAnswer:
    """The probability that the answer is yes.

    There is no separate confidence: for a binary judgment the probability is the
    whole distribution, and confidence is derived from it.
    """

    probability: float


@dataclass(frozen=True, kw_only=True)
class ScoreAnswer:
    """The expected score, Jev's confidence, and the distribution over levels."""

    score: float
    confidence: float
    probabilities: Mapping[int, float]
    legend: Mapping[int, JSONContent] = field(default_factory=dict)
    """The rubric echoed back, keyed by level."""


Answer: TypeAlias = ChoiceAnswer | BooleanAnswer | ScoreAnswer
"""An answer to a single question."""


@dataclass(frozen=True, kw_only=True)
class Usage:
    """Token counts reported by Jev."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, kw_only=True)
class BackendResponse:
    """Answers to one request, keyed by the question names that were asked."""

    answers: Mapping[str, Answer]
    model: str
    usage: Usage | None = None
    request_id: str | None = None


@runtime_checkable
class Backend(Protocol):
    """The single method Jueding uses to reach Jev.

    `TypeSafeBackend` is the real implementation. The protocol exists so that a test
    can supply fixed answers instead, without credentials or a network — not as a
    plug-in point for other providers.

    The mapping is plural because one System One request answers several questions
    about the same state. V1 always sends exactly one, but keeping the seam plural
    means batching later needs no change here.
    """

    def ask(self, *, state: JSONContent, questions: Mapping[str, Question]) -> BackendResponse:
        """Answer `questions` about `state`.

        Which model answers is this object's business, not the caller's.
        """
        ...
