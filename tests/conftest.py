"""Shared fixtures: a deterministic backend and the enums the tests decide over.

No test in this file or its siblings makes a network request or needs credentials.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jueding.models import (
    Answer,
    BackendResponse,
    BooleanAnswer,
    ChoiceAnswer,
    Question,
    ScoreAnswer,
    Usage,
)


class Intent(StrEnum):
    """The enum most tests decide over."""

    SEARCH = "search"
    ANALYZE = "analyze"
    CHAT = "chat"


class Single(StrEnum):
    """A one-member enum, for the degenerate ranking cases."""

    ONLY = "only"


@dataclass
class FakeBackend:
    """A backend that returns whatever it was constructed with and records its calls."""

    answer: Answer | None = None
    answers: Mapping[str, Answer] | None = None
    model: str = "fake-1"
    usage: Usage | None = field(default_factory=lambda: Usage(input_tokens=120, output_tokens=12))
    request_id: str | None = "req_fake"
    error: Exception | None = None
    calls: list[tuple[Any, dict[str, Question]]] = field(default_factory=list)

    def ask(self, state: Any, questions: Mapping[str, Question]) -> BackendResponse:
        self.calls.append((state, dict(questions)))
        if self.error is not None:
            raise self.error
        answers = (
            dict(self.answers)
            if self.answers is not None
            else {name: self.answer for name in questions}
        )
        return BackendResponse(
            answers=answers,
            model=self.model,
            usage=self.usage,
            request_id=self.request_id,
        )

    @property
    def state(self) -> Any:
        """State passed to the most recent call."""
        return self.calls[-1][0]

    @property
    def question(self) -> Question:
        """The single question sent in the most recent call."""
        return next(iter(self.calls[-1][1].values()))

    @property
    def question_name(self) -> str:
        """The name the question was filed under in the most recent call."""
        return next(iter(self.calls[-1][1]))


def choice_answer(
    label: str = "analyze",
    confidence: float = 0.91,
    probabilities: Mapping[str, float] | None = None,
) -> ChoiceAnswer:
    """A choice answer with a sensible default distribution."""
    if probabilities is None:
        probabilities = {"search": 0.06, "analyze": 0.91, "chat": 0.03}
    return ChoiceAnswer(label=label, confidence=confidence, probabilities=probabilities)


def boolean_answer(probability: float = 0.98) -> BooleanAnswer:
    """A yes/no answer."""
    return BooleanAnswer(probability=probability)


def score_answer(
    score: float = 1.3,
    confidence: float = 0.54,
    probabilities: Mapping[int, float] | None = None,
    legend: Mapping[int, Any] | None = None,
) -> ScoreAnswer:
    """A score answer over a three-level rubric."""
    if probabilities is None:
        probabilities = {0: 0.0, 1: 0.7, 2: 0.3}
    return ScoreAnswer(
        score=score,
        confidence=confidence,
        probabilities=probabilities,
        legend=legend if legend is not None else {},
    )
