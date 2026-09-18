"""Judgments: what a decision returns.

Every result keeps Jev's full distribution alongside the selected value, so
uncertainty is never collapsed on the way out. The three concrete types are siblings,
not subclasses of one another: a condition is not a kind of decision, and each adds
only what its own primitive makes meaningful.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from jueding.models import JSONContent, Usage
from jueding.errors import LowConfidenceError

__all__ = ["ConditionResult", "DecisionResult", "Judgment", "ScoreResult"]

T = TypeVar("T")
E = TypeVar("E", bound=Enum)


@dataclass(frozen=True, eq=False, kw_only=True)
class Judgment(Generic[T]):
    """What every primitive returns: a selected value that still carries its uncertainty.

    A judgment stands in for its value in comparisons, so it can be used directly:

    ```python
    if decide_intent(user_input=text) == Intent.SEARCH:
        run_search(text)
    ```

    When a `threshold` was configured and the confidence falls below it, that
    comparison raises `LowConfidenceError` instead of quietly answering. Reading
    `value`, `confidence`, `probabilities` or `meets_threshold` never raises.
    """

    name: str
    """The decision's name; also the question name sent to Jev."""
    value: T
    """The selected value."""
    confidence: float
    """Confidence in `value`, from 0 to 1."""
    probabilities: Mapping[T, float]
    """The complete distribution Jev returned."""
    threshold: float | None
    """The configured threshold, or `None` when no confidence policy is set."""
    model: str
    """The model that answered, which may differ from the alias that was requested."""
    usage: Usage | None
    """Token usage, when Jev reported it."""
    request_id: str | None
    """Jev's request id, useful when asking TypeSafe about a specific answer."""
    latency_seconds: float
    """Wall-clock duration of the backend call."""

    @property
    def meets_threshold(self) -> bool:
        """Whether confidence reached the threshold.

        `True` when no threshold is configured: with no policy in place there is
        nothing for a result to fail.
        """
        return True if self.threshold is None else self.confidence >= self.threshold

    def _value_types(self) -> tuple[type, ...]:
        """Types that count as comparing against the answer itself."""
        return (type(self.value),)

    def _guard(self) -> None:
        """Raise if this judgment is being used as though it were certain."""
        if not self.meets_threshold:
            raise LowConfidenceError(
                f"{self.name}: confidence {self.confidence:.3f} is below the configured "
                f"threshold {self.threshold:.3f}; selected {self.value!r}. Check "
                f"`meets_threshold` first, or read `.value` to use it anyway.",
                self,
            )

    def __eq__(self, other: object) -> bool:
        """Compare against the selected value, guarded by the threshold.

        Comparing two judgments compares their values only — not confidence or
        probabilities — and is never guarded, so results stay usable in collections
        and test assertions.
        """
        if isinstance(other, Judgment):
            return bool(self.value == other.value)
        if isinstance(other, self._value_types()):
            self._guard()
            return bool(self.value == other)
        return NotImplemented

    def __hash__(self) -> int:
        """Hash as the selected value, so lookups keyed by the value work."""
        return hash(self.value)


@dataclass(frozen=True, eq=False, kw_only=True)
class DecisionResult(Judgment[E]):
    """A choice among enum members.

    `ranked` and `margin` live here rather than on `Judgment` because they only mean
    something for unordered alternatives.
    """

    @property
    def ranked(self) -> tuple[tuple[E, float], ...]:
        """Members with their probabilities, most likely first.

        Ties break on enum definition order, so the ranking is stable across calls.
        `ranked[0][0]` is normally `value`; if Jev ever disagreed with its own
        distribution, `value` is what it selected and wins.
        """
        order = {member: index for index, member in enumerate(type(self.value))}
        return tuple(
            sorted(
                self.probabilities.items(),
                key=lambda item: (-item[1], order[item[0]]),
            )
        )

    @property
    def margin(self) -> float:
        """The gap between the two most likely members; `0.0` when there are fewer than two.

        A narrow margin is a different condition from diffuse uncertainty: it means two
        specific candidates are in contention, which is the case worth disambiguating.
        """
        ranked = self.ranked
        if len(ranked) < 2:
            return 0.0
        return ranked[0][1] - ranked[1][1]


@dataclass(frozen=True, eq=False, kw_only=True)
class ConditionResult(Judgment[bool]):
    """A yes/no judgment."""

    def __bool__(self) -> bool:
        """Truth-test as the answer, guarded by the threshold.

        `if is_urgent(ticket):` reads naturally and, with a threshold configured,
        refuses to answer when the model was not confident enough.
        """
        self._guard()
        return self.value


@dataclass(frozen=True, eq=False, kw_only=True)
class ScoreResult(Judgment[float]):
    """An expected score over an ordered rubric.

    `value` is the probability-weighted mean of the level numbers, so it may fall
    between levels: `1.3` means mostly level 1, leaning 2.
    """

    legend: Mapping[int, JSONContent]
    """The rubric, keyed by level, so the score can be interpreted."""

    def _value_types(self) -> tuple[type, ...]:
        return (int, float)

    def __lt__(self, other: Any) -> bool:
        """Order against a number, guarded by the threshold."""
        if not isinstance(other, self._value_types()):
            return NotImplemented
        self._guard()
        return self.value < other

    def __le__(self, other: Any) -> bool:
        """Order against a number, guarded by the threshold."""
        if not isinstance(other, self._value_types()):
            return NotImplemented
        self._guard()
        return self.value <= other

    def __gt__(self, other: Any) -> bool:
        """Order against a number, guarded by the threshold."""
        if not isinstance(other, self._value_types()):
            return NotImplemented
        self._guard()
        return self.value > other

    def __ge__(self, other: Any) -> bool:
        """Order against a number, guarded by the threshold."""
        if not isinstance(other, self._value_types()):
            return NotImplemented
        self._guard()
        return self.value >= other
