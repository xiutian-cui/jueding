"""The three decision primitives.

Each primitive is a callable object built once and called many times: the question is
fixed at construction, the state varies per call. Construction validates everything it
can, so a misconfigured decision fails where it is defined rather than on its first
production call.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from time import perf_counter
from typing import Any, ClassVar, Generic, TypeVar

from jueding._state import normalize_state
from jueding.models import (
    Answer,
    Backend,
    BackendResponse,
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    JSONContent,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from jueding.errors import BackendResponseError, ConfigurationError
from jueding.results import ConditionResult, DecisionResult, Judgment, ScoreResult

__all__ = ["Condition", "Decision", "Score"]

E = TypeVar("E", bound=Enum)

_SUM_TOLERANCE = 0.02
"""How far a distribution may sum from 1 before it is rejected."""

_EPSILON = 1e-9
"""Slack for floating-point values that should sit in [0, 1]."""

_MAX_LABELS = 255
"""Labels allowed in a single choice question by the TypeSafe API."""


def _check_probability(name: str, field: str, value: Any) -> float:
    """Reject a probability that is not a finite number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendResponseError(f"{name}: {field} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number) or not (-_EPSILON <= number <= 1 + _EPSILON):
        raise BackendResponseError(f"{name}: {field} must be between 0 and 1, got {number!r}.")
    return number


def _check_distribution(name: str, probabilities: Mapping[Any, float]) -> None:
    """Reject a distribution whose values are invalid or do not sum to about 1.

    Values are validated but never renormalized: a distribution that does not add up
    is a Jev problem, and silently rescaling it would hide it.
    """
    if not probabilities:
        raise BackendResponseError(f"{name}: probabilities are empty.")
    total = 0.0
    for key, value in probabilities.items():
        total += _check_probability(name, f"probabilities[{key!r}]", value)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise BackendResponseError(
            f"{name}: probabilities sum to {total:.4f}, which is not within "
            f"{_SUM_TOLERANCE} of 1."
        )


class _Primitive(ABC):
    """Shared machinery: state normalization, the call, and answer validation.

    Subclasses supply the question to ask and the judgment to build from the answer;
    everything between those two points is identical for all three primitives.
    """

    _answer_type: ClassVar[type]

    def __init__(
        self,
        *,
        instructions: str,
        backend: Backend,
        name: str,
        threshold: float | None,
    ) -> None:
        if not isinstance(instructions, str) or not instructions.strip():
            raise ConfigurationError(f"{name}: instructions must be a non-empty string.")
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ConfigurationError(f"{name}: threshold must be a number or None.")
            if not 0.0 <= float(threshold) <= 1.0:
                raise ConfigurationError(
                    f"{name}: threshold must be between 0 and 1, got {threshold!r}."
                )
            threshold = float(threshold)
        self.name = name
        """The question name sent to Jev."""
        self.instructions = instructions
        self.threshold = threshold
        self._backend = backend

    @abstractmethod
    def _question(self) -> Question:
        """Build the Jev question this primitive asks."""

    @abstractmethod
    def _build(self, answer: Any, response: BackendResponse, latency: float) -> Judgment[Any]:
        """Turn a validated answer into this primitive's judgment."""

    def __call__(self, *args: Any, **kwargs: Any) -> Judgment[Any]:
        """Judge the supplied state.

        Pass one positional argument to use it as the state directly, or keyword
        arguments to build a state object from them. `__call__` takes no options of
        its own: every keyword is state, so per-call overrides would be indistinguishable
        from content.
        """
        state = normalize_state(self.name, args, kwargs)
        started = perf_counter()
        response = self._backend.ask(state=state, questions={self.name: self._question()})
        latency = perf_counter() - started
        return self._build(self._answer(response), response, latency)

    def _answer(self, response: BackendResponse) -> Answer:
        """Pull this decision's answer out of the response and check its type."""
        answer = response.answers.get(self.name)
        if answer is None:
            raise BackendResponseError(
                f"{self.name}: the backend returned no answer for this question."
            )
        if not isinstance(answer, self._answer_type):
            raise BackendResponseError(
                f"{self.name}: expected {self._answer_type.__name__}, got "
                f"{type(answer).__name__}."
            )
        return answer


class Decision(_Primitive, Generic[E]):
    """Choose one member of a string-valued enum.

    ```python
    decide_intent = Decision(instructions="Determine the user's primary intent.", choices=Intent,
        criteria={Intent.SEARCH: "Retrieve existing information."},
        backend=backend,
    )
    if decide_intent(user_input=text) == Intent.SEARCH:
        run_search(text)
    ```
    """

    _answer_type = ChoiceAnswer

    def __init__(
        self,
        *,
        instructions: str,
        choices: type[E],
        backend: Backend,
        criteria: Mapping[E | str, JSONContent | None] | None = None,
        threshold: float | None = None,
        name: str | None = None,
    ) -> None:
        """Build a decision over `choices`.

        Args:
            instructions: What the model should decide.
            choices: An `Enum` whose members all have unique, non-empty string values.
            backend: Where to send the question.
            criteria: Optional descriptions per member. Values may be text, a mapping or
                a sequence, which is how stable domain knowledge — definitions, examples,
                counter-examples — is attached to a label.
            threshold: Confidence below which the result refuses to be compared. `None`
                applies no policy; there is no default threshold.
            name: The question name sent to Jev. Defaults to the enum's name followed
                by the class name, such as `UserIntentDecision`.
        """
        if not (isinstance(choices, type) and issubclass(choices, Enum)):
            raise ConfigurationError(
                f"{name or 'decision'}: choices must be an Enum subclass, got {choices!r}."
            )
        resolved = name or f"{choices.__name__}{type(self).__name__}"
        members = list(choices)
        if not members:
            raise ConfigurationError(f"{resolved}: {choices.__name__} has no members.")
        if len(members) > _MAX_LABELS:
            raise ConfigurationError(
                f"{resolved}: {choices.__name__} has {len(members)} members; "
                f"at most {_MAX_LABELS} are supported."
            )
        aliases = sorted(set(choices.__members__) - {member.name for member in members})
        if aliases:
            raise ConfigurationError(
                f"{resolved}: {choices.__name__} has aliases {aliases}, which would be "
                f"invisible as choices. Give every member a distinct value."
            )
        seen: dict[str, E] = {}
        for member in members:
            value = member.value
            if not isinstance(value, str) or not value:
                raise ConfigurationError(
                    f"{resolved}: {choices.__name__}.{member.name} must have a non-empty "
                    f"string value, got {value!r}."
                )
            seen[value] = member
        super().__init__(
            instructions=instructions,
            backend=backend,
            name=resolved,
            threshold=threshold,
        )
        self.choices = choices
        self._members = seen
        self._criteria = self._resolve_criteria(criteria)

    def _resolve_criteria(
        self, criteria: Mapping[E | str, JSONContent | None] | None
    ) -> dict[str, JSONContent | None]:
        """Map every member to its description, defaulting to `None`."""
        described: dict[str, JSONContent | None] = {value: None for value in self._members}
        for key, description in (criteria or {}).items():
            if isinstance(key, self.choices):
                label = key.value
            elif isinstance(key, str) and key in described:
                label = key
            else:
                raise ConfigurationError(
                    f"{self.name}: criteria key {key!r} is not a member of "
                    f"{self.choices.__name__}."
                )
            described[label] = description
        return described

    def _question(self) -> Question:
        return ChoiceQuestion(instructions=self.instructions, criteria=self._criteria)

    def _build(
        self, answer: ChoiceAnswer, response: BackendResponse, latency: float
    ) -> DecisionResult[E]:
        unknown = set(answer.probabilities) - set(self._members)
        if unknown:
            raise BackendResponseError(
                f"{self.name}: probabilities contain labels that are not members of "
                f"{self.choices.__name__}: {sorted(unknown)}."
            )
        if answer.label not in self._members:
            raise BackendResponseError(
                f"{self.name}: selected label {answer.label!r} is not a member of "
                f"{self.choices.__name__}."
            )
        _check_distribution(self.name, answer.probabilities)
        probabilities = {
            member: float(answer.probabilities.get(label, 0.0))
            for label, member in self._members.items()
        }
        return DecisionResult(
            name=self.name,
            value=self._members[answer.label],
            confidence=_check_probability(self.name, "confidence", answer.confidence),
            probabilities=probabilities,
            threshold=self.threshold,
            model=response.model,
            usage=response.usage,
            request_id=response.request_id,
            latency_seconds=latency,
        )


class Condition(_Primitive):
    """Judge a yes/no question without collapsing it to a bare boolean.

    ```python
    is_urgent = Condition(instructions="Does this communicate urgency?", backend=backend)
    if is_urgent(ticket):
        escalate(ticket)
    ```
    """

    _answer_type = BooleanAnswer

    def __init__(
        self,
        *,
        instructions: str,
        backend: Backend,
        true: JSONContent | None = None,
        false: JSONContent | None = None,
        threshold: float | None = None,
        name: str | None = None,
    ) -> None:
        """Build a yes/no judgment.

        Args:
            instructions: The question or statement to evaluate, phrased so that a high
                probability means yes.
            backend: Where to send the question.
            true: What counts as a yes answer.
            false: What counts as a no answer.
            threshold: Confidence below which the result refuses to be compared or
                truth-tested. Note that confidence here is `max(p, 1 - p)` and so never
                falls below 0.5; a threshold under 0.5 is always met.
            name: The question name sent to Jev. Defaults to the class name.
        """
        super().__init__(
            instructions=instructions,
            backend=backend,
            name=name or type(self).__name__,
            threshold=threshold,
        )
        self.true = true
        self.false = false

    def _question(self) -> Question:
        return BooleanQuestion(instructions=self.instructions, true=self.true, false=self.false)

    def _build(
        self, answer: BooleanAnswer, response: BackendResponse, latency: float
    ) -> ConditionResult:
        probability = _check_probability(self.name, "probability", answer.probability)
        return ConditionResult(
            name=self.name,
            value=probability >= 0.5,
            confidence=max(probability, 1.0 - probability),
            probabilities={True: probability, False: 1.0 - probability},
            threshold=self.threshold,
            model=response.model,
            usage=response.usage,
            request_id=response.request_id,
            latency_seconds=latency,
        )


class Score(_Primitive):
    """Rate content against an ordered rubric and return the expected score.

    ```python
    severity = Score(
        instructions="How severe is this?",
        rubric=["Cosmetic", "Degraded", "Blocking"],
        backend=backend,
    )
    if severity(bug) >= 1.5:
        page_oncall(bug)
    ```
    """

    _answer_type = ScoreAnswer

    def __init__(
        self,
        *,
        instructions: str,
        rubric: Sequence[JSONContent],
        backend: Backend,
        threshold: float | None = None,
        name: str | None = None,
    ) -> None:
        """Build a score over an ordered rubric.

        Args:
            instructions: What the model should rate.
            rubric: Ordered level descriptions, lowest first. Position is the score, so
                the first entry scores 0. At least two levels are required.
            backend: Where to send the question.
            threshold: Confidence below which the result refuses to be compared or ordered.
            name: The question name sent to Jev. Defaults to the class name.
        """
        resolved = name or type(self).__name__
        if isinstance(rubric, (str, bytes)) or not isinstance(rubric, Sequence):
            raise ConfigurationError(
                f"{resolved}: rubric must be a sequence of level descriptions, got "
                f"{type(rubric).__name__}."
            )
        if len(rubric) < 2:
            raise ConfigurationError(
                f"{resolved}: rubric needs at least two levels, got {len(rubric)}."
            )
        super().__init__(
            instructions=instructions,
            backend=backend,
            name=resolved,
            threshold=threshold,
        )
        self.rubric = tuple(rubric)

    def _question(self) -> Question:
        return ScoreQuestion(instructions=self.instructions, criteria=self.rubric)

    def _build(
        self, answer: ScoreAnswer, response: BackendResponse, latency: float
    ) -> ScoreResult:
        levels = range(len(self.rubric))
        unknown = set(answer.probabilities) - set(levels)
        if unknown:
            raise BackendResponseError(
                f"{self.name}: probabilities contain levels outside the rubric: "
                f"{sorted(unknown)}."
            )
        _check_distribution(self.name, answer.probabilities)
        if isinstance(answer.score, bool) or not isinstance(answer.score, (int, float)):
            raise BackendResponseError(
                f"{self.name}: score must be a number, got {answer.score!r}."
            )
        score = float(answer.score)
        highest = len(self.rubric) - 1
        if not math.isfinite(score) or not (-_EPSILON <= score <= highest + _EPSILON):
            raise BackendResponseError(
                f"{self.name}: score {score!r} is outside the rubric range 0 to {highest}."
            )
        legend = dict(answer.legend) if answer.legend else dict(enumerate(self.rubric))
        return ScoreResult(
            name=self.name,
            value=score,
            confidence=_check_probability(self.name, "confidence", answer.confidence),
            probabilities={level: float(answer.probabilities.get(level, 0.0)) for level in levels},
            threshold=self.threshold,
            model=response.model,
            usage=response.usage,
            request_id=response.request_id,
            latency_seconds=latency,
            legend=legend,
        )
