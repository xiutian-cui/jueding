"""Condition: binary judgments and the confidence derived from them."""

from __future__ import annotations

import pytest

from conftest import FakeBackend, boolean_answer
from jueding import (
    BackendResponseError,
    BooleanQuestion,
    Condition,
    ConditionResult,
    LowConfidenceError,
)


def build(probability: float = 0.98, **kwargs) -> Condition:
    backend = kwargs.pop("backend", None) or FakeBackend(answer=boolean_answer(probability))
    return Condition(instructions="Is this urgent?", backend=backend, **kwargs)


def test_returns_a_condition_result() -> None:
    result = build()("ticket")
    assert isinstance(result, ConditionResult)
    assert result.value is True


def test_probabilities_cover_both_outcomes() -> None:
    result = build(0.98)("ticket")
    assert result.probabilities == {True: 0.98, False: pytest.approx(0.02)}


def test_confidence_is_the_larger_side_of_the_split() -> None:
    assert build(0.02)("ticket").confidence == pytest.approx(0.98)


def test_a_low_probability_answers_no() -> None:
    result = build(0.02)("ticket")
    assert result.value is False


def test_an_even_split_answers_yes() -> None:
    result = build(0.5)("ticket")
    assert result.value is True
    assert result.confidence == 0.5


def test_criteria_are_sent_when_supplied() -> None:
    condition = build(true="Has a deadline.", false="No deadline.")
    condition("ticket")
    question = condition._backend.question
    assert isinstance(question, BooleanQuestion)
    assert question.true == "Has a deadline."
    assert question.false == "No deadline."


def test_criteria_default_to_unset() -> None:
    condition = build()
    condition("ticket")
    assert condition._backend.question.true is None
    assert condition._backend.question.false is None


def test_name_defaults_to_the_class_name() -> None:
    condition = build()
    condition("ticket")
    assert condition.name == "Condition"
    assert condition._backend.question_name == "Condition"


def test_truth_testing_uses_the_answer() -> None:
    assert bool(build(0.98)("ticket")) is True
    assert bool(build(0.02)("ticket")) is False


def test_truth_testing_below_the_threshold_raises() -> None:
    result = build(0.6, threshold=0.8)("ticket")
    with pytest.raises(LowConfidenceError):
        bool(result)


def test_value_is_readable_below_the_threshold() -> None:
    result = build(0.6, threshold=0.8)("ticket")
    assert result.value is True
    assert result.meets_threshold is False


def test_invalid_probability_is_rejected() -> None:
    backend = FakeBackend(answer=boolean_answer(1.8))
    with pytest.raises(BackendResponseError, match="probability must be between 0 and 1"):
        build(backend=backend)("ticket")


def test_wrong_answer_type_is_rejected() -> None:
    from conftest import choice_answer

    backend = FakeBackend(answer=choice_answer())
    with pytest.raises(BackendResponseError, match="expected BooleanAnswer"):
        build(backend=backend)("ticket")
