"""Score: expected scores over an ordered rubric."""

from __future__ import annotations

import pytest

from conftest import FakeBackend, score_answer
from jueding import (
    BackendResponseError,
    ConfigurationError,
    LowConfidenceError,
    Score,
    ScoreQuestion,
    ScoreResult,
)

RUBRIC = ["Cosmetic", "Workaround exists", "Blocking"]


def build(**kwargs) -> Score:
    backend = kwargs.pop("backend", None) or FakeBackend(answer=score_answer())
    rubric = kwargs.pop("rubric", RUBRIC)
    return Score(instructions="How severe is this?", rubric=rubric, backend=backend, **kwargs)


class TestConstruction:
    def test_rubric_needs_at_least_two_levels(self) -> None:
        with pytest.raises(ConfigurationError, match="at least two levels"):
            build(rubric=["Only one"])

    def test_rubric_must_not_be_a_string(self) -> None:
        with pytest.raises(ConfigurationError, match="must be a sequence"):
            build(rubric="Cosmetic")

    def test_rubric_must_be_a_sequence(self) -> None:
        with pytest.raises(ConfigurationError, match="must be a sequence"):
            build(rubric=42)

    def test_rubric_is_sent_as_ordered_criteria(self) -> None:
        score = build()
        score("bug")
        question = score._backend.question
        assert isinstance(question, ScoreQuestion)
        assert question.criteria == tuple(RUBRIC)

    def test_name_defaults_to_the_class_name(self) -> None:
        score = build()
        score("bug")
        assert score.name == "Score"
        assert score._backend.question_name == "Score"


class TestCall:
    def test_value_is_the_expected_score(self) -> None:
        result = build()("bug")
        assert isinstance(result, ScoreResult)
        assert result.value == 1.3
        assert result.confidence == 0.54

    def test_probabilities_are_keyed_by_level(self) -> None:
        assert build()("bug").probabilities == {0: 0.0, 1: 0.7, 2: 0.3}

    def test_missing_levels_are_filled_with_zero(self) -> None:
        backend = FakeBackend(answer=score_answer(probabilities={1: 0.7, 2: 0.3}))
        assert build(backend=backend)("bug").probabilities[0] == 0.0

    def test_legend_falls_back_to_the_rubric(self) -> None:
        assert build()("bug").legend == {0: "Cosmetic", 1: "Workaround exists", 2: "Blocking"}

    def test_legend_from_the_provider_is_kept(self) -> None:
        backend = FakeBackend(answer=score_answer(legend={0: "a", 1: "b", 2: "c"}))
        assert build(backend=backend)("bug").legend == {0: "a", 1: "b", 2: "c"}


class TestOrdering:
    def test_compares_against_numbers(self) -> None:
        result = build()("bug")
        assert result >= 1.3
        assert result > 1.0
        assert result < 2.0
        assert result <= 1.3
        assert not result > 2.0

    def test_equality_against_an_integer_level(self) -> None:
        backend = FakeBackend(answer=score_answer(score=2.0, probabilities={2: 1.0}))
        assert build(backend=backend)("bug") == 2

    def test_ordering_is_not_defined_against_other_types(self) -> None:
        result = build()("bug")
        with pytest.raises(TypeError):
            result < "high"  # noqa: B015
        with pytest.raises(TypeError):
            result <= "high"  # noqa: B015
        with pytest.raises(TypeError):
            result > "high"  # noqa: B015
        with pytest.raises(TypeError):
            result >= "high"  # noqa: B015

    @pytest.mark.parametrize("comparison", [
        lambda r: r < 2.0,
        lambda r: r <= 2.0,
        lambda r: r > 1.0,
        lambda r: r >= 1.0,
        lambda r: r == 1.3,
    ])
    def test_ordering_below_the_threshold_raises(self, comparison) -> None:
        result = build(threshold=0.8)("bug")
        with pytest.raises(LowConfidenceError):
            comparison(result)


class TestBadResponses:
    def test_levels_outside_the_rubric_are_rejected(self) -> None:
        backend = FakeBackend(answer=score_answer(probabilities={1: 0.5, 9: 0.5}))
        with pytest.raises(BackendResponseError, match=r"outside the rubric: \[9\]"):
            build(backend=backend)("bug")

    @pytest.mark.parametrize("bad", [-0.5, 2.5, float("nan")])
    def test_score_outside_the_rubric_range_is_rejected(self, bad: float) -> None:
        backend = FakeBackend(answer=score_answer(score=bad))
        with pytest.raises(BackendResponseError, match="outside the rubric range 0 to 2"):
            build(backend=backend)("bug")

    @pytest.mark.parametrize("bad", ["1.3", True, None])
    def test_non_numeric_score_is_rejected(self, bad: object) -> None:
        backend = FakeBackend(answer=score_answer(score=bad))
        with pytest.raises(BackendResponseError, match="score must be a number"):
            build(backend=backend)("bug")

    def test_wrong_answer_type_is_rejected(self) -> None:
        from conftest import boolean_answer

        backend = FakeBackend(answer=boolean_answer())
        with pytest.raises(BackendResponseError, match="expected ScoreAnswer"):
            build(backend=backend)("bug")
