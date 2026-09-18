"""Judgments: transparent comparison, hashing, and the low-confidence guard."""

from __future__ import annotations

import pytest

from conftest import FakeBackend, Intent, choice_answer
from jueding import Decision, Judgment, LowConfidenceError


def build(confidence: float = 0.91, threshold: float | None = None) -> Decision[Intent]:
    backend = FakeBackend(answer=choice_answer(confidence=confidence))
    return Decision(
        instructions="Determine intent.",
        choices=Intent,
        backend=backend,
        threshold=threshold,
    )


def test_a_result_is_a_judgment() -> None:
    assert isinstance(build()("text"), Judgment)


class TestTransparentComparison:
    def test_compares_equal_to_its_value(self) -> None:
        assert build()("text") == Intent.ANALYZE

    def test_compares_unequal_to_other_members(self) -> None:
        assert build()("text") != Intent.SEARCH

    def test_is_not_equal_to_unrelated_types(self) -> None:
        result = build()("text")
        assert result != "analyze"
        assert result != 42

    def test_hashes_as_its_value(self) -> None:
        result = build()("text")
        assert hash(result) == hash(Intent.ANALYZE)
        assert {Intent.ANALYZE: "handled"}[result] == "handled"

    def test_works_with_set_membership(self) -> None:
        assert build()("text") in {Intent.ANALYZE, Intent.SEARCH}

    def test_works_as_a_match_pattern(self) -> None:
        match build()("text"):
            case Intent.ANALYZE:
                matched = True
            case _:
                matched = False
        assert matched

    def test_two_results_compare_on_value_alone(self) -> None:
        confident = build(confidence=0.99)("text")
        unsure = build(confidence=0.34)("text")
        assert confident == unsure


class TestThreshold:
    def test_no_threshold_means_no_policy(self) -> None:
        result = build(confidence=0.10)("text")
        assert result.threshold is None
        assert result.meets_threshold is True
        assert result == Intent.ANALYZE

    def test_confidence_at_the_threshold_passes(self) -> None:
        result = build(confidence=0.8, threshold=0.8)("text")
        assert result.meets_threshold is True
        assert result == Intent.ANALYZE

    def test_comparison_below_the_threshold_raises(self) -> None:
        result = build(confidence=0.42, threshold=0.8)("text")
        with pytest.raises(LowConfidenceError, match="below the configured threshold"):
            _ = result == Intent.ANALYZE

    def test_inequality_below_the_threshold_also_raises(self) -> None:
        result = build(confidence=0.42, threshold=0.8)("text")
        with pytest.raises(LowConfidenceError):
            _ = result != Intent.SEARCH

    def test_membership_below_the_threshold_raises(self) -> None:
        result = build(confidence=0.42, threshold=0.8)("text")
        with pytest.raises(LowConfidenceError):
            _ = result in {Intent.ANALYZE}

    def test_the_error_carries_the_result(self) -> None:
        result = build(confidence=0.42, threshold=0.8)("text")
        with pytest.raises(LowConfidenceError) as raised:
            _ = result == Intent.ANALYZE
        assert raised.value.result is result

    def test_reading_the_answer_never_raises(self) -> None:
        result = build(confidence=0.42, threshold=0.8)("text")
        assert result.value is Intent.ANALYZE
        assert result.confidence == 0.42
        assert result.meets_threshold is False
        assert result.probabilities[Intent.ANALYZE] == 0.91
        assert result.ranked[0][0] is Intent.ANALYZE
        assert result.margin == pytest.approx(0.85)
        assert repr(result)

    def test_comparing_two_low_confidence_results_is_allowed(self) -> None:
        first = build(confidence=0.42, threshold=0.8)("text")
        second = build(confidence=0.42, threshold=0.8)("text")
        assert first == second

    def test_comparison_with_an_unrelated_type_is_not_guarded(self) -> None:
        result = build(confidence=0.42, threshold=0.8)("text")
        assert result != "analyze"
