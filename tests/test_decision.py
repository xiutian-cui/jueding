"""Decision: construction, the call, and the result it builds."""

from __future__ import annotations

from enum import Enum, StrEnum

import pytest

from conftest import FakeBackend, Intent, Single, choice_answer
from jueding import (
    BackendResponseError,
    ChoiceQuestion,
    ConfigurationError,
    Decision,
    DecisionResult,
    Usage,
)


def build(**kwargs) -> Decision[Intent]:
    backend = kwargs.pop("backend", None) or FakeBackend(answer=choice_answer())
    return Decision(instructions="Determine intent.", choices=Intent, backend=backend, **kwargs)


class TestConstruction:
    def test_choices_must_be_an_enum(self) -> None:
        with pytest.raises(ConfigurationError, match="must be an Enum subclass"):
            Decision(
                instructions="Pick.", choices=dict, backend=FakeBackend()
            )  # type: ignore[arg-type]

    def test_enum_must_have_members(self) -> None:
        class Empty(StrEnum):
            pass

        with pytest.raises(ConfigurationError, match="has no members"):
            Decision(instructions="Pick.", choices=Empty, backend=FakeBackend())

    def test_enum_values_must_be_strings(self) -> None:
        class Numeric(Enum):
            ONE = 1

        with pytest.raises(ConfigurationError, match="non-empty string value"):
            Decision(instructions="Pick.", choices=Numeric, backend=FakeBackend())

    def test_enum_values_must_not_be_empty(self) -> None:
        class Blank(StrEnum):
            NOTHING = ""

        with pytest.raises(ConfigurationError, match="non-empty string value"):
            Decision(instructions="Pick.", choices=Blank, backend=FakeBackend())

    def test_aliases_are_rejected(self) -> None:
        class Aliased(StrEnum):
            FIRST = "same"
            SECOND = "same"

        with pytest.raises(ConfigurationError, match=r"has aliases \['SECOND'\]"):
            Decision(instructions="Pick.", choices=Aliased, backend=FakeBackend())

    def test_too_many_choices_are_rejected(self) -> None:
        big = StrEnum("Big", {f"M{index}": f"v{index}" for index in range(256)})
        with pytest.raises(ConfigurationError, match="at most 255"):
            Decision(instructions="Pick.", choices=big, backend=FakeBackend())

    def test_instructions_must_be_a_non_empty_string(self) -> None:
        with pytest.raises(ConfigurationError, match="non-empty string"):
            Decision(instructions="   ", choices=Intent, backend=FakeBackend())

    def test_instructions_must_be_a_string(self) -> None:
        with pytest.raises(ConfigurationError, match="non-empty string"):
            Decision(
                instructions=None, choices=Intent, backend=FakeBackend()
            )  # type: ignore[arg-type]

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_threshold_must_be_within_zero_and_one(self, threshold: float) -> None:
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            build(threshold=threshold)

    @pytest.mark.parametrize("threshold", [True, "0.8"])
    def test_threshold_must_be_a_number(self, threshold: object) -> None:
        with pytest.raises(ConfigurationError, match="must be a number or None"):
            build(threshold=threshold)

    def test_criteria_accepts_members_and_their_values(self) -> None:
        decision = build(criteria={Intent.SEARCH: "Retrieve.", "chat": "Converse."})
        decision("text")
        question = decision._backend.question
        assert isinstance(question, ChoiceQuestion)
        assert question.criteria == {
            "search": "Retrieve.",
            "analyze": None,
            "chat": "Converse.",
        }

    def test_criteria_accepts_structured_descriptions(self) -> None:
        described = {"what": "Reason about information.", "examples": ["compare these"]}
        decision = build(criteria={Intent.ANALYZE: described})
        decision("text")
        assert decision._backend.question.criteria["analyze"] == described

    def test_unknown_criteria_key_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="is not a member of Intent"):
            build(criteria={"nonsense": "?"})

    def test_unknown_criteria_object_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="is not a member of Intent"):
            build(criteria={42: "?"})

    def test_name_defaults_to_the_enum_and_class_name(self) -> None:
        decision = build()
        decision("text")
        assert decision.name == "IntentDecision"
        assert decision._backend.question_name == "IntentDecision"

    def test_name_can_be_overridden(self) -> None:
        decision = build(name="intent")
        decision("text")
        assert decision._backend.question_name == "intent"


class TestCall:
    def test_returns_a_typed_result_with_metadata(self) -> None:
        decision = build()
        result = decision("text")
        assert isinstance(result, DecisionResult)
        assert result.value is Intent.ANALYZE
        assert result.confidence == 0.91
        assert result.model == "fake-1"
        assert result.usage == Usage(input_tokens=120, output_tokens=12)
        assert result.request_id == "req_fake"
        assert result.latency_seconds >= 0.0
        assert result.threshold is None

    def test_probabilities_are_keyed_by_enum_member(self) -> None:
        result = build()("text")
        assert result.probabilities == {
            Intent.SEARCH: 0.06,
            Intent.ANALYZE: 0.91,
            Intent.CHAT: 0.03,
        }

    def test_missing_labels_are_filled_with_zero(self) -> None:
        backend = FakeBackend(
            answer=choice_answer(probabilities={"search": 0.4, "analyze": 0.6})
        )
        result = build(backend=backend)("text")
        assert result.probabilities[Intent.CHAT] == 0.0

    def test_question_carries_the_instructions(self) -> None:
        decision = build()
        decision("text")
        assert decision._backend.question.instructions == "Determine intent."


class TestProviderErrors:
    def test_provider_exceptions_are_not_wrapped(self) -> None:
        failure = KeyError("upstream")
        decision = build(backend=FakeBackend(error=failure))
        with pytest.raises(KeyError) as raised:
            decision("text")
        assert raised.value is failure


class TestBadResponses:
    def test_missing_answer_is_rejected(self) -> None:
        backend = FakeBackend(answers={})
        with pytest.raises(BackendResponseError, match="no answer for this question"):
            build(backend=backend)("text")

    def test_wrong_answer_type_is_rejected(self) -> None:
        from jueding import BooleanAnswer

        backend = FakeBackend(answer=BooleanAnswer(probability=0.9))
        with pytest.raises(BackendResponseError, match="expected ChoiceAnswer, got BooleanAnswer"):
            build(backend=backend)("text")

    def test_unknown_labels_are_rejected(self) -> None:
        backend = FakeBackend(
            answer=choice_answer(probabilities={"analyze": 0.5, "nonsense": 0.5})
        )
        with pytest.raises(BackendResponseError, match=r"not members of Intent: \['nonsense'\]"):
            build(backend=backend)("text")

    def test_selected_label_must_be_a_member(self) -> None:
        backend = FakeBackend(
            answer=choice_answer(label="nonsense", probabilities={"analyze": 1.0})
        )
        with pytest.raises(BackendResponseError, match="selected label 'nonsense'"):
            build(backend=backend)("text")

    def test_distribution_must_sum_to_about_one(self) -> None:
        backend = FakeBackend(answer=choice_answer(probabilities={"analyze": 0.5, "chat": 0.1}))
        with pytest.raises(BackendResponseError, match="sum to 0.6000"):
            build(backend=backend)("text")

    def test_empty_distribution_is_rejected(self) -> None:
        backend = FakeBackend(answer=choice_answer(probabilities={}))
        with pytest.raises(BackendResponseError, match="probabilities are empty"):
            build(backend=backend)("text")

    def test_small_rounding_error_is_tolerated(self) -> None:
        backend = FakeBackend(
            answer=choice_answer(probabilities={"analyze": 0.9, "chat": 0.11})
        )
        assert build(backend=backend)("text").value is Intent.ANALYZE

    @pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), float("inf")])
    def test_probabilities_outside_zero_and_one_are_rejected(self, bad: float) -> None:
        backend = FakeBackend(answer=choice_answer(probabilities={"analyze": bad}))
        with pytest.raises(BackendResponseError, match="must be between 0 and 1"):
            build(backend=backend)("text")

    @pytest.mark.parametrize("bad", ["0.9", True, None])
    def test_non_numeric_probabilities_are_rejected(self, bad: object) -> None:
        backend = FakeBackend(answer=choice_answer(probabilities={"analyze": bad}))
        with pytest.raises(BackendResponseError, match="must be a number"):
            build(backend=backend)("text")

    def test_invalid_confidence_is_rejected(self) -> None:
        backend = FakeBackend(answer=choice_answer(confidence=1.4))
        with pytest.raises(BackendResponseError, match="confidence must be between 0 and 1"):
            build(backend=backend)("text")


class TestRanking:
    def test_ranked_orders_by_probability(self) -> None:
        result = build()("text")
        assert result.ranked == (
            (Intent.ANALYZE, 0.91),
            (Intent.SEARCH, 0.06),
            (Intent.CHAT, 0.03),
        )

    def test_ties_break_on_definition_order(self) -> None:
        backend = FakeBackend(
            answer=choice_answer(
                label="search", probabilities={"search": 0.5, "chat": 0.5, "analyze": 0.0}
            )
        )
        result = build(backend=backend)("text")
        assert [member for member, _ in result.ranked] == [
            Intent.SEARCH,
            Intent.CHAT,
            Intent.ANALYZE,
        ]

    def test_margin_is_the_gap_between_the_top_two(self) -> None:
        result = build()("text")
        assert result.margin == pytest.approx(0.85)

    def test_margin_is_zero_for_a_single_member_enum(self) -> None:
        backend = FakeBackend(answer=choice_answer(label="only", probabilities={"only": 1.0}))
        decision = Decision(instructions="Pick.", choices=Single, backend=backend)
        assert decision("text").margin == 0.0
