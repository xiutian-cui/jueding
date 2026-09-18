"""How call arguments become provider state."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from conftest import FakeBackend, Intent, choice_answer
from jueding import Decision, StateError


@pytest.fixture
def decide() -> Decision[Intent]:
    backend = FakeBackend(answer=choice_answer())
    return Decision(instructions="Pick one.", choices=Intent, backend=backend)


def test_single_positional_string_is_used_directly(decide: Decision[Intent]) -> None:
    decide("compare these approaches")
    assert decide._backend.state == "compare these approaches"


def test_mapping_is_copied_not_aliased(decide: Decision[Intent]) -> None:
    original = {"message": "hello"}
    decide(original)
    original["message"] = "changed"
    assert decide._backend.state == {"message": "hello"}


def test_sequence_state_is_listed(decide: Decision[Intent]) -> None:
    decide(("first", "second"))
    assert decide._backend.state == ["first", "second"]


def test_dataclass_state_is_converted(decide: Decision[Intent]) -> None:
    @dataclass
    class Ticket:
        subject: str
        priority: int

    decide(Ticket("Duplicate charge", 2))
    assert decide._backend.state == {"subject": "Duplicate charge", "priority": 2}


def test_keywords_become_a_state_object(decide: Decision[Intent]) -> None:
    decide(user_input="compare", context={"surface": "cli"})
    assert decide._backend.state == {"user_input": "compare", "context": {"surface": "cli"}}


def test_mixing_positional_and_keyword_state_is_rejected(decide: Decision[Intent]) -> None:
    with pytest.raises(StateError, match="not both"):
        decide("text", extra="more")


def test_multiple_positional_arguments_are_rejected(decide: Decision[Intent]) -> None:
    with pytest.raises(StateError, match="expected one state argument, got 2"):
        decide("one", "two")


def test_missing_state_is_rejected(decide: Decision[Intent]) -> None:
    with pytest.raises(StateError, match="no state was supplied"):
        decide()


def test_unsupported_state_type_is_rejected(decide: Decision[Intent]) -> None:
    with pytest.raises(StateError, match="got int"):
        decide(42)


def test_bytes_are_not_treated_as_a_sequence(decide: Decision[Intent]) -> None:
    with pytest.raises(StateError, match="got bytes"):
        decide(b"raw")


def test_dataclass_type_is_not_an_instance(decide: Decision[Intent]) -> None:
    @dataclass
    class Ticket:
        subject: str

    with pytest.raises(StateError, match="got type"):
        decide(Ticket)
