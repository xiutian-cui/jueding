"""Opt-in integration test against the real Jev API.

Skipped by default and excluded from the default pytest run. To run it:

    TYPESAFE_API_KEY=... uv run pytest -m live

The API is in early access, so this will not run for anyone without a key.
"""

from __future__ import annotations

import os

import pytest

from conftest import Intent
from jueding import Decision

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("TYPESAFE_API_KEY"),
        reason="TYPESAFE_API_KEY is not set",
    ),
]


def test_a_real_decision_round_trips() -> None:
    from typesafe_sdk import TypeSafeClient

    from jueding.typesafe import TypeSafeBackend

    with TypeSafeClient() as client:
        decide = Decision(instructions="Determine the user's primary intent.", choices=Intent,
            backend=TypeSafeBackend(client=client),
            criteria={
                Intent.SEARCH: "Retrieve existing information.",
                Intent.ANALYZE: "Reason about or compare information.",
                Intent.CHAT: "Primarily conversational.",
            },
        )
        result = decide(user_input="Compare these two implementation approaches.")

    assert result.value in set(Intent)
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == set(Intent)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=0.02)
    assert result.usage is not None
