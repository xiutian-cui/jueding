"""The TypeSafe adapter, exercised against a real client over a mock transport.

These tests drive the genuine `TypeSafeClient` — including its request building and
response parsing — without a network connection or an API key.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from conftest import Intent
from jueding import (
    BackendResponseError,
    BooleanQuestion,
    ChoiceQuestion,
    Condition,
    ConfigurationError,
    Decision,
    Score,
    ScoreQuestion,
    Usage,
)
from jueding.typesafe import TypeSafeBackend, _from_answer, _to_question
from typesafe_sdk import TypeSafeClient

PLACEHOLDER_KEY = "not-a-real-key"

ANSWERS: dict[str, Any] = {
    "IntentDecision": {
        "type": "choice",
        "choice": "analyze",
        "confidence": 0.91,
        "probabilities": {"search": 0.06, "analyze": 0.91, "chat": 0.03},
    },
    "Condition": {"type": "noul", "noul": 0.98},
    "Score": {
        "type": "score",
        "score": 1.3,
        "confidence": 0.54,
        "legend": {"0": "Cosmetic", "1": "Workaround exists", "2": "Blocking"},
        "probabilities": {"0": 0.0, "1": 0.7, "2": 0.3},
    },
}


@pytest.fixture
def captured() -> dict[str, Any]:
    return {}


@pytest.fixture
def backend(captured: dict[str, Any]) -> TypeSafeBackend:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        asked = captured["body"]["questions"]
        return httpx2.Response(
            200,
            json={
                "model": "jev-1",
                "usage": {"input_tokens": 120, "output_tokens": 12},
                "answers": {name: ANSWERS[name] for name in asked},
            },
            headers={"x-typesafe-request-id": "req_123"},
        )

    client = TypeSafeClient(api_key=PLACEHOLDER_KEY, transport=httpx2.MockTransport(handler))
    with client:
        yield TypeSafeBackend(client=client)


class TestDecision:
    def test_round_trip(self, backend: TypeSafeBackend, captured: dict[str, Any]) -> None:
        decision = Decision(instructions="Determine intent.", choices=Intent,
            backend=backend,
            criteria={Intent.SEARCH: "Retrieve information."},
        )
        result = decision(user_input="compare these approaches")

        assert result == Intent.ANALYZE
        assert result.confidence == 0.91
        assert result.probabilities[Intent.SEARCH] == 0.06
        assert result.model == "jev-1"
        assert result.usage == Usage(input_tokens=120, output_tokens=12)
        assert result.request_id == "req_123"

    def test_request_body(self, backend: TypeSafeBackend, captured: dict[str, Any]) -> None:
        Decision(instructions="Determine intent.", choices=Intent,
            backend=backend,
            criteria={Intent.SEARCH: "Retrieve information."},
        )(user_input="compare these approaches")

        assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
        assert captured["body"]["state"] == {"user_input": "compare these approaches"}
        assert captured["body"]["model"] == "jev-latest"
        assert captured["body"]["questions"] == {
            "IntentDecision": {
                "type": "choice",
                "instructions": "Determine intent.",
                "criteria": {
                    "search": "Retrieve information.",
                    "analyze": None,
                    "chat": None,
                },
            }
        }


class TestCondition:
    def test_round_trip(self, backend: TypeSafeBackend, captured: dict[str, Any]) -> None:
        result = Condition(
            instructions="Is this urgent?",
            backend=backend,
            true="Has a deadline.",
            false="No deadline.",
        )("Please fix before payroll closes.")

        assert bool(result) is True
        assert result.probabilities[True] == 0.98
        assert captured["body"]["questions"]["Condition"] == {
            "type": "noul",
            "instructions": "Is this urgent?",
            "criteria": {"true": "Has a deadline.", "false": "No deadline."},
        }

    def test_criteria_are_omitted_when_unset(
        self, backend: TypeSafeBackend, captured: dict[str, Any]
    ) -> None:
        Condition(instructions="Is this urgent?", backend=backend)("ticket")
        assert captured["body"]["questions"]["Condition"] == {
            "type": "noul",
            "instructions": "Is this urgent?",
        }


class TestScore:
    def test_round_trip(self, backend: TypeSafeBackend, captured: dict[str, Any]) -> None:
        result = Score(
            instructions="How severe is this?",
            rubric=["Cosmetic", "Workaround exists", "Blocking"],
            backend=backend,
        )("The export button crashes Safari.")

        assert result == 1.3
        assert result.probabilities == {0: 0.0, 1: 0.7, 2: 0.3}
        assert result.legend[2] == "Blocking"
        assert captured["body"]["questions"]["Score"]["criteria"] == [
            "Cosmetic",
            "Workaround exists",
            "Blocking",
        ]


class TestBackendOptions:
    def test_backend_level_model_is_sent(self, captured: dict[str, Any]) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            captured["body"] = json.loads(request.content)
            return httpx2.Response(
                200,
                json={
                    "model": "jev-1",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "answers": {"IntentDecision": ANSWERS["IntentDecision"]},
                },
                headers={"x-typesafe-request-id": "req_123"},
            )

        with TypeSafeClient(
            api_key=PLACEHOLDER_KEY, transport=httpx2.MockTransport(handler)
        ) as client:
            backend = TypeSafeBackend(client=client, model="jev-pinned")
            Decision(instructions="Determine intent.", choices=Intent, backend=backend)("text")
        assert captured["body"]["model"] == "jev-pinned"

    def test_missing_usage_is_tolerated(self) -> None:
        class StubResponse:
            model = "jev-1"
            usage = None
            request_id = "req_123"
            answers: dict[str, Any] = {}

        class StubClient:
            def system_one(self, state: Any, questions: Any, *, model: Any = None) -> StubResponse:
                return StubResponse()

        backend = TypeSafeBackend(client=StubClient())  # type: ignore[arg-type]
        response = backend.ask(state="state", questions={})
        assert response.usage is None
        assert response.model == "jev-1"


class TestTranslationErrors:
    def test_provider_errors_are_not_wrapped(self) -> None:
        from typesafe_sdk import TypeSafeAuthenticationError

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(401, json={"error": "bad key"})

        with TypeSafeClient(
            api_key=PLACEHOLDER_KEY, transport=httpx2.MockTransport(handler)
        ) as client:
            decision = Decision(
                instructions="Determine intent.",
                choices=Intent,
                backend=TypeSafeBackend(client=client),
            )
            with pytest.raises(TypeSafeAuthenticationError) as raised:
                decision("text")
        assert raised.value.status == 401

    def test_a_response_without_a_request_id_raises(self) -> None:
        """Jueding follows the SDK in treating a request id as always present.

        A response missing the header did not reach us from the API unaltered, so the
        adapter surfaces the SDK's error rather than inventing a `None`.
        """
        from typesafe_sdk import TypeSafeError

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                200,
                json={
                    "model": "jev-1",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "answers": {"IntentDecision": ANSWERS["IntentDecision"]},
                },
            )

        with TypeSafeClient(
            api_key=PLACEHOLDER_KEY, transport=httpx2.MockTransport(handler)
        ) as client:
            decision = Decision(
                instructions="Determine intent.",
                choices=Intent,
                backend=TypeSafeBackend(client=client),
            )
            with pytest.raises(TypeSafeError, match="did not include a request ID"):
                decision("text")

    def test_unsupported_question_type_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="unsupported question type"):
            _to_question("x", object())  # type: ignore[arg-type]

    def test_unsupported_answer_type_is_rejected(self) -> None:
        with pytest.raises(BackendResponseError, match="unsupported answer type"):
            _from_answer("x", object())

    @pytest.mark.parametrize(
        "question",
        [
            ChoiceQuestion(instructions="Pick.", criteria={"a": None}),
            BooleanQuestion(instructions="Yes?"),
            ScoreQuestion(instructions="Rate.", criteria=["low", "high"]),
        ],
    )
    def test_every_question_type_translates(self, question: Any) -> None:
        assert _to_question("x", question) is not None
