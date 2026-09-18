"""Adapter for TypeSafe's Jev model, via the official synchronous `TypeSafeClient`.

The client is supplied by the caller, so credentials, connection lifetime, timeouts,
retries and test doubles stay under application control. HTTP, authentication, response
parsing and transport retries all belong to the SDK; this module only translates
questions and answers, and never wraps the SDK's exceptions.
"""

from __future__ import annotations

from collections.abc import Mapping

from typesafe_sdk import (
    Choice as _Choice,
)
from typesafe_sdk import (
    ChoiceAnswer as _ChoiceAnswer,
)
from typesafe_sdk import (
    Noul as _Noul,
)
from typesafe_sdk import (
    NoulAnswer as _NoulAnswer,
)
from typesafe_sdk import (
    Score as _Score,
)
from typesafe_sdk import (
    ScoreAnswer as _ScoreAnswer,
)
from typesafe_sdk import (
    TypeSafeClient,
)

from jueding.models import (
    Answer,
    BackendResponse,
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    JSONContent,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)
from jueding.errors import BackendResponseError, ConfigurationError

__all__ = ["TypeSafeBackend"]


class TypeSafeBackend:
    """A `Backend` backed by a caller-supplied `TypeSafeClient`.

    ```python
    from typesafe_sdk import TypeSafeClient
    from jueding.typesafe import TypeSafeBackend

    with TypeSafeClient() as client:
        backend = TypeSafeBackend(client=client)
    ```
    """

    def __init__(self, *, client: TypeSafeClient, model: str | None = None) -> None:
        """Wrap a client.

        Args:
            client: An open `TypeSafeClient`. Its lifetime belongs to the caller; this
                adapter never closes it.
            model: Default model for questions sent through this backend. `None` inherits
                the client's own default.
        """
        self.client = client
        self.model = model

    def ask(self, *, state: JSONContent, questions: Mapping[str, Question]) -> BackendResponse:
        """Send one System One request and translate the answers back.

        Follows the SDK's own contract that a response carries a request id: reading it
        raises `TypeSafeError` when the `x-typesafe-request-id` header is absent, which
        in practice means the response did not come from the API unaltered — a test
        double that omits the header, or an intermediary that stripped it.
        """
        response = self.client.system_one(
            state,
            {name: _to_question(name, question) for name, question in questions.items()},
            model=self.model,
        )
        usage = (
            Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            if response.usage is not None
            else None
        )
        return BackendResponse(
            answers={name: _from_answer(name, answer) for name, answer in response.answers.items()},
            model=response.model,
            usage=usage,
            request_id=response.request_id,
        )


def _to_question(name: str, question: Question) -> _Choice | _Noul | _Score:
    """Translate a Jueding question into its System One primitive."""
    if isinstance(question, ChoiceQuestion):
        return _Choice(instructions=question.instructions, criteria=dict(question.criteria))
    if isinstance(question, BooleanQuestion):
        criteria = (
            {"true": question.true, "false": question.false}
            if question.true is not None or question.false is not None
            else None
        )
        return _Noul(instructions=question.instructions, criteria=criteria)
    if isinstance(question, ScoreQuestion):
        return _Score(instructions=question.instructions, criteria=list(question.criteria))
    raise ConfigurationError(f"{name}: unsupported question type {type(question).__name__}.")


def _from_answer(name: str, answer: object) -> Answer:
    """Translate a System One answer into its Jueding equivalent."""
    if isinstance(answer, _ChoiceAnswer):
        return ChoiceAnswer(
            label=answer.choice,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
        )
    if isinstance(answer, _NoulAnswer):
        return BooleanAnswer(probability=answer.noul)
    if isinstance(answer, _ScoreAnswer):
        return ScoreAnswer(
            score=answer.score,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
            legend=dict(answer.legend),
        )
    raise BackendResponseError(f"{name}: unsupported answer type {type(answer).__name__}.")
