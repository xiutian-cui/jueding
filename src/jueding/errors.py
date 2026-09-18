"""Exceptions raised by Jueding itself.

Provider exceptions are never wrapped. A failure inside the TypeSafe SDK surfaces as
the SDK's own exception, so its status code, response body and request id stay intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jueding.results import Judgment

__all__ = [
    "BackendResponseError",
    "ConfigurationError",
    "JuedingError",
    "LowConfidenceError",
    "StateError",
]


class JuedingError(Exception):
    """Base class for every error raised by Jueding."""


class ConfigurationError(JuedingError):
    """A decision was constructed with arguments it cannot use.

    Raised while building the decision, not while calling it, so a misconfigured
    decision fails at import time rather than on its first production call.
    """


class StateError(JuedingError):
    """A decision was called with state it cannot turn into input for Jev."""


class BackendResponseError(JuedingError):
    """A backend returned an answer that does not match the question that was asked."""


class LowConfidenceError(JuedingError):
    """A judgment below its configured threshold was used as though it were certain.

    Raised only when a `threshold` was configured and the confidence fell below it.
    Reading `value`, `confidence`, `probabilities` or `meets_threshold` never raises;
    only comparing, truth-testing or ordering the judgment does.
    """

    def __init__(self, message: str, result: Judgment[Any]) -> None:
        super().__init__(message)
        self.result = result
        """The judgment that fell below its threshold."""
