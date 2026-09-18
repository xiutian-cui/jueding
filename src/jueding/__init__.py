"""Jueding — a typed Python wrapper around TypeSafe's Jev model.

Unstructured state goes in; a typed value, a confidence and a complete probability
distribution come out. Your own code decides what happens next.

The Jev integration lives in `jueding.typesafe`. Import it only where you build a
backend, so that code which replaces Jev does not import the SDK at all.
"""

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
    Usage,
)
from jueding.errors import (
    BackendResponseError,
    ConfigurationError,
    JuedingError,
    LowConfidenceError,
    StateError,
)
from jueding.primitives import Condition, Decision, Score
from jueding.results import ConditionResult, DecisionResult, Judgment, ScoreResult

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Backend",
    "BackendResponse",
    "BackendResponseError",
    "BooleanAnswer",
    "BooleanQuestion",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "Condition",
    "ConditionResult",
    "ConfigurationError",
    "Decision",
    "DecisionResult",
    "JSONContent",
    "Judgment",
    "JuedingError",
    "LowConfidenceError",
    "Question",
    "Score",
    "ScoreAnswer",
    "ScoreQuestion",
    "ScoreResult",
    "StateError",
    "Usage",
    "__version__",
]
