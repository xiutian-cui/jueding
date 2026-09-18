"""Turning call arguments into the state sent to Jev."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from jueding.models import JSONContent
from jueding.errors import StateError


def normalize_state(name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> JSONContent:
    """Resolve a call's arguments into a single piece of state for Jev.

    One positional argument is used as the state directly. Keyword arguments become a
    dictionary whose keys are the state's field names — and those names are read by the
    model, so they are part of the question, not plumbing.
    """
    if args and kwargs:
        raise StateError(
            f"{name}: pass state either positionally or as keywords, not both."
        )
    if len(args) > 1:
        raise StateError(f"{name}: expected one state argument, got {len(args)}.")
    if kwargs:
        return dict(kwargs)
    if not args:
        raise StateError(f"{name}: no state was supplied.")

    state = args[0]
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        return dict(state)
    if is_dataclass(state) and not isinstance(state, type):
        return asdict(state)
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray)):
        return list(state)
    raise StateError(
        f"{name}: state must be a string, mapping, sequence or dataclass instance; "
        f"got {type(state).__name__}."
    )
