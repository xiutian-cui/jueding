"""The version is declared once and read from there by the build."""

from __future__ import annotations

import importlib.metadata
import re

import jueding


def test_version_is_the_installed_version() -> None:
    """`__init__.py` is the single source; hatchling reads the built version from it."""
    assert jueding.__version__ == importlib.metadata.version("jueding")


def test_version_is_a_release_number() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", jueding.__version__)
