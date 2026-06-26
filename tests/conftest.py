"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prompter.colors import Palette
from prompter.ui import Console


@pytest.fixture
def palette() -> Palette:
    """A colour-disabled palette so assertions match plain text."""
    return Palette(enabled=False)


@pytest.fixture
def console(palette) -> Console:
    """A real Console with colour off (for output assertions via capsys)."""
    return Console(palette)


@pytest.fixture
def mock_console() -> MagicMock:
    """A Console double for asserting interactions in the agent loop."""
    return MagicMock(spec=Console)
