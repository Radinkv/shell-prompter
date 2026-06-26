"""ANSI colour palette, auto-disabled when output isn't a TTY."""

from __future__ import annotations

import os
import sys

from .constants import EMPTY

ENV_NO_COLOR = "NO_COLOR"

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get(ENV_NO_COLOR) is None


class Palette:
    """ANSI codes exposed as attributes, blanked out when colour is off."""

    def __init__(self, enabled: bool | None = None):
        active = supports_color() if enabled is None else enabled

        def pick(code: str) -> str:
            return code if active else EMPTY

        self.RESET = pick(_RESET)
        self.BOLD = pick(_BOLD)
        self.DIM = pick(_DIM)
        self.RED = pick(_RED)
        self.GREEN = pick(_GREEN)
        self.YELLOW = pick(_YELLOW)
        self.BLUE = pick(_BLUE)
        self.MAGENTA = pick(_MAGENTA)
        self.CYAN = pick(_CYAN)


palette = Palette()
