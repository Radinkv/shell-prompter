"""ANSI colour palette, auto-disabled when output isn't a TTY."""

from __future__ import annotations

import os
import sys

from .constants import ENV_NO_COLOR


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get(ENV_NO_COLOR) is None


class Palette:
    """ANSI codes exposed as attributes, blanked out when colour is off."""

    def __init__(self, enabled: bool | None = None):
        active = supports_color() if enabled is None else enabled

        def pick(code: str) -> str:
            return code if active else ""

        self.RESET = pick("\033[0m")
        self.BOLD = pick("\033[1m")
        self.DIM = pick("\033[2m")
        self.RED = pick("\033[31m")
        self.GREEN = pick("\033[32m")
        self.YELLOW = pick("\033[33m")
        self.BLUE = pick("\033[34m")
        self.MAGENTA = pick("\033[35m")
        self.CYAN = pick("\033[36m")


palette = Palette()
