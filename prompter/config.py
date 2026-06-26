"""User config (``~/.prompter/config.json``): the Config dataclass, defaults,
and approval modes.

So you can say ``prompter "make a project called hunchday and run claude"`` and
it creates ``~/Code/hunchday`` instead of dumping it in the current directory.
Edit the JSON directly to change defaults; ``preferences`` is free-form guidance
handed to the model verbatim.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from enum import Enum

from .colors import palette
from .constants import DEFAULT_MAX_FIX_ATTEMPTS, DEFAULT_MODEL

CONFIG_PATH = os.path.expanduser("~/.prompter/config.json")

DEFAULT_WORKSPACE = "~/Code"
DEFAULT_PREFERENCES = [
    "When compiling C++, prefer clang++ with -std=c++17, "
    "and fall back to g++ if clang++ isn't available.",
]


class ApprovalMode(Enum):
    """How freely commands run without an explicit confirmation."""

    SMART = "smart"
    ASK_ALL = "ask-all"
    YOLO = "yolo"


@dataclass
class Config:
    default_workspace: str = DEFAULT_WORKSPACE
    model: str = DEFAULT_MODEL
    max_fix_attempts: int = DEFAULT_MAX_FIX_ATTEMPTS
    auto_approve_safe: bool = True
    preferences: list[str] = field(
        default_factory=lambda: list(DEFAULT_PREFERENCES)
    )

    @property
    def workspace_path(self) -> str:
        return os.path.expanduser(self.default_workspace)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def load_config() -> Config:
    """Load config, writing defaults on first run. Tolerant of a bad file."""
    try:
        if not os.path.exists(CONFIG_PATH):
            return _write_default_config()
        with open(CONFIG_PATH) as f:
            return Config.from_dict(json.load(f))
    except (OSError, ValueError) as e:
        print(f"{palette.YELLOW}Warning: couldn't read {CONFIG_PATH} ({e}); "
              f"using defaults.{palette.RESET}", file=sys.stderr)
        return Config()


def _write_default_config() -> Config:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    config = Config()
    with open(CONFIG_PATH, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    return config
