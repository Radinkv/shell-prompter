"""User config (``~/.prompter/config.json``): the Config dataclass, app
defaults, approval modes, and provider identifiers.

So you can say ``prompter "make a project called hunchday and run claude"`` and
it creates ``~/Code/hunchday`` instead of dumping it in the current directory.
Switch backends by setting ``provider`` (and a ``model`` for it); leave ``model``
empty to use that provider's default. ``base_url`` points the OpenAI adapter at
a compatible endpoint (Groq, OpenRouter); ``api_key_env`` overrides which
environment variable holds the key.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from enum import Enum

from .colors import palette

PROGRAM_NAME = "prompter"

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
DEFAULT_PROVIDER = PROVIDER_ANTHROPIC

DEFAULT_MODELS = {
    PROVIDER_ANTHROPIC: "claude-sonnet-4-6",
    PROVIDER_OPENAI: "gpt-5.4",
    PROVIDER_GEMINI: "gemini-3.5-flash",
}
DEFAULT_API_KEY_ENVS = {
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_GEMINI: "GEMINI_API_KEY",
}

DEFAULT_WORKSPACE = "~/Code"
DEFAULT_MAX_FIX_ATTEMPTS = 3
DEFAULT_PREFERENCES = [
    "When compiling C++, prefer clang++ with -std=c++17, "
    "and fall back to g++ if clang++ isn't available.",
]

CONFIG_PATH = os.path.expanduser("~/.prompter/config.json")
_JSON_INDENT = 2
_READ_FAILURE_WARNING = "Warning: couldn't read {path} ({error}); using defaults."


class ApprovalMode(Enum):
    """How freely commands run without an explicit confirmation."""

    SMART = "smart"
    ASK_ALL = "ask-all"
    YOLO = "yolo"


def default_model_for(provider: str) -> str:
    return DEFAULT_MODELS.get(provider, "")


def default_api_key_env(provider: str) -> str | None:
    return DEFAULT_API_KEY_ENVS.get(provider)


@dataclass
class Config:
    default_workspace: str = DEFAULT_WORKSPACE
    provider: str = DEFAULT_PROVIDER
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    max_fix_attempts: int = DEFAULT_MAX_FIX_ATTEMPTS
    auto_approve_safe: bool = True
    preferences: list[str] = field(
        default_factory=lambda: list(DEFAULT_PREFERENCES)
    )

    @property
    def workspace_path(self) -> str:
        return os.path.expanduser(self.default_workspace)

    @property
    def resolved_model(self) -> str:
        return self.model or default_model_for(self.provider)

    @property
    def key_env(self) -> str | None:
        return self.api_key_env or default_api_key_env(self.provider)

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
        warning = _READ_FAILURE_WARNING.format(path=CONFIG_PATH, error=e)
        print(f"{palette.YELLOW}{warning}{palette.RESET}", file=sys.stderr)
        return Config()


def _write_default_config() -> Config:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    config = Config()
    with open(CONFIG_PATH, "w") as f:
        json.dump(config.to_dict(), f, indent=_JSON_INDENT)
    return config
