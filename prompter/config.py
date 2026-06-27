"""User config (``~/.prompter/config.json``): the Config dataclass, app
defaults, approval modes, and provider identifiers.

So you can say ``prompter "make a project called hunchday and run claude"`` and
it creates ``~/Code/hunchday`` instead of dumping it in the current directory.
Switch backends by setting ``provider`` (and a ``model`` for it). Leave ``model``
empty to use that provider's default. ``base_url`` points the OpenAI adapter at
a compatible endpoint (Groq, OpenRouter). ``api_key_env`` overrides which
environment variable holds the key.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from pathlib import Path

from .colors import palette
from .constants import EMPTY, FILE_WRITE_MODE, JSON_INDENT

PROGRAM_NAME = "prompter"
DISTRIBUTION_NAME = "shell-prompter"
_UNKNOWN_VERSION = "0.0.0+unknown"


def program_version() -> str:
    """The installed package version, read from distribution metadata.

    Single source of truth is pyproject.toml; the metadata carries it into the
    built package. Falls back when running from a tree that was never installed.
    """
    try:
        return _distribution_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _UNKNOWN_VERSION


_PYPROJECT_NAME = "pyproject.toml"
_PYPROJECT_VERSION_RE = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def source_version() -> str | None:
    """The version declared in a sibling pyproject.toml, or None.

    Only a source checkout has pyproject.toml next to the package; an installed
    wheel does not ship it. So a non-None result that differs from
    program_version() means the editable install's metadata is stale -- the dev
    bumped the version but has not reinstalled. End users never hit this path.
    """
    pyproject = Path(__file__).resolve().parent.parent / _PYPROJECT_NAME
    try:
        text = pyproject.read_text()
    except OSError:
        return None
    match = _PYPROJECT_VERSION_RE.search(text)
    return match.group(1) if match else None

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
_READ_FAILURE_WARNING = "Warning: couldn't read {path} ({error}). Using defaults."


class ApprovalMode(Enum):
    """How freely commands run without an explicit confirmation."""

    SMART = "smart"
    ASK_ALL = "ask-all"
    YOLO = "yolo"


PROVIDER_ALIASES = {
    "claude": PROVIDER_ANTHROPIC,
    "gpt": PROVIDER_OPENAI,
    "chatgpt": PROVIDER_OPENAI,
    "google": PROVIDER_GEMINI,
}


def normalize_provider(name: str) -> str:
    """Lowercase and de-alias a provider name (Gemini, claude, gpt, ...)."""
    key = (name or EMPTY).strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def default_model_for(provider: str) -> str:
    return DEFAULT_MODELS.get(provider, EMPTY)


def default_api_key_env(provider: str) -> str | None:
    return DEFAULT_API_KEY_ENVS.get(provider)


@dataclass
class Config:
    default_workspace: str = DEFAULT_WORKSPACE
    provider: str = DEFAULT_PROVIDER
    model: str = EMPTY
    base_url: str | None = None
    api_key_env: str | None = None
    max_fix_attempts: int = DEFAULT_MAX_FIX_ATTEMPTS
    auto_approve_safe: bool = True
    concise: bool = False
    show_usage: bool = False
    pricing: dict = field(default_factory=dict)
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
    def from_dict(cls, data: dict) -> Config:
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


def save_config(config: Config) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, FILE_WRITE_MODE) as f:
        json.dump(config.to_dict(), f, indent=JSON_INDENT)


def _write_default_config() -> Config:
    config = Config()
    save_config(config)
    return config
