"""Stored API keys at ~/.prompter/keys.json.

An alternative to exporting environment variables. `prompter keys set <provider>`
writes a key here with file mode 0600. At runtime an environment variable still
wins over a stored key, so CI and one-off overrides keep working.
"""

from __future__ import annotations

import json
import os

KEYS_PATH = os.path.expanduser("~/.prompter/keys.json")
_FILE_MODE = 0o600


def load_keys() -> dict:
    try:
        with open(KEYS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_keys(keys: dict) -> None:
    os.makedirs(os.path.dirname(KEYS_PATH), exist_ok=True)
    with open(KEYS_PATH, "w") as f:
        json.dump(keys, f, indent=2)
    os.chmod(KEYS_PATH, _FILE_MODE)


def stored_key(provider: str) -> str | None:
    return load_keys().get(provider)


def set_key(provider: str, key: str) -> None:
    keys = load_keys()
    keys[provider] = key
    save_keys(keys)


def clear_key(provider: str) -> bool:
    keys = load_keys()
    if provider not in keys:
        return False
    del keys[provider]
    save_keys(keys)
    return True


def resolve_api_key(config) -> str | None:
    """An environment variable (config.key_env) wins over a stored key."""
    if config.key_env:
        env_value = os.environ.get(config.key_env)
        if env_value:
            return env_value
    return stored_key(config.provider)
