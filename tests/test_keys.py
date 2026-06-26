"""Unit tests for stored API keys and key resolution."""

from __future__ import annotations

import os
import stat

import pytest

from prompter import keys as keys_mod
from prompter.config import Config


@pytest.fixture
def keys_path(tmp_path, monkeypatch):
    path = tmp_path / "keys.json"
    monkeypatch.setattr(keys_mod, "KEYS_PATH", str(path))
    return path


def test_set_and_get(keys_path):
    assert keys_mod.stored_key("openai") is None
    keys_mod.set_key("openai", "sk-123")
    assert keys_mod.stored_key("openai") == "sk-123"
    assert keys_path.exists()


def test_file_is_owner_only(keys_path):
    keys_mod.set_key("anthropic", "k")
    mode = stat.S_IMODE(os.stat(keys_path).st_mode)
    assert mode == 0o600


def test_clear(keys_path):
    keys_mod.set_key("gemini", "g")
    assert keys_mod.clear_key("gemini") is True
    assert keys_mod.clear_key("gemini") is False
    assert keys_mod.stored_key("gemini") is None


def test_load_bad_file_returns_empty(keys_path):
    keys_path.write_text("{not json")
    assert keys_mod.load_keys() == {}


def test_resolve_env_wins_over_stored(keys_path, monkeypatch):
    keys_mod.set_key("anthropic", "stored-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert keys_mod.resolve_api_key(Config(provider="anthropic")) == "env-key"


def test_resolve_falls_back_to_stored(keys_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    keys_mod.set_key("anthropic", "stored-key")
    assert keys_mod.resolve_api_key(Config(provider="anthropic")) == "stored-key"


def test_resolve_none_when_absent(keys_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert keys_mod.resolve_api_key(Config(provider="openai")) is None
