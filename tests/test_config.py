"""Unit tests for config loading and the Config dataclass."""

from __future__ import annotations

import json

import pytest

from prompter import config as config_mod
from prompter.config import ApprovalMode, Config


def test_defaults():
    cfg = Config()
    assert cfg.provider == "anthropic"
    assert cfg.model == ""
    assert cfg.resolved_model == "claude-sonnet-4-6"
    assert cfg.key_env == "ANTHROPIC_API_KEY"
    assert cfg.base_url is None
    assert cfg.max_fix_attempts == 3
    assert cfg.concise is False
    assert cfg.show_usage is False
    assert cfg.pricing == {}


def test_concise_round_trips():
    restored = Config.from_dict(Config(concise=True).to_dict())
    assert restored.concise is True


def test_usage_and_pricing_round_trip():
    original = Config(show_usage=True, pricing={"m": {"input": 3.0, "output": 15.0}})
    restored = Config.from_dict(original.to_dict())
    assert restored.show_usage is True
    assert restored.pricing == {"m": {"input": 3.0, "output": 15.0}}


@pytest.mark.parametrize("provider, model", [
    ("anthropic", "claude-sonnet-4-6"),
    ("openai", "gpt-5.4"),
    ("gemini", "gemini-3.5-flash"),
])
def test_resolved_model_per_provider(provider, model):
    assert Config(provider=provider).resolved_model == model


def test_explicit_model_overrides_default():
    assert Config(provider="openai", model="gpt-4o").resolved_model == "gpt-4o"


def test_key_env_override():
    assert Config(api_key_env="MY_KEY").key_env == "MY_KEY"


@pytest.mark.parametrize("given, expected", [
    ("anthropic", "anthropic"),
    ("Gemini", "gemini"),
    ("OPENAI", "openai"),
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("chatgpt", "openai"),
    ("google", "gemini"),
    ("  Claude  ", "anthropic"),
])
def test_normalize_provider(given, expected):
    assert config_mod.normalize_provider(given) == expected


def test_workspace_path_expands():
    assert Config().workspace_path.endswith("/Code")
    assert "~" not in Config().workspace_path


def test_from_dict_ignores_unknown_keys():
    cfg = Config.from_dict({"provider": "openai", "bogus": 123})
    assert cfg.provider == "openai"
    assert not hasattr(cfg, "bogus")


def test_round_trip():
    cfg = Config(provider="gemini", base_url="https://x", max_fix_attempts=5)
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_approval_mode_values():
    assert ApprovalMode.SMART.value == "smart"
    assert ApprovalMode.ASK_ALL.value == "ask-all"
    assert ApprovalMode.YOLO.value == "yolo"


def test_load_writes_default_on_first_run(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(path))
    cfg = config_mod.load_config()
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["provider"] == "anthropic"
    assert data["model"] == ""
    assert cfg == Config()


def test_load_reads_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"provider": "openai", "model": "gpt-4o"}))
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(path))
    cfg = config_mod.load_config()
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"


def test_load_bad_file_falls_back(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text("{not valid json")
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(path))
    cfg = config_mod.load_config()
    assert cfg == Config()
    assert "couldn't read" in capsys.readouterr().err
