"""Unit tests for config loading and the Config dataclass."""

from __future__ import annotations

import json

from prompter import config as config_mod
from prompter.config import ApprovalMode, Config


def test_defaults():
    cfg = Config()
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.default_workspace == "~/Code"
    assert cfg.max_fix_attempts == 3
    assert cfg.auto_approve_safe is True
    assert cfg.preferences


def test_workspace_path_expands():
    assert Config().workspace_path.endswith("/Code")
    assert "~" not in Config().workspace_path


def test_from_dict_ignores_unknown_keys():
    cfg = Config.from_dict({"model": "claude-opus-4-8", "bogus": 123})
    assert cfg.model == "claude-opus-4-8"
    assert not hasattr(cfg, "bogus")


def test_round_trip():
    cfg = Config(default_workspace="~/X", max_fix_attempts=5)
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
    assert data["model"] == "claude-sonnet-4-6"
    assert cfg == Config()


def test_load_reads_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "claude-opus-4-8", "default_workspace": "~/X"}))
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(path))
    cfg = config_mod.load_config()
    assert cfg.model == "claude-opus-4-8"
    assert cfg.default_workspace == "~/X"


def test_load_bad_file_falls_back(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text("{not valid json")
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(path))
    cfg = config_mod.load_config()
    assert cfg == Config()
    assert "couldn't read" in capsys.readouterr().err
