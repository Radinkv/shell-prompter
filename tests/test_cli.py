"""Unit tests for argument parsing, override resolution, and error dispatch."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from prompter import cli
from prompter.agent import QuitRequested
from prompter.config import ApprovalMode, Config
from prompter.providers.base import ProviderAuthError, ProviderError
from prompter.ui import Console


def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.prompt == []
    assert args.provider is None
    assert args.model is None
    assert args.yolo is False


def test_parser_collects_prompt_and_flags():
    args = cli.build_parser().parse_args(["make", "a", "thing", "--yolo"])
    assert args.prompt == ["make", "a", "thing"]
    assert args.yolo is True


def test_apply_overrides_resolves_default_model():
    args = cli.build_parser().parse_args([])
    cfg = Config()
    cli._apply_overrides(args, cfg)
    assert cfg.model == "claude-sonnet-4-6"


def test_apply_overrides_provider_changes_default_model():
    args = cli.build_parser().parse_args(["--provider", "openai"])
    cfg = Config()
    cli._apply_overrides(args, cfg)
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-5"


def test_apply_overrides_model_flag_wins():
    args = cli.build_parser().parse_args(["--provider", "gemini", "--model", "x"])
    cfg = Config()
    cli._apply_overrides(args, cfg)
    assert cfg.model == "x"


def test_apply_overrides_base_url_and_workspace():
    args = cli.build_parser().parse_args(
        ["--base-url", "https://groq", "--workspace", "~/P", "--max-fix", "7"])
    cfg = Config()
    cli._apply_overrides(args, cfg)
    assert cfg.base_url == "https://groq"
    assert cfg.default_workspace == "~/P"
    assert cfg.max_fix_attempts == 7


@pytest.mark.parametrize("argv, mode", [
    ([], ApprovalMode.SMART),
    (["--ask-all"], ApprovalMode.ASK_ALL),
    (["--yolo"], ApprovalMode.YOLO),
])
def test_resolve_mode(argv, mode):
    args = cli.build_parser().parse_args(argv)
    assert cli._resolve_mode(args, Config()) is mode


def test_resolve_mode_config_disables_auto_approve():
    args = cli.build_parser().parse_args([])
    assert cli._resolve_mode(args, Config(auto_approve_safe=False)) is ApprovalMode.ASK_ALL


# -- error-handler registry --------------------------------------------------
def test_error_registry_structure():
    assert cli._ERROR_HANDLERS[0][0] == (KeyboardInterrupt, QuitRequested)
    assert cli._ERROR_HANDLERS[1] == (ProviderAuthError, cli._handle_auth)
    assert cli._ERROR_HANDLERS[2] == (ProviderError, cli._handle_api)


def test_handlers_return_codes():
    console = MagicMock(spec=Console)
    assert cli._handle_interrupt(console, KeyboardInterrupt()) == cli.SIGINT_EXIT_CODE
    console.stopped.assert_called_once()
    assert cli._handle_auth(console, ProviderAuthError()) == cli.ERROR_EXIT_CODE
    assert cli._handle_api(console, ProviderError("x")) == cli.ERROR_EXIT_CODE


def _stub_run(monkeypatch, raiser):
    monkeypatch.setattr(cli, "load_config", lambda: Config())
    monkeypatch.setattr(
        cli, "make_agent",
        lambda args, config, console: types.SimpleNamespace(mode=ApprovalMode.SMART),
    )
    monkeypatch.setattr(cli, "_dispatch", raiser)


def test_run_dispatches_interrupt(monkeypatch):
    def boom(agent, console, goal):
        raise KeyboardInterrupt

    _stub_run(monkeypatch, boom)
    args = cli.build_parser().parse_args([])
    assert cli.run(args, MagicMock(spec=Console)) == cli.SIGINT_EXIT_CODE


def test_run_dispatches_provider_auth(monkeypatch):
    def boom(agent, console, goal):
        raise ProviderAuthError("nope")

    _stub_run(monkeypatch, boom)
    args = cli.build_parser().parse_args([])
    assert cli.run(args, MagicMock(spec=Console)) == cli.ERROR_EXIT_CODE


def test_run_reraises_unmapped(monkeypatch):
    def boom(agent, console, goal):
        raise ValueError("unexpected")

    _stub_run(monkeypatch, boom)
    args = cli.build_parser().parse_args([])
    with pytest.raises(ValueError):
        cli.run(args, MagicMock(spec=Console))


def test_run_returns_ok(monkeypatch):
    _stub_run(monkeypatch, lambda agent, console, goal: None)
    args = cli.build_parser().parse_args([])
    assert cli.run(args, MagicMock(spec=Console)) == cli.OK_EXIT_CODE


# -- main --------------------------------------------------------------------
def test_main_config_prints_path(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: Config())
    monkeypatch.setattr(cli, "CONFIG_PATH", "/tmp/foo.json")
    assert cli.main(["--config"]) == cli.OK_EXIT_CODE
    assert "/tmp/foo.json" in capsys.readouterr().out


def test_main_dispatches_to_run(monkeypatch):
    seen = {}

    def fake_run(args, console):
        seen["ran"] = True
        return 0

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli.main(["do", "thing"]) == 0
    assert seen["ran"] is True
