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
    assert cfg.model == "gpt-5.4"


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


@pytest.mark.parametrize("value", [
    "gemini", "openai", "anthropic", "claude", "gpt", "google", "Gemini"])
def test_model_that_is_a_provider_name_is_rejected(value):
    args = cli.build_parser().parse_args(["--model", value])
    with pytest.raises(SystemExit):
        cli._reject_provider_as_model(args)


@pytest.mark.parametrize("argv", [[], ["--model", "gemini-3.5-flash"]])
def test_valid_model_passes_validation(argv):
    cli._reject_provider_as_model(cli.build_parser().parse_args(argv))


@pytest.mark.parametrize("given, expected", [
    ("Gemini", "gemini"),
    ("OPENAI", "openai"),
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("google", "gemini"),
])
def test_apply_overrides_normalizes_provider(given, expected):
    args = cli.build_parser().parse_args(["--provider", given])
    cfg = Config()
    cli._apply_overrides(args, cfg)
    assert cfg.provider == expected


def test_validate_rejects_zero_max_fix():
    with pytest.raises(SystemExit):
        cli._validate_config(Config(max_fix_attempts=0), MagicMock(spec=Console))


def test_validate_warns_base_url_for_non_openai():
    console = MagicMock(spec=Console)
    cli._validate_config(Config(base_url="https://x", provider="gemini"), console)
    console.note.assert_called_once()


def test_validate_no_warn_base_url_for_openai():
    console = MagicMock(spec=Console)
    cli._validate_config(Config(base_url="https://x", provider="openai"), console)
    console.note.assert_not_called()


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
    assert cli._ERROR_HANDLERS[1] == (ProviderError, cli._handle_api)


def test_handlers_return_codes():
    console = MagicMock(spec=Console)
    assert cli._handle_interrupt(console, KeyboardInterrupt()) == cli.SIGINT_EXIT_CODE
    console.stopped.assert_called_once()
    assert cli._handle_auth(console, ProviderAuthError()) == cli.ERROR_EXIT_CODE
    assert cli._handle_api(console, ProviderError("x")) == cli.ERROR_EXIT_CODE


def _stub_run(monkeypatch, dispatch):
    monkeypatch.setattr(cli, "load_config", lambda: Config())
    monkeypatch.setattr(
        cli, "make_agent",
        lambda args, config, console: types.SimpleNamespace(mode=ApprovalMode.SMART),
    )
    monkeypatch.setattr(cli, "_dispatch", dispatch)


def test_run_dispatches_interrupt(monkeypatch):
    def boom(agent, goal):
        raise KeyboardInterrupt

    _stub_run(monkeypatch, boom)
    args = cli.build_parser().parse_args([])
    assert cli.run(args, MagicMock(spec=Console)) == cli.SIGINT_EXIT_CODE


def test_run_reraises_unmapped(monkeypatch):
    def boom(agent, goal):
        raise ValueError("unexpected")

    _stub_run(monkeypatch, boom)
    args = cli.build_parser().parse_args([])
    with pytest.raises(ValueError):
        cli.run(args, MagicMock(spec=Console))


def test_run_returns_ok(monkeypatch):
    _stub_run(monkeypatch, lambda agent, goal: None)
    args = cli.build_parser().parse_args([])
    assert cli.run(args, MagicMock(spec=Console)) == cli.OK_EXIT_CODE


def test_run_auth_prompt_blank_exits(monkeypatch, capsys):
    def boom(agent, goal):
        raise ProviderAuthError("no key")

    _stub_run(monkeypatch, boom)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _p="": "")
    args = cli.build_parser().parse_args([])
    assert cli.run(args, MagicMock(spec=Console)) == cli.ERROR_EXIT_CODE
    assert "keys set" in capsys.readouterr().err


def test_run_auth_rejected_key_uses_handler(monkeypatch, keys_path):
    def boom(agent, goal):
        raise ProviderAuthError("bad key")

    _stub_run(monkeypatch, boom)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _p="": "sk-wrong-0000")
    console = MagicMock(spec=Console)
    args = cli.build_parser().parse_args([])
    assert cli.run(args, console) == cli.ERROR_EXIT_CODE
    console.auth_error.assert_called_once()


def test_run_auth_retries_after_key(monkeypatch, keys_path):
    from prompter import keys as keys_mod
    calls = {"n": 0}

    def dispatch(agent, goal):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderAuthError("no key")

    _stub_run(monkeypatch, dispatch)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _p="": "sk-retry-2222")
    args = cli.build_parser().parse_args([])
    assert cli.run(args, MagicMock(spec=Console)) == cli.OK_EXIT_CODE
    assert keys_mod.stored_key("anthropic") == "sk-retry-2222"


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


# -- keys subcommand ---------------------------------------------------------
@pytest.fixture
def keys_path(tmp_path, monkeypatch):
    from prompter import keys as keys_mod
    path = tmp_path / "keys.json"
    monkeypatch.setattr(keys_mod, "KEYS_PATH", str(path))
    monkeypatch.setattr(cli, "KEYS_PATH", str(path))
    return path


def test_main_routes_keys_path(keys_path, capsys):
    assert cli.main(["keys", "path"]) == cli.OK_EXIT_CODE
    assert str(keys_path) in capsys.readouterr().out


def test_keys_set_stores_key(keys_path, monkeypatch):
    from prompter import keys as keys_mod
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt="": "sk-stored-9999")
    assert cli.main(["keys", "set", "openai"]) == cli.OK_EXIT_CODE
    assert keys_mod.stored_key("openai") == "sk-stored-9999"


def test_keys_set_rejects_unknown_provider(keys_path, monkeypatch):
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt="": "x")
    assert cli.main(["keys", "set", "bogus"]) == cli.ERROR_EXIT_CODE


def test_keys_list_masks_stored(keys_path, monkeypatch, capsys):
    from prompter import keys as keys_mod
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    keys_mod.set_key("openai", "sk-abcd1234")
    assert cli.main(["keys", "list"]) == cli.OK_EXIT_CODE
    out = capsys.readouterr().out
    assert "openai" in out
    assert "1234" in out          # masked tail shown
    assert "sk-abcd1234" not in out  # full key never printed


def test_keys_clear(keys_path):
    from prompter import keys as keys_mod
    keys_mod.set_key("gemini", "g")
    assert cli.main(["keys", "clear", "gemini"]) == cli.OK_EXIT_CODE
    assert keys_mod.stored_key("gemini") is None


def test_keys_unknown_action(keys_path):
    assert cli.main(["keys", "wat"]) == cli.ERROR_EXIT_CODE


# -- missing-key prompt flow -------------------------------------------------
def _raise_provider_error(_config):
    from prompter.providers.base import ProviderError
    raise ProviderError("missing key")


def test_provider_created_when_key_present(monkeypatch, keys_path):
    monkeypatch.setattr(cli, "create_provider", lambda c: "PROVIDER")
    assert cli._create_provider_or_prompt(Config(provider="openai")) == "PROVIDER"


def test_provider_prompts_then_retries(monkeypatch, keys_path):
    from prompter import keys as keys_mod
    calls = {"n": 0}

    def flaky_create(config):
        calls["n"] += 1
        if calls["n"] == 1:
            from prompter.providers.base import ProviderError
            raise ProviderError("missing key")
        return "PROVIDER"

    monkeypatch.setattr(cli, "create_provider", flaky_create)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _p="": "sk-entered-1111")
    assert cli._create_provider_or_prompt(Config(provider="openai")) == "PROVIDER"
    assert keys_mod.stored_key("openai") == "sk-entered-1111"


def test_provider_blank_key_exits(monkeypatch, keys_path):
    monkeypatch.setattr(cli, "create_provider", _raise_provider_error)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _p="": "")
    with pytest.raises(SystemExit):
        cli._create_provider_or_prompt(Config(provider="openai"))


def test_provider_unknown_exits_without_prompting(keys_path):
    with pytest.raises(SystemExit):
        cli._create_provider_or_prompt(Config(provider="bogus"))
