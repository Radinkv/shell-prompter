"""Unit tests for the command dispatcher, run setup, and the subcommands."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prompter import cli
from prompter.config import ApprovalMode, Config
from prompter.providers.base import ProviderAuthError, ProviderError
from prompter.ui import Console


def parse(argv):
    return cli._run_parser().parse_args(argv)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point config and keys storage at temp files."""
    from prompter import config as config_mod
    from prompter import keys as keys_mod
    config_path = str(tmp_path / "config.json")
    keys_path = str(tmp_path / "keys.json")
    monkeypatch.setattr(config_mod, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)
    monkeypatch.setattr(keys_mod, "KEYS_PATH", keys_path)
    monkeypatch.setattr(cli, "KEYS_PATH", keys_path)
    return tmp_path


def _raise_provider_error(_config):
    raise ProviderError("no key")


def test_resolves_default_model():
    cfg = Config()
    cli._apply_overrides(parse([]), cfg)
    assert cfg.model == "claude-sonnet-4-6"


def test_switching_provider_drops_other_providers_pinned_model():
    cfg = Config(provider="anthropic", model="claude-sonnet-4-6")
    cli._apply_overrides(parse(["--provider", "gemini"]), cfg)
    assert cfg.model == "gemini-3.5-flash"


def test_pinned_model_kept_when_provider_matches():
    cfg = Config(provider="gemini", model="gemini-1.5-pro")
    cli._apply_overrides(parse([]), cfg)
    assert cfg.model == "gemini-1.5-pro"


def test_model_flag_wins():
    cfg = Config(provider="anthropic", model="claude-sonnet-4-6")
    cli._apply_overrides(parse(["--provider", "gemini", "--model", "g-x"]), cfg)
    assert cfg.model == "g-x"


def test_concise_flag_enables():
    cfg = Config()
    cli._apply_overrides(parse(["--concise"]), cfg)
    assert cfg.concise is True


def test_verbose_flag_overrides_config_concise():
    cfg = Config(concise=True)
    cli._apply_overrides(parse(["--verbose"]), cfg)
    assert cfg.concise is False


@pytest.mark.parametrize("given, expected", [
    ("Gemini", "gemini"), ("OPENAI", "openai"), ("claude", "anthropic"),
    ("gpt", "openai"), ("google", "gemini"),
])
def test_normalizes_provider(given, expected):
    cfg = Config()
    cli._apply_overrides(parse(["--provider", given]), cfg)
    assert cfg.provider == expected


@pytest.mark.parametrize("argv, mode", [
    ([], ApprovalMode.SMART),
    (["--ask-all"], ApprovalMode.ASK_ALL),
    (["--yolo"], ApprovalMode.YOLO),
])
def test_resolve_mode(argv, mode):
    assert cli._resolve_mode(parse(argv), Config()) is mode


@pytest.mark.parametrize("value", [
    "gemini", "openai", "anthropic", "claude", "gpt", "google", "Gemini"])
def test_model_that_is_a_provider_rejected(value):
    with pytest.raises(cli._CommandError):
        cli._reject_provider_as_model(parse(["--model", value]))


@pytest.mark.parametrize("argv", [[], ["--model", "gemini-3.5-flash"]])
def test_valid_model_passes(argv):
    cli._reject_provider_as_model(parse(argv))


def test_validate_rejects_bad_max_fix():
    with pytest.raises(cli._CommandError):
        cli._validate_config(Config(max_fix_attempts=0), MagicMock(spec=Console))


def test_validate_warns_base_url_for_non_openai():
    console = MagicMock(spec=Console)
    cli._validate_config(Config(base_url="x", provider="gemini"), console)
    console.note.assert_called_once()


def test_validate_no_warn_base_url_for_openai():
    console = MagicMock(spec=Console)
    cli._validate_config(Config(base_url="x", provider="openai"), console)
    console.note.assert_not_called()


def test_create_unknown_provider_raises():
    with pytest.raises(cli._CommandError):
        cli._create_provider_checked(Config(provider="bogus"))


def test_create_missing_key_raises(monkeypatch):
    monkeypatch.setattr(cli, "create_provider", _raise_provider_error)
    with pytest.raises(cli._CommandError):
        cli._create_provider_checked(Config(provider="openai"))


def test_create_success(monkeypatch):
    monkeypatch.setattr(cli, "create_provider", lambda c: "PROVIDER")
    assert cli._create_provider_checked(Config(provider="openai")) == "PROVIDER"


def test_run_agent_ok(monkeypatch):
    monkeypatch.setattr(cli, "_dispatch", lambda a, g: None)
    assert cli._run_agent(MagicMock(), Config(), MagicMock(spec=Console), "x") \
        == cli.OK_EXIT_CODE


def test_run_agent_interrupt(monkeypatch):
    def boom(a, g):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "_dispatch", boom)
    console = MagicMock(spec=Console)
    assert cli._run_agent(MagicMock(), Config(), console, "x") == cli.SIGINT_EXIT_CODE
    console.stopped.assert_called_once()


def test_run_agent_auth_shows_missing_key(monkeypatch):
    def boom(a, g):
        raise ProviderAuthError("no key")
    monkeypatch.setattr(cli, "_dispatch", boom)
    console = MagicMock(spec=Console)
    assert cli._run_agent(MagicMock(), Config(provider="anthropic"), console, "x") \
        == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_run_agent_model_404_shows_problem(monkeypatch):
    def boom(a, g):
        raise ProviderError("models/x is not found for API version v1beta")
    monkeypatch.setattr(cli, "_dispatch", boom)
    console = MagicMock(spec=Console)
    assert cli._run_agent(MagicMock(), Config(provider="gemini", model="x"),
                          console, "x") == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_cmd_run_renders_command_error(monkeypatch):
    def boom(args, console):
        raise cli._CommandError("boom", [("x", "y")])
    monkeypatch.setattr(cli, "_prepare", boom)
    console = MagicMock(spec=Console)
    assert cli.cmd_run(["hi"], console) == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_keys_add_stores(isolated):
    from prompter import keys
    console = MagicMock(spec=Console)
    assert cli.cmd_keys(["add", "Gemini", "AIza-x"], console) == cli.OK_EXIT_CODE
    assert keys.stored_key("gemini") == "AIza-x"
    console.success.assert_called_once()


def test_keys_add_unknown_provider(isolated):
    console = MagicMock(spec=Console)
    assert cli.cmd_keys(["add", "bogus", "k"], console) == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_keys_add_usage(isolated):
    console = MagicMock(spec=Console)
    assert cli.cmd_keys(["add", "gemini"], console) == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_keys_list(isolated):
    console = MagicMock(spec=Console)
    assert cli.cmd_keys(["list"], console) == cli.OK_EXIT_CODE
    assert console.key_status.call_count == 3


def test_keys_remove(isolated):
    from prompter import keys
    keys.set_key("gemini", "g")
    console = MagicMock(spec=Console)
    assert cli.cmd_keys(["remove", "gemini"], console) == cli.OK_EXIT_CODE
    assert keys.stored_key("gemini") is None


def test_keys_unknown_action(isolated):
    console = MagicMock(spec=Console)
    assert cli.cmd_keys(["wat"], console) == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_use_sets_default(isolated):
    from prompter import config as config_mod
    console = MagicMock(spec=Console)
    assert cli.cmd_use(["gemini"], console) == cli.OK_EXIT_CODE
    saved = config_mod.load_config()
    assert saved.provider == "gemini"
    assert saved.model == ""
    assert saved.resolved_model == "gemini-3.5-flash"
    console.success.assert_called_once()


def test_use_with_model(isolated):
    from prompter import config as config_mod
    cli.cmd_use(["gemini", "gemini-1.5-pro"], MagicMock(spec=Console))
    assert config_mod.load_config().model == "gemini-1.5-pro"


def test_use_unknown_provider(isolated):
    console = MagicMock(spec=Console)
    assert cli.cmd_use(["bogus"], console) == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_use_usage(isolated):
    console = MagicMock(spec=Console)
    assert cli.cmd_use([], console) == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_status(isolated):
    console = MagicMock(spec=Console)
    assert cli.cmd_status([], console) == cli.OK_EXIT_CODE
    assert console.field.call_count == 5
    assert console.key_status.call_count == 3


def test_config_prints_path(isolated):
    console = MagicMock(spec=Console)
    assert cli.cmd_config([], console) == cli.OK_EXIT_CODE
    console.info.assert_called_once_with(cli.CONFIG_PATH)


def test_help():
    console = MagicMock(spec=Console)
    assert cli.cmd_help(console) == cli.OK_EXIT_CODE
    console.info.assert_called_once()
    assert "prompter keys add" in console.info.call_args[0][0]


@pytest.mark.parametrize("shell", ["zsh", "bash", "ZSH"])
def test_completions_emits_script(shell):
    console = MagicMock(spec=Console)
    assert cli.cmd_completions([shell], console) == cli.OK_EXIT_CODE
    script = console.info.call_args[0][0]
    assert "_prompter" in script


def test_completions_usage_without_shell():
    console = MagicMock(spec=Console)
    assert cli.cmd_completions([], console) == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_completions_unknown_shell():
    console = MagicMock(spec=Console)
    assert cli.cmd_completions(["fish"], console) == cli.ERROR_EXIT_CODE
    console.problem.assert_called_once()


def test_main_routes_to_completions(capsys):
    assert cli.main(["completions", "zsh"]) == cli.OK_EXIT_CODE
    assert "#compdef prompter" in capsys.readouterr().out


def test_version_reports_installed_version():
    from prompter.config import program_version
    console = MagicMock(spec=Console)
    assert cli.cmd_version(console) == cli.OK_EXIT_CODE
    printed = console.info.call_args[0][0]
    assert printed == f"prompter {program_version()}"


@pytest.mark.parametrize("argv", [["version"], ["--version"], ["-V"]])
def test_main_version_forms(argv, capsys):
    from prompter.config import program_version
    assert cli.main(argv) == cli.OK_EXIT_CODE
    assert program_version() in capsys.readouterr().out


def test_version_warns_when_source_ahead_of_install(monkeypatch):
    monkeypatch.setattr(cli, "program_version", lambda: "0.2.0")
    monkeypatch.setattr(cli, "source_version", lambda: "0.3.0")
    console = MagicMock(spec=Console)
    cli.cmd_version(console)
    console.note.assert_called_once()
    assert "0.3.0" in console.note.call_args[0][0]


def test_version_quiet_when_in_sync(monkeypatch):
    monkeypatch.setattr(cli, "program_version", lambda: "0.2.0")
    monkeypatch.setattr(cli, "source_version", lambda: "0.2.0")
    console = MagicMock(spec=Console)
    cli.cmd_version(console)
    console.note.assert_not_called()


def test_version_quiet_for_installed_wheel(monkeypatch):
    """No pyproject beside the package (wheel install) -> never warns."""
    monkeypatch.setattr(cli, "source_version", lambda: None)
    console = MagicMock(spec=Console)
    cli.cmd_version(console)
    console.note.assert_not_called()


def test_main_routes_to_keys(isolated):
    assert cli.main(["keys", "list"]) == cli.OK_EXIT_CODE


def test_main_help_flag(capsys):
    assert cli.main(["--help"]) == cli.OK_EXIT_CODE
    assert "Manage:" in capsys.readouterr().out


def test_main_config(isolated, capsys):
    assert cli.main(["config"]) == cli.OK_EXIT_CODE
    assert str(isolated) in capsys.readouterr().out


@pytest.mark.parametrize("argv, expected", [
    (["do", "thing"], ["do", "thing"]),
    (["run", "do"], ["do"]),
    ([], []),
    (["xyz"], ["xyz"]),
])
def test_main_run_dispatch(monkeypatch, argv, expected):
    seen = {}

    def fake_run(rest, console):
        seen["argv"] = rest
        return 0

    monkeypatch.setattr(cli, "cmd_run", fake_run)
    assert cli.main(argv) == 0
    assert seen["argv"] == expected
