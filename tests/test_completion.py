"""Unit tests for shell completion script generation."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from prompter import cli, completion


@pytest.fixture
def surface():
    return cli.command_surface()


@pytest.mark.parametrize("shell", completion.SUPPORTED_SHELLS)
def test_render_includes_commands_flags_and_providers(shell, surface):
    script = completion.render(shell, surface)
    for token in surface.commands + surface.run_flags + surface.providers:
        assert token in script


def test_surface_is_derived_from_cli(monkeypatch):
    """The surface reads the live parser and registry, not a hand-kept copy."""
    monkeypatch.setattr(cli, "known_providers", lambda: ["acme"])
    surface = cli.command_surface()
    assert surface.providers == ("acme",)
    # Flags come straight from the run parser.
    assert "--provider" in surface.value_flags
    assert "--yolo" in surface.bool_flags


def test_freeform_flags_exclude_provider_and_dir(surface):
    assert "--provider" not in surface.freeform_flags  # completes to a provider
    assert "--workspace" not in surface.freeform_flags  # completes to a directory
    assert "--model" in surface.freeform_flags          # free-form value


def test_zsh_and_bash_differ(surface):
    assert completion.render(completion.ZSH, surface) != \
        completion.render(completion.BASH, surface)
    assert completion.render(completion.ZSH, surface).startswith("#compdef")


@pytest.mark.parametrize("shell", completion.SUPPORTED_SHELLS)
def test_generated_script_parses(shell, surface):
    """The emitted script must be valid syntax for its target shell."""
    interpreter = shutil.which(shell)
    if interpreter is None:
        pytest.skip(f"{shell} not installed")
    result = subprocess.run(
        [interpreter, "-n"], input=completion.render(shell, surface),
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
