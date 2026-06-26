"""Unit tests for shell execution, cwd tracking, and helpers."""

from __future__ import annotations

import subprocess

import pytest

from prompter import shell as shell_mod
from prompter.shell import (
    MAX_OUTPUT_CHARS,
    TIMEOUT_EXIT_CODE,
    CommandResult,
    Shell,
    looks_interactive,
    truncate,
)


@pytest.mark.parametrize("command, expected", [
    ("claude", True),
    ("vim notes.txt", True),
    ("python3", True),
    ("ssh host", True),
    ("ls", False),
    ("git status", False),
])
def test_looks_interactive(command, expected):
    assert looks_interactive(command) is expected


def test_truncate_short_text_untouched():
    assert truncate("hello") == "hello"


def test_truncate_caps_long_text():
    out = truncate("x" * (MAX_OUTPUT_CHARS * 2))
    assert len(out) < MAX_OUTPUT_CHARS * 2
    assert "characters omitted" in out


def test_command_result_failed():
    assert CommandResult(1, "", "", False).failed is True
    assert CommandResult(0, "", "", False).failed is False


def test_run_captures_output(tmp_path):
    sh = Shell(cwd=str(tmp_path))
    result = sh.run("echo hello")
    assert isinstance(result, CommandResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert result.cwd_changed is False


def test_cwd_persists_across_calls(tmp_path):
    sh = Shell(cwd=str(tmp_path))
    result = sh.run("mkdir sub && cd sub && pwd")
    assert result.cwd_changed is True
    assert sh.cwd.endswith("sub")
    again = sh.run("pwd")
    assert again.stdout.strip().endswith("sub")


def test_nonzero_exit_reported(tmp_path):
    sh = Shell(cwd=str(tmp_path))
    result = sh.run("exit 3")
    assert result.exit_code == 3
    assert result.failed is True


def test_timeout_path(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(shell_mod.subprocess, "run", boom)
    result = Shell(cwd=str(tmp_path)).run("sleep 999")
    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert "timed out" in result.stderr


def test_interactive_run_inherits_terminal(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs

        class P:
            returncode = 0
        return P()

    monkeypatch.setattr(shell_mod.subprocess, "run", fake_run)
    result = Shell(cwd=str(tmp_path)).run("claude", interactive=True)
    assert result.exit_code == 0
    assert "capture_output" not in captured["kwargs"]  # terminal inherited
