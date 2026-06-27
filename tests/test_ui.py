"""Unit tests for the Console: prompts, banners, rendering, streaming."""

from __future__ import annotations

import pytest

from prompter.colors import Palette
from prompter.config import ApprovalMode, Config
from prompter.risk import RiskAssessment, RiskTier
from prompter.shell import CommandResult
from prompter.ui import Decision, _ANSWERS


@pytest.mark.parametrize("answer, decision", [
    ("y", Decision.RUN),
    ("yes", Decision.RUN),
    ("", Decision.RUN),
    ("n", Decision.SKIP),
    ("no", Decision.SKIP),
    ("a", Decision.ALL),
    ("all", Decision.ALL),
    ("q", Decision.QUIT),
    ("quit", Decision.QUIT),
])
def test_answer_map(answer, decision):
    assert _ANSWERS[answer] is decision


def _assessment(tier=RiskTier.CONFIRM, reason="installs things"):
    return RiskAssessment(tier, reason)


def test_confirm_returns_decision(console, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    assert console.confirm(_assessment(), "pip install x", "why") is Decision.RUN


def test_confirm_retries_on_bad_input(console, monkeypatch):
    answers = iter(["maybe", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert console.confirm(_assessment(), "rm x", "") is Decision.SKIP


def test_confirm_eof_is_quit(console, monkeypatch):
    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    assert console.confirm(_assessment(), "rm x", "") is Decision.QUIT


def test_banner_contents(console, capsys):
    console.banner(Config(provider="anthropic", model="claude-opus-4-8"),
                   ApprovalMode.SMART)
    out = capsys.readouterr().out
    assert "prompter" in out
    assert "mode=smart" in out
    assert "anthropic/claude-opus-4-8" in out


def test_banner_yolo_warning(console, capsys):
    console.banner(Config(), ApprovalMode.YOLO)
    assert "YOLO mode" in capsys.readouterr().out


def test_command_output_indents(console, capsys):
    console.command_output(CommandResult(0, "line one", "an error", False))
    out = capsys.readouterr().out
    assert "│ line one" in out
    assert "│ an error" in out


def test_cwd_change(console, capsys):
    console.cwd_change("/new/dir")
    assert "/new/dir" in capsys.readouterr().out


def test_force_stop_notice(console, capsys):
    console.force_stop_notice(3)
    assert "max_fix_attempts (3)" in capsys.readouterr().out


def test_streaming_brackets_text(console, capsys):
    console.begin_stream()
    console.stream_text("hello ")
    console.stream_text("world")
    console.end_stream()
    assert "hello world" in capsys.readouterr().out


def test_end_stream_without_text_is_silent(console, capsys):
    console.begin_stream()
    console.end_stream()
    assert capsys.readouterr().out == ""


def test_usage_line_with_cost(console, capsys):
    console.usage(1240, 380, 766, cost=0.004)
    out = capsys.readouterr().out
    assert "1,240 in" in out
    assert "380 out" in out
    assert "766 cached" in out
    assert "est $0.0040" in out


def test_usage_line_without_cost(console, capsys):
    console.usage(10, 5, 0)
    assert "est $" not in capsys.readouterr().out


def test_retry_notice(console, capsys):
    console.retry_notice("rate limited (429) extra detail", 1, 3, 2)
    err = capsys.readouterr().err
    assert "retrying in 2s (1/3)" in err
    assert "rate limited" in err


def test_stream_renders_markdown_when_colour_on(capsys):
    from prompter.ui import Console
    c = Console(Palette(enabled=True))
    c.begin_stream()
    c.stream_text("## Title\n")
    c.end_stream()
    out = capsys.readouterr().out
    assert Palette(enabled=True).BOLD in out
    assert "## " not in out


def test_stream_passthrough_when_markdown_disabled(capsys):
    from prompter.ui import Console
    c = Console(Palette(enabled=True), render_markdown=False)
    c.begin_stream()
    c.stream_text("## Title\n")
    c.end_stream()
    assert "## Title" in capsys.readouterr().out
