"""Unit tests for the agent loop, gating, and conversation state."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prompter.agent import Agent, CommandRequest, Conversation, QuitRequested
from prompter.config import ApprovalMode, Config
from prompter.risk import RiskTier
from prompter.shell import CommandResult
from prompter.ui import Console, Decision

from _helpers import FakeShell, LoopingClaude, ScriptedClaude, final_message, \
    text_block, tool_use_block


def _agent(client, shell, console, mode=ApprovalMode.YOLO, config=None):
    return Agent(client, shell, console, config or Config(), mode)


# -- CommandRequest ----------------------------------------------------------
def test_request_infers_interactive():
    req = CommandRequest.from_tool_input({"command": "claude", "explanation": "x"})
    assert req.interactive is True


def test_request_respects_explicit_flag():
    req = CommandRequest.from_tool_input(
        {"command": "ls", "explanation": "x", "interactive": True})
    assert req.interactive is True


def test_request_defaults():
    req = CommandRequest.from_tool_input({})
    assert req.command == "" and req.explanation == "" and req.interactive is False


# -- Conversation ------------------------------------------------------------
def test_conversation_roles():
    conv = Conversation()
    conv.add_user_text("hi")
    conv.add_assistant([text_block()])
    conv.add_user_blocks([{"type": "tool_result"}])
    assert [m["role"] for m in conv.messages] == ["user", "assistant", "user"]


# -- approval policy ---------------------------------------------------------
def test_auto_approves_matrix(mock_console):
    agent = _agent(MagicMock(), FakeShell(), mock_console, ApprovalMode.SMART)
    assert agent._auto_approves(RiskTier.SAFE) is True
    assert agent._auto_approves(RiskTier.CONFIRM) is False

    agent.mode = ApprovalMode.YOLO
    assert agent._auto_approves(RiskTier.DANGER) is True

    agent.mode = ApprovalMode.ASK_ALL
    assert agent._auto_approves(RiskTier.SAFE) is False

    agent.mode = ApprovalMode.SMART
    agent._approve_all = True
    assert agent._auto_approves(RiskTier.CONFIRM) is True


# -- the loop ----------------------------------------------------------------
def test_run_turn_no_tools(mock_console):
    client = ScriptedClaude([final_message("end_turn", [text_block("done")])])
    agent = _agent(client, FakeShell(), mock_console)
    agent.run_turn("just talk")
    assert [m["role"] for m in agent.conversation.messages] == ["user", "assistant"]
    assert client.disable_tools_log == [False]


def test_run_turn_executes_tool(mock_console):
    shell = FakeShell(results=[CommandResult(0, "done", "", False)])
    client = ScriptedClaude([
        final_message("tool_use", [tool_use_block("echo hi")]),
        final_message("end_turn", [text_block("ok")]),
    ])
    agent = _agent(client, shell, mock_console)
    agent.run_turn("do it")

    assert shell.calls == [("echo hi", False)]
    assert [m["role"] for m in agent.conversation.messages] == \
        ["user", "assistant", "user", "assistant"]
    tool_result = agent.conversation.messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["is_error"] is False
    assert "exit_code: 0" in tool_result["content"]
    assert agent.consecutive_failures == 0


def test_declined_command_not_run(mock_console):
    mock_console.confirm.return_value = Decision.SKIP
    client = ScriptedClaude([
        final_message("tool_use", [tool_use_block("rm -rf x")]),
        final_message("end_turn", [text_block("ok")]),
    ])
    shell = FakeShell()
    agent = _agent(client, shell, mock_console, ApprovalMode.ASK_ALL)
    agent.run_turn("delete stuff")

    assert shell.calls == []
    tool_result = agent.conversation.messages[2]["content"][0]
    assert tool_result["is_error"] is True
    assert "declined" in tool_result["content"].lower()


def test_quit_raises(mock_console):
    mock_console.confirm.return_value = Decision.QUIT
    client = ScriptedClaude([final_message("tool_use", [tool_use_block("rm x")])])
    agent = _agent(client, FakeShell(), mock_console, ApprovalMode.ASK_ALL)
    with pytest.raises(QuitRequested):
        agent.run_turn("go")


def test_bounded_retry_stops(mock_console):
    client = LoopingClaude("false")
    shell = FakeShell(results=[CommandResult(1, "", "boom", False)] * 10)
    agent = _agent(client, shell, mock_console, ApprovalMode.YOLO,
                   Config(max_fix_attempts=2))
    agent.run_turn("impossible task")

    assert agent.consecutive_failures == 2
    assert agent._force_stop is True
    assert agent.conversation.messages[-1]["role"] == "assistant"
    mock_console.force_stop_notice.assert_called_once_with(2)
    assert client.disable_tools_log[-1] is True


def test_successful_command_resets_failure_count(mock_console):
    shell = FakeShell(results=[
        CommandResult(1, "", "err", False),
        CommandResult(0, "ok", "", False),
    ])
    client = ScriptedClaude([
        final_message("tool_use", [tool_use_block("flaky")]),
        final_message("tool_use", [tool_use_block("retry")]),
        final_message("end_turn", [text_block("fixed")]),
    ])
    agent = _agent(client, shell, mock_console)
    agent.run_turn("fix it")
    assert agent.consecutive_failures == 0


def test_interactive_skips_output_render(mock_console):
    shell = FakeShell(results=[CommandResult(0, "ignored", "", False)])
    client = ScriptedClaude([
        final_message("tool_use", [tool_use_block("claude", interactive=True)]),
        final_message("end_turn", [text_block("ok")]),
    ])
    agent = _agent(client, shell, mock_console)
    agent.run_turn("launch claude")
    assert shell.calls == [("claude", True)]
    mock_console.command_output.assert_not_called()
