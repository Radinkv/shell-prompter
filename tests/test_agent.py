"""Unit tests for the agent loop, gating, and conversation state."""

from __future__ import annotations

import pytest

from prompter.agent import Agent, CommandRequest, Conversation, QuitRequested
from prompter.config import ApprovalMode, Config
from prompter.providers.base import (
    AssistantMessage,
    ToolInvocation,
    ToolResultsMessage,
    UserMessage,
)
from prompter.risk import RiskTier
from prompter.shell import CommandResult
from prompter.ui import Decision

from _helpers import FakeProvider, FakeShell, LoopingProvider, make_call, make_turn


def _agent(provider, shell, console, mode=ApprovalMode.YOLO, config=None):
    return Agent(provider, shell, console, config or Config(), mode)


def test_request_infers_interactive():
    req = CommandRequest.from_invocation(ToolInvocation("c", "claude"))
    assert req.interactive is True


def test_request_respects_explicit_flag():
    req = CommandRequest.from_invocation(ToolInvocation("c", "ls", interactive=True))
    assert req.interactive is True


def test_request_plain_command_not_interactive():
    req = CommandRequest.from_invocation(ToolInvocation("c", "ls -la"))
    assert req.interactive is False


def test_conversation_item_types():
    conv = Conversation()
    conv.add_user("hi")
    conv.add_assistant(make_turn(text="ok"))
    conv.add_tool_results([])
    assert [type(i) for i in conv.history] == \
        [UserMessage, AssistantMessage, ToolResultsMessage]


def test_auto_approves_matrix(mock_console):
    agent = _agent(FakeProvider([]), FakeShell(), mock_console, ApprovalMode.SMART)
    assert agent._auto_approves(RiskTier.SAFE) is True
    assert agent._auto_approves(RiskTier.CONFIRM) is False

    agent.mode = ApprovalMode.YOLO
    assert agent._auto_approves(RiskTier.DANGER) is True

    agent.mode = ApprovalMode.ASK_ALL
    assert agent._auto_approves(RiskTier.SAFE) is False

    agent.mode = ApprovalMode.SMART
    agent._approve_all = True
    assert agent._auto_approves(RiskTier.CONFIRM) is True


def test_run_turn_no_tools(mock_console):
    provider = FakeProvider([make_turn(text="done")])
    agent = _agent(provider, FakeShell(), mock_console)
    agent.run_turn("just talk")
    assert [type(i) for i in agent.conversation.history] == \
        [UserMessage, AssistantMessage]
    assert provider.disable_tools_log == [False]


def test_run_turn_executes_tool(mock_console):
    shell = FakeShell(results=[CommandResult(0, "done", "", False)])
    provider = FakeProvider([
        make_turn(calls=[make_call("echo hi", call_id="c1")]),
        make_turn(text="ok"),
    ])
    agent = _agent(provider, shell, mock_console)
    agent.run_turn("do it")

    assert shell.calls == [("echo hi", False)]
    assert [type(i) for i in agent.conversation.history] == \
        [UserMessage, AssistantMessage, ToolResultsMessage, AssistantMessage]
    result = agent.conversation.history[2].results[0]
    assert result.call_id == "c1"
    assert result.is_error is False
    assert "exit_code: 0" in result.content
    assert agent.consecutive_failures == 0


def test_declined_command_not_run(mock_console):
    mock_console.confirm.return_value = Decision.SKIP
    provider = FakeProvider([
        make_turn(calls=[make_call("rm -rf x")]),
        make_turn(text="ok"),
    ])
    shell = FakeShell()
    agent = _agent(provider, shell, mock_console, ApprovalMode.ASK_ALL)
    agent.run_turn("delete stuff")

    assert shell.calls == []
    result = agent.conversation.history[2].results[0]
    assert result.is_error is True
    assert "declined" in result.content.lower()


def test_quit_raises(mock_console):
    mock_console.confirm.return_value = Decision.QUIT
    provider = FakeProvider([make_turn(calls=[make_call("rm x")])])
    agent = _agent(provider, FakeShell(), mock_console, ApprovalMode.ASK_ALL)
    with pytest.raises(QuitRequested):
        agent.run_turn("go")


def test_bounded_retry_stops(mock_console):
    provider = LoopingProvider("false")
    shell = FakeShell(results=[CommandResult(1, "", "boom", False)] * 10)
    agent = _agent(provider, shell, mock_console, ApprovalMode.YOLO,
                   Config(max_fix_attempts=2))
    agent.run_turn("impossible task")

    assert agent.consecutive_failures == 2
    assert agent._force_stop is True
    assert isinstance(agent.conversation.history[-1], AssistantMessage)
    mock_console.force_stop_notice.assert_called_once_with(2)
    assert provider.disable_tools_log[-1] is True


def test_successful_command_resets_failure_count(mock_console):
    shell = FakeShell(results=[
        CommandResult(1, "", "err", False),
        CommandResult(0, "ok", "", False),
    ])
    provider = FakeProvider([
        make_turn(calls=[make_call("flaky", call_id="a")]),
        make_turn(calls=[make_call("retry", call_id="b")]),
        make_turn(text="fixed"),
    ])
    agent = _agent(provider, shell, mock_console)
    agent.run_turn("fix it")
    assert agent.consecutive_failures == 0


def test_interactive_skips_output_render(mock_console):
    shell = FakeShell(results=[CommandResult(0, "ignored", "", False)])
    provider = FakeProvider([
        make_turn(calls=[make_call("claude", interactive=True)]),
        make_turn(text="ok"),
    ])
    agent = _agent(provider, shell, mock_console)
    agent.run_turn("launch claude")
    assert shell.calls == [("claude", True)]
    mock_console.command_output.assert_not_called()
