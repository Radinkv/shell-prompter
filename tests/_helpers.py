"""Fakes and factories shared across the test modules.

These stand in for the collaborators the agent has injected, a model provider
and a shell, so the loop can be exercised without a network, an API key, or a
real subprocess.
"""

from __future__ import annotations

from prompter.providers.base import AssistantTurn, ToolInvocation
from prompter.shell import CommandResult


def make_call(command: str = "ls -la", explanation: str = "list",
              interactive: bool = False, call_id: str = "c1") -> ToolInvocation:
    return ToolInvocation(call_id=call_id, command=command,
                          explanation=explanation, interactive=interactive)


def make_turn(text: str = "", calls: list | None = None) -> AssistantTurn:
    return AssistantTurn(text, list(calls or []))


class FakeShell:
    """Records calls and returns queued CommandResults (default: success)."""

    def __init__(self, cwd: str = "/work", results: list | None = None):
        self.cwd = cwd
        self._results = list(results or [])
        self.calls: list[tuple[str, bool]] = []

    def run(self, command: str, interactive: bool = False) -> CommandResult:
        self.calls.append((command, interactive))
        if self._results:
            return self._results.pop(0)
        return CommandResult(0, "ok", "", False)


class FakeProvider:
    """Returns queued AssistantTurns and records the disable_tools flag per call."""

    name = "fake"
    model = "fake-model"

    def __init__(self, turns: list):
        self._turns = list(turns)
        self.disable_tools_log: list[bool] = []

    def complete(self, _history, _system_texts, disable_tools, _on_text) -> AssistantTurn:
        self.disable_tools_log.append(disable_tools)
        return self._turns.pop(0)


class LoopingProvider:
    """Proposes a failing command each round until tools are disabled."""

    name = "fake"
    model = "fake-model"

    def __init__(self, command: str = "false"):
        self._command = command
        self.disable_tools_log: list[bool] = []

    def complete(self, _history, _system_texts, disable_tools, _on_text) -> AssistantTurn:
        self.disable_tools_log.append(disable_tools)
        if disable_tools:
            return make_turn(text="(summary)")
        return make_turn(calls=[make_call(self._command)])
