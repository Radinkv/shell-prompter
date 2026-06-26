"""Fakes and factories shared across the test modules.

These stand in for the collaborators the agent has injected — a model client, a
shell, and response blocks — so the loop can be exercised without a network,
an API key, or a real subprocess.
"""

from __future__ import annotations

import types

from prompter.shell import CommandResult


def text_block(text: str = "hi"):
    return types.SimpleNamespace(type="text", text=text)


def tool_use_block(command: str = "ls -la", explanation: str = "list",
                   interactive: bool = False, block_id: str = "t1"):
    return types.SimpleNamespace(
        type="tool_use",
        id=block_id,
        input={
            "command": command,
            "explanation": explanation,
            "interactive": interactive,
        },
    )


def final_message(stop_reason: str, content: list):
    return types.SimpleNamespace(stop_reason=stop_reason, content=content)


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


class ScriptedClaude:
    """Returns queued final messages; records disable_tools per call."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.disable_tools_log: list[bool] = []

    def stream(self, messages, system_blocks, disable_tools, on_text):
        self.disable_tools_log.append(disable_tools)
        return self._responses.pop(0)


class LoopingClaude:
    """Proposes a failing command each round until tools are disabled."""

    def __init__(self, command: str = "false"):
        self._command = command
        self.disable_tools_log: list[bool] = []

    def stream(self, messages, system_blocks, disable_tools, on_text):
        self.disable_tools_log.append(disable_tools)
        if disable_tools:
            return final_message("end_turn", [text_block("(summary)")])
        return final_message("tool_use", [tool_use_block(self._command)])
