"""The agent loop: drive Claude's tool calls, gate them by risk, run them, and
keep the conversation and consecutive-failure state.

Agent orchestrates its injected collaborators (a ClaudeClient, a Shell, and a
Console) but performs no I/O or API access of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ApprovalMode, Config
from .constants import (
    BLOCK_TEXT,
    BLOCK_TOOL_RESULT,
    BLOCK_TOOL_USE,
    INPUT_COMMAND,
    INPUT_EXPLANATION,
    INPUT_INTERACTIVE,
    ROLE_ASSISTANT,
    ROLE_USER,
    STOP_REASON_TOOL_USE,
)
from .llm import ClaudeClient, build_environment_context, build_system_blocks
from .risk import RiskTier, classify
from .shell import CommandResult, Shell, looks_interactive
from .ui import Console, Decision

_DECLINED_MESSAGE = (
    "User declined to run this command. "
    "Suggest an alternative or ask what they'd prefer."
)


class QuitRequested(Exception):
    """Raised when the user chooses 'quit' at a confirmation prompt."""


@dataclass
class CommandRequest:
    command: str
    explanation: str
    interactive: bool

    @classmethod
    def from_tool_input(cls, data: dict) -> "CommandRequest":
        command = data.get(INPUT_COMMAND, "")
        interactive = bool(data.get(INPUT_INTERACTIVE, False))
        return cls(
            command=command,
            explanation=data.get(INPUT_EXPLANATION, ""),
            interactive=interactive or looks_interactive(command),
        )


@dataclass
class ToolOutcome:
    """Result of running one tool call.

    is_failure is True only when a command actually ran and exited non-zero —
    a user decline is an error to report but not a failed fix attempt.
    """

    text: str
    is_error: bool
    is_failure: bool


class Conversation:
    """The running Messages-API history."""

    def __init__(self):
        self.messages: list[dict] = []

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": ROLE_USER, "content": text})

    def add_assistant(self, content) -> None:
        self.messages.append({"role": ROLE_ASSISTANT, "content": content})

    def add_user_blocks(self, blocks: list[dict]) -> None:
        self.messages.append({"role": ROLE_USER, "content": blocks})


def _force_stop_message(count: int, limit: int) -> str:
    return (f"[prompter] {count} commands have failed in a row, reaching the "
            f"max_fix_attempts limit ({limit}). Stop running commands now. "
            f"Briefly explain what's going wrong and what the user could try.")


class Agent:
    def __init__(self, client: ClaudeClient, shell: Shell, console: Console,
                 config: Config, mode: ApprovalMode):
        self.client = client
        self.shell = shell
        self.console = console
        self.config = config
        self.mode = mode
        self.conversation = Conversation()
        self._approve_all = False
        self.consecutive_failures = 0
        self._force_stop = False

    def run_turn(self, user_text: str) -> None:
        self.consecutive_failures = 0
        self._force_stop = False
        self.conversation.add_user_text(user_text)
        another_round = True
        while another_round:
            another_round = self._run_round()

    def _run_round(self) -> bool:
        """Stream one assistant turn; return whether another round is needed."""
        final = self._stream()
        self.conversation.add_assistant(final.content)
        if final.stop_reason != STOP_REASON_TOOL_USE:
            return False
        results = self._process_tool_calls(final)
        self._maybe_force_stop(results)
        self.conversation.add_user_blocks(results)
        return True

    def _stream(self):
        context = build_environment_context(self.config, self.shell.cwd)
        system_blocks = build_system_blocks(context)
        self.console.begin_stream()
        final = self.client.stream(
            self.conversation.messages,
            system_blocks,
            self._force_stop,
            self.console.stream_text,
        )
        self.console.end_stream()
        return final

    def _process_tool_calls(self, final) -> list[dict]:
        results = []
        for block in final.content:
            if block.type != BLOCK_TOOL_USE:
                continue
            outcome = self._execute(CommandRequest.from_tool_input(block.input))
            self.consecutive_failures = (
                self.consecutive_failures + 1 if outcome.is_failure else 0
            )
            results.append({
                "type": BLOCK_TOOL_RESULT,
                "tool_use_id": block.id,
                "content": outcome.text,
                "is_error": outcome.is_error,
            })
        return results

    def _maybe_force_stop(self, results: list[dict]) -> None:
        """Bound the self-repair loop once too many commands fail in a row."""
        limit = self.config.max_fix_attempts
        if self.consecutive_failures < limit or self._force_stop:
            return
        results.append({
            "type": BLOCK_TEXT,
            "text": _force_stop_message(self.consecutive_failures, limit),
        })
        self._force_stop = True
        self.console.force_stop_notice(limit)

    def _execute(self, request: CommandRequest) -> ToolOutcome:
        assessment = classify(request.command)
        if not self._authorize(request, assessment):
            return ToolOutcome(_DECLINED_MESSAGE, is_error=True, is_failure=False)

        result = self.shell.run(request.command, interactive=request.interactive)
        if result.cwd_changed:
            self.console.cwd_change(self.shell.cwd)
        if not request.interactive:
            self.console.command_output(result)
        return ToolOutcome(
            self._format_payload(result),
            is_error=result.failed,
            is_failure=result.failed,
        )

    def _authorize(self, request: CommandRequest,
                   assessment) -> bool:
        if self._auto_approves(assessment.tier):
            self.console.auto_run(assessment, request.command)
            return True
        decision = self.console.confirm(
            assessment, request.command, request.explanation
        )
        if decision == Decision.QUIT:
            raise QuitRequested
        if decision == Decision.SKIP:
            return False
        if decision == Decision.ALL:
            self._approve_all = True
        return True

    def _auto_approves(self, tier: RiskTier) -> bool:
        if self._approve_all or self.mode == ApprovalMode.YOLO:
            return True
        if self.mode == ApprovalMode.ASK_ALL:
            return False
        return tier == RiskTier.SAFE

    def _format_payload(self, result: CommandResult) -> str:
        return (f"exit_code: {result.exit_code}\n"
                f"cwd: {self.shell.cwd}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}")
