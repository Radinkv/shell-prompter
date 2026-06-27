"""The agent loop: drive the model's tool calls, gate them by risk, run them,
and keep the conversation and consecutive-failure state.

The loop is provider-neutral. It speaks only the types in providers.base and
calls one method, provider.complete(). Swapping Anthropic for OpenAI or Gemini
changes nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ApprovalMode, Config
from .history import compact
from .prompts import build_system_texts
from .providers.base import (
    AssistantMessage,
    AssistantTurn,
    ModelProvider,
    ToolInvocation,
    ToolResult,
    ToolResultsMessage,
    UserMessage,
)
from .risk import RiskAssessment, RiskTier, classify
from .shell import CommandResult, Shell, looks_interactive
from .ui import Console, Decision

_DECLINED_MESSAGE = (
    "User declined to run this command. "
    "Suggest an alternative or ask what they'd prefer."
)
_FORCE_STOP_TEMPLATE = (
    "{count} commands have failed in a row, reaching the max_fix_attempts limit "
    "({limit}). Stop running commands now. Briefly explain what's going wrong "
    "and what the user could try."
)
_PAYLOAD_TEMPLATE = (
    "exit_code: {exit_code}\n"
    "cwd: {cwd}\n"
    "stdout:\n{stdout}\n"
    "stderr:\n{stderr}"
)


class QuitRequested(Exception):
    """Raised when the user chooses 'quit' at a confirmation prompt."""


@dataclass
class CommandRequest:
    command: str
    explanation: str
    interactive: bool

    @classmethod
    def from_invocation(cls, invocation: ToolInvocation) -> CommandRequest:
        return cls(
            command=invocation.command,
            explanation=invocation.explanation,
            interactive=invocation.interactive or looks_interactive(invocation.command),
        )


@dataclass
class ToolOutcome:
    """Result of running one tool call.

    is_failure is True only when a command actually ran and exited non-zero.
    A user decline is an error to report but not a failed fix attempt.
    """

    text: str
    is_error: bool
    is_failure: bool


class Conversation:
    """The running neutral history handed to the provider each turn."""

    def __init__(self):
        self.history: list = []

    def add_user(self, text: str) -> None:
        self.history.append(UserMessage(text))

    def add_assistant(self, turn: AssistantTurn) -> None:
        self.history.append(AssistantMessage(turn.text, turn.tool_calls))

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.history.append(ToolResultsMessage(results))


class Agent:
    def __init__(self, provider: ModelProvider, shell: Shell, console: Console,
                 config: Config, mode: ApprovalMode):
        self.provider = provider
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
        self.conversation.add_user(user_text)
        another_round = True
        while another_round:
            another_round = self._run_round()

    def _run_round(self) -> bool:
        """Run one model turn and return whether another round is needed."""
        turn = self._complete()
        self.conversation.add_assistant(turn)
        if not turn.tool_calls:
            return False
        results = self._process_tool_calls(turn.tool_calls)
        self.conversation.add_tool_results(results)
        self._maybe_force_stop()
        return True

    def _complete(self) -> AssistantTurn:
        system_texts = build_system_texts(self.config, self.shell.cwd)
        self.console.begin_stream()
        turn = self.provider.complete(
            compact(self.conversation.history),
            system_texts,
            self._force_stop,
            self.console.stream_text,
        )
        self.console.end_stream()
        return turn

    def _process_tool_calls(self, tool_calls: list[ToolInvocation]) -> list[ToolResult]:
        results = []
        for invocation in tool_calls:
            outcome = self._execute(CommandRequest.from_invocation(invocation))
            self.consecutive_failures = (
                self.consecutive_failures + 1 if outcome.is_failure else 0
            )
            results.append(
                ToolResult(invocation.call_id, outcome.text, outcome.is_error)
            )
        return results

    def _maybe_force_stop(self) -> None:
        """Bound the self-repair loop once too many commands fail in a row."""
        limit = self.config.max_fix_attempts
        if self.consecutive_failures < limit or self._force_stop:
            return
        message = _FORCE_STOP_TEMPLATE.format(
            count=self.consecutive_failures, limit=limit)
        self.conversation.add_user(message)
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
                   assessment: RiskAssessment) -> bool:
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
        return _PAYLOAD_TEMPLATE.format(
            exit_code=result.exit_code,
            cwd=self.shell.cwd,
            stdout=result.stdout,
            stderr=result.stderr,
        )
