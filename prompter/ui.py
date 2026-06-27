"""All terminal interaction: confirmation prompts, banners, command output,
streamed-response printing, and REPL input.

Centralising I/O here keeps the agent free of print/input calls and makes it
drivable with a fake console in tests. Every user-facing string and every line
template is a named constant in this module.
"""

from __future__ import annotations

import sys
from enum import Enum

from .colors import Palette, palette as default_palette
from .config import ApprovalMode, Config, PROGRAM_NAME
from .constants import EMPTY, NEWLINE
from .risk import RiskAssessment
from .shell import CommandResult

_TAGLINE = "natural-language shell agent"
_MODE_PREFIX = "mode="
_STATUS_TEMPLATE = (
    "{provider}/{model} · workspace: {workspace} · max-fix: {max_fix}"
)
_YOLO_WARNING = (
    "⚠ YOLO mode: every command runs without asking, including dangerous ones."
)

_COMMAND_PREFIX = "$ "
_TAG_TEMPLATE = "[{label}]"
_GUTTER = "│"
_CWD_PREFIX = "↪ now in "
_FORCE_STOP_TEMPLATE = (
    "⚠ hit max_fix_attempts ({limit}). Asking Claude to summarize and stop."
)

_RUN_QUESTION = "run?"
_ANSWER_HINT_TEMPLATE = (
    "[{green}yes{reset} / {yellow}no{reset} / {cyan}all{reset} / {red}quit{reset}]"
)
_RETRY_PROMPT = "  Please answer y, n, a, or q."

_STOPPED = "Stopped."
_PROBLEM_MARK = "✗"
_SUCCESS_MARK = "✓"
_HINT_LABEL = "{label}:"
_FIELD_WIDTH = 12
_PROVIDER_WIDTH = 10

_REPL_INTRO_TEMPLATE = (
    "Interactive mode. Type a request, or 'exit' to quit. Working dir: {cwd}"
)
_REPL_ARROW = "➜"

_BANNER_HEAD = "{magenta}{bold}{program}{reset} {dim}· {tagline} · {mode_prefix}{mode}{reset}"
_BANNER_STATUS = "{dim}{status}{reset}"
_BANNER_YOLO = "{red}{bold}{warning}{reset}"
_AUTO_RUN_LINE = "\n  {color}{tag}{reset} {bold}{prefix}{command}{reset}"
_CONFIRM_HEADER = "  {color}{bold}{tag}{reset} {dim}{reason}{reset}"
_CONFIRM_EXPLANATION = "  {dim}{explanation}{reset}"
_CONFIRM_COMMAND = "  {bold}{prefix}{command}{reset}"
_DECISION_PROMPT = "  {color}{question}{reset} {hint} "
_CWD_LINE = "  {dim}{prefix}{cwd}{reset}"
_GUTTER_LINE = "  {color}{gutter}{reset} {line}"
_FORCE_STOP_LINE = "\n  {yellow}{message}{reset}"
_STREAM_OPEN = "\n{cyan}"
_STREAM_CLOSE = "{reset}\n"
_NOTE_LINE = "{dim}{text}{reset}"
_STOPPED_LINE = "\n{dim}{stopped}{reset}"
_PROBLEM_TITLE_LINE = "{red}{bold}{mark}{reset} {bold}{title}{reset}"
_PROBLEM_DETAIL_LINE = "    {dim}{detail}{reset}"
_PROBLEM_HINT_LINE = "    {dim}{label}{reset}  {cyan}{command}{reset}"
_SUCCESS_LINE = "{green}{mark}{reset} {message}"
_FIELD_ROW = "  {dim}{label}{reset} {value}"
_KEY_ROW = "  {provider} {color}{status}{reset}"
_REPL_INTRO_LINE = "{dim}{intro}{reset}"
_REPL_PROMPT_LINE = "\n{magenta}{program}{reset} {dim}{cwd}{reset} {arrow} "


class Decision(Enum):
    RUN = "run"
    SKIP = "skip"
    ALL = "all"
    QUIT = "quit"


_ANSWER_YES = "y"
_ANSWER_YES_LONG = "yes"
_ANSWER_NO = "n"
_ANSWER_NO_LONG = "no"
_ANSWER_ALL = "a"
_ANSWER_ALL_LONG = "all"
_ANSWER_QUIT = "q"
_ANSWER_QUIT_LONG = "quit"

_ANSWERS = {
    _ANSWER_YES: Decision.RUN,
    _ANSWER_YES_LONG: Decision.RUN,
    EMPTY: Decision.RUN,
    _ANSWER_NO: Decision.SKIP,
    _ANSWER_NO_LONG: Decision.SKIP,
    _ANSWER_ALL: Decision.ALL,
    _ANSWER_ALL_LONG: Decision.ALL,
    _ANSWER_QUIT: Decision.QUIT,
    _ANSWER_QUIT_LONG: Decision.QUIT,
}


class Console:
    def __init__(self, palette: Palette | None = None):
        self.c = palette or default_palette
        self._stream_open = False

    def banner(self, config: Config, mode: ApprovalMode) -> None:
        c = self.c
        status = _STATUS_TEMPLATE.format(
            provider=config.provider,
            model=config.resolved_model,
            workspace=config.workspace_path,
            max_fix=config.max_fix_attempts,
        )
        print(_BANNER_HEAD.format(
            magenta=c.MAGENTA, bold=c.BOLD, program=PROGRAM_NAME, reset=c.RESET,
            dim=c.DIM, tagline=_TAGLINE, mode_prefix=_MODE_PREFIX, mode=mode.value))
        print(_BANNER_STATUS.format(dim=c.DIM, status=status, reset=c.RESET))
        if mode == ApprovalMode.YOLO:
            print(_BANNER_YOLO.format(red=c.RED, bold=c.BOLD, warning=_YOLO_WARNING,
                                      reset=c.RESET))

    def auto_run(self, assessment: RiskAssessment, command: str) -> None:
        c = self.c
        tag = _TAG_TEMPLATE.format(label=assessment.tier.label)
        print(_AUTO_RUN_LINE.format(
            color=assessment.tier.color(c), tag=tag, reset=c.RESET, bold=c.BOLD,
            prefix=_COMMAND_PREFIX, command=command))

    def confirm(self, assessment: RiskAssessment, command: str,
                explanation: str) -> Decision:
        c = self.c
        color = assessment.tier.color(c)
        tag = _TAG_TEMPLATE.format(label=assessment.tier.label)
        print()
        print(_CONFIRM_HEADER.format(color=color, bold=c.BOLD, tag=tag, reset=c.RESET,
                                     dim=c.DIM, reason=assessment.reason))
        if explanation:
            print(_CONFIRM_EXPLANATION.format(dim=c.DIM, explanation=explanation,
                                              reset=c.RESET))
        print(_CONFIRM_COMMAND.format(bold=c.BOLD, prefix=_COMMAND_PREFIX,
                                      command=command, reset=c.RESET))
        return self._read_decision(color)

    def _read_decision(self, color: str) -> Decision:
        c = self.c
        hint = _ANSWER_HINT_TEMPLATE.format(
            green=c.GREEN, yellow=c.YELLOW, cyan=c.CYAN, red=c.RED, reset=c.RESET
        )
        prompt = _DECISION_PROMPT.format(
            color=color, question=_RUN_QUESTION, reset=c.RESET, hint=hint)
        while True:
            try:
                answer = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return Decision.QUIT
            if answer in _ANSWERS:
                return _ANSWERS[answer]
            print(_RETRY_PROMPT)

    def cwd_change(self, cwd: str) -> None:
        c = self.c
        print(_CWD_LINE.format(dim=c.DIM, prefix=_CWD_PREFIX, cwd=cwd, reset=c.RESET))

    def command_output(self, result: CommandResult) -> None:
        out = (result.stdout or EMPTY).rstrip()
        err = (result.stderr or EMPTY).rstrip()
        if out:
            print(self._indent(out))
        if err:
            print(self._indent(err, self.c.RED))

    def _indent(self, text: str, color: str | None = None) -> str:
        color = self.c.DIM if color is None else color
        return NEWLINE.join(
            _GUTTER_LINE.format(color=color, gutter=_GUTTER, reset=self.c.RESET, line=line)
            for line in text.splitlines())

    def force_stop_notice(self, limit: int) -> None:
        message = _FORCE_STOP_TEMPLATE.format(limit=limit)
        print(_FORCE_STOP_LINE.format(yellow=self.c.YELLOW, message=message,
                                      reset=self.c.RESET))

    def begin_stream(self) -> None:
        self._stream_open = False

    def stream_text(self, text: str) -> None:
        if not self._stream_open:
            sys.stdout.write(_STREAM_OPEN.format(cyan=self.c.CYAN))
            self._stream_open = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def end_stream(self) -> None:
        if self._stream_open:
            sys.stdout.write(_STREAM_CLOSE.format(reset=self.c.RESET))
            sys.stdout.flush()
            self._stream_open = False

    def note(self, text: str) -> None:
        print(_NOTE_LINE.format(dim=self.c.DIM, text=text, reset=self.c.RESET),
              file=sys.stderr)

    def stopped(self) -> None:
        print(_STOPPED_LINE.format(dim=self.c.DIM, stopped=_STOPPED, reset=self.c.RESET))

    def problem(self, title: str, hints=(), detail: str | None = None) -> None:
        """Render a failure as a problem line plus runnable fix commands."""
        c = self.c
        print(_PROBLEM_TITLE_LINE.format(red=c.RED, bold=c.BOLD, mark=_PROBLEM_MARK,
                                         reset=c.RESET, title=title), file=sys.stderr)
        if detail:
            print(_PROBLEM_DETAIL_LINE.format(dim=c.DIM, detail=detail, reset=c.RESET),
                  file=sys.stderr)
        rows = [(_HINT_LABEL.format(label=label), command) for label, command in hints]
        width = max((len(label) for label, _ in rows), default=0)
        for label, command in rows:
            print(_PROBLEM_HINT_LINE.format(dim=c.DIM, label=label.ljust(width),
                                            reset=c.RESET, cyan=c.CYAN, command=command),
                  file=sys.stderr)

    def success(self, message: str) -> None:
        print(_SUCCESS_LINE.format(green=self.c.GREEN, mark=_SUCCESS_MARK,
                                   reset=self.c.RESET, message=message))

    def info(self, text: str) -> None:
        print(text)

    def field(self, label: str, value: str) -> None:
        c = self.c
        print(_FIELD_ROW.format(dim=c.DIM, label=label.ljust(_FIELD_WIDTH),
                                reset=c.RESET, value=value))

    def key_status(self, provider: str, status: str, present: bool) -> None:
        c = self.c
        color = c.GREEN if present else c.DIM
        print(_KEY_ROW.format(provider=provider.ljust(_PROVIDER_WIDTH), color=color,
                              status=status, reset=c.RESET))

    def repl_intro(self, cwd: str) -> None:
        intro = _REPL_INTRO_TEMPLATE.format(cwd=cwd)
        print(_REPL_INTRO_LINE.format(dim=self.c.DIM, intro=intro, reset=self.c.RESET))

    def repl_prompt(self, cwd: str) -> str:
        c = self.c
        return input(_REPL_PROMPT_LINE.format(
            magenta=c.MAGENTA, program=PROGRAM_NAME, reset=c.RESET, dim=c.DIM,
            cwd=cwd, arrow=_REPL_ARROW)).strip()
