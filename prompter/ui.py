"""All terminal interaction: confirmation prompts, banners, command output,
streamed-response printing, and REPL input.

Centralising I/O here keeps the agent free of print/input calls and makes it
drivable with a fake console in tests. Every user-facing string is a named
constant in this module.
"""

from __future__ import annotations

import sys
from enum import Enum

from .colors import Palette, palette as default_palette
from .config import ApprovalMode, Config, PROGRAM_NAME
from .risk import RiskAssessment
from .shell import CommandResult

_TAGLINE = "natural-language shell agent"
_MODE_PREFIX = "mode="
_STATUS_TEMPLATE = "model: {model} · workspace: {workspace} · max-fix: {max_fix}"
_YOLO_WARNING = (
    "⚠ YOLO mode: every command runs without asking, including dangerous ones."
)

_COMMAND_PREFIX = "$ "
_TAG_TEMPLATE = "[{label}]"
_GUTTER = "│"
_CWD_PREFIX = "↪ now in "
_FORCE_STOP_TEMPLATE = (
    "⚠ hit max_fix_attempts ({limit}); asking Claude to summarize and stop."
)

_RUN_QUESTION = "run?"
_ANSWER_HINT_TEMPLATE = (
    "[{green}y{reset}es / {yellow}n{reset}o / {cyan}a{reset}ll / {red}q{reset}uit]"
)
_RETRY_PROMPT = "  Please answer y, n, a, or q."

_STOPPED = "Stopped."
_AUTH_ERROR = "Authentication failed."
_AUTH_HINT = "Set ANTHROPIC_API_KEY or run `ant auth login`."
_API_ERROR_PREFIX = "API error:"

_REPL_INTRO_TEMPLATE = (
    "Interactive mode. Type a request, or 'exit' to quit. Working dir: {cwd}"
)
_REPL_ARROW = "➜"


class Decision(Enum):
    RUN = "run"
    SKIP = "skip"
    ALL = "all"
    QUIT = "quit"


_ANSWERS = {
    "y": Decision.RUN,
    "yes": Decision.RUN,
    "": Decision.RUN,
    "n": Decision.SKIP,
    "no": Decision.SKIP,
    "a": Decision.ALL,
    "all": Decision.ALL,
    "q": Decision.QUIT,
    "quit": Decision.QUIT,
}


class Console:
    def __init__(self, palette: Palette | None = None):
        self.c = palette or default_palette
        self._stream_open = False

    # -- startup banner ----------------------------------------------------
    def banner(self, config: Config, mode: ApprovalMode) -> None:
        c = self.c
        status = _STATUS_TEMPLATE.format(
            model=config.model,
            workspace=config.workspace_path,
            max_fix=config.max_fix_attempts,
        )
        print(f"{c.MAGENTA}{c.BOLD}{PROGRAM_NAME}{c.RESET} "
              f"{c.DIM}· {_TAGLINE} · {_MODE_PREFIX}{mode.value}{c.RESET}")
        print(f"{c.DIM}{status}{c.RESET}")
        if mode == ApprovalMode.YOLO:
            print(f"{c.RED}{c.BOLD}{_YOLO_WARNING}{c.RESET}")

    # -- command lifecycle -------------------------------------------------
    def auto_run(self, assessment: RiskAssessment, command: str) -> None:
        c = self.c
        tag = _TAG_TEMPLATE.format(label=assessment.tier.label)
        print(f"\n  {assessment.tier.color(c)}{tag}{c.RESET} "
              f"{c.BOLD}{_COMMAND_PREFIX}{command}{c.RESET}")

    def confirm(self, assessment: RiskAssessment, command: str,
                explanation: str) -> Decision:
        c = self.c
        color = assessment.tier.color(c)
        tag = _TAG_TEMPLATE.format(label=assessment.tier.label)
        print()
        print(f"  {color}{c.BOLD}{tag}{c.RESET} {c.DIM}{assessment.reason}{c.RESET}")
        if explanation:
            print(f"  {c.DIM}{explanation}{c.RESET}")
        print(f"  {c.BOLD}{_COMMAND_PREFIX}{command}{c.RESET}")
        return self._read_decision(color)

    def _read_decision(self, color: str) -> Decision:
        c = self.c
        hint = _ANSWER_HINT_TEMPLATE.format(
            green=c.GREEN, yellow=c.YELLOW, cyan=c.CYAN, red=c.RED, reset=c.RESET
        )
        prompt = f"  {color}{_RUN_QUESTION}{c.RESET} {hint} "
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
        print(f"  {self.c.DIM}{_CWD_PREFIX}{cwd}{self.c.RESET}")

    def command_output(self, result: CommandResult) -> None:
        out = (result.stdout or "").rstrip()
        err = (result.stderr or "").rstrip()
        if out:
            print(self._indent(out))
        if err:
            print(self._indent(err, self.c.RED))

    def _indent(self, text: str, color: str | None = None) -> str:
        color = self.c.DIM if color is None else color
        return "\n".join(f"  {color}{_GUTTER}{self.c.RESET} {line}"
                         for line in text.splitlines())

    def force_stop_notice(self, limit: int) -> None:
        message = _FORCE_STOP_TEMPLATE.format(limit=limit)
        print(f"\n  {self.c.YELLOW}{message}{self.c.RESET}")

    # -- streamed assistant text ------------------------------------------
    def begin_stream(self) -> None:
        self._stream_open = False

    def stream_text(self, text: str) -> None:
        if not self._stream_open:
            sys.stdout.write(f"\n{self.c.CYAN}")
            self._stream_open = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def end_stream(self) -> None:
        if self._stream_open:
            sys.stdout.write(f"{self.c.RESET}\n")
            sys.stdout.flush()
            self._stream_open = False

    # -- notices and errors -----------------------------------------------
    def note(self, text: str) -> None:
        print(f"{self.c.DIM}{text}{self.c.RESET}", file=sys.stderr)

    def stopped(self) -> None:
        print(f"\n{self.c.DIM}{_STOPPED}{self.c.RESET}")

    def auth_error(self) -> None:
        print(f"{self.c.RED}{_AUTH_ERROR}{self.c.RESET} {_AUTH_HINT}",
              file=sys.stderr)

    def api_error(self, detail: object) -> None:
        print(f"{self.c.RED}{_API_ERROR_PREFIX}{self.c.RESET} {detail}",
              file=sys.stderr)

    # -- REPL --------------------------------------------------------------
    def repl_intro(self, cwd: str) -> None:
        intro = _REPL_INTRO_TEMPLATE.format(cwd=cwd)
        print(f"{self.c.DIM}{intro}{self.c.RESET}")

    def repl_prompt(self, cwd: str) -> str:
        c = self.c
        return input(f"\n{c.MAGENTA}{PROGRAM_NAME}{c.RESET} "
                     f"{c.DIM}{cwd}{c.RESET} {_REPL_ARROW} ").strip()
