"""All terminal interaction: confirmation prompts, banners, command output,
streamed-response printing, and REPL input.

Centralising I/O here keeps the agent free of print/input calls and makes it
drivable with a fake console in tests.
"""

from __future__ import annotations

import sys
from enum import Enum

from .colors import Palette, palette as default_palette
from .config import ApprovalMode, Config
from .risk import RiskAssessment
from .shell import CommandResult


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
_RETRY_PROMPT = "  Please answer y, n, a, or q."


class Console:
    def __init__(self, palette: Palette | None = None):
        self.c = palette or default_palette
        self._stream_open = False

    # -- startup banner ----------------------------------------------------
    def banner(self, config: Config, mode: ApprovalMode) -> None:
        c = self.c
        print(f"{c.MAGENTA}{c.BOLD}prompter{c.RESET} "
              f"{c.DIM}· natural-language shell agent · mode={mode.value}{c.RESET}")
        print(f"{c.DIM}model: {config.model} · "
              f"workspace: {config.workspace_path} · "
              f"max-fix: {config.max_fix_attempts}{c.RESET}")
        if mode == ApprovalMode.YOLO:
            print(f"{c.RED}{c.BOLD}⚠ YOLO mode: every command runs without "
                  f"asking, including dangerous ones.{c.RESET}")

    # -- command lifecycle -------------------------------------------------
    def auto_run(self, assessment: RiskAssessment, command: str) -> None:
        c = self.c
        color = assessment.tier.color(c)
        print(f"\n  {color}[{assessment.tier.label}]{c.RESET} "
              f"{c.BOLD}$ {command}{c.RESET}")

    def confirm(self, assessment: RiskAssessment, command: str,
                explanation: str) -> Decision:
        c = self.c
        color = assessment.tier.color(c)
        print()
        print(f"  {color}{c.BOLD}[{assessment.tier.label}]{c.RESET} "
              f"{c.DIM}{assessment.reason}{c.RESET}")
        if explanation:
            print(f"  {c.DIM}{explanation}{c.RESET}")
        print(f"  {c.BOLD}$ {command}{c.RESET}")
        return self._read_decision(color)

    def _read_decision(self, color: str) -> Decision:
        c = self.c
        prompt = (f"  {color}run?{c.RESET} "
                  f"[{c.GREEN}y{c.RESET}es / {c.YELLOW}n{c.RESET}o / "
                  f"{c.CYAN}a{c.RESET}ll / {c.RED}q{c.RESET}uit] ")
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
        print(f"  {self.c.DIM}↪ now in {cwd}{self.c.RESET}")

    def command_output(self, result: CommandResult) -> None:
        out = (result.stdout or "").rstrip()
        err = (result.stderr or "").rstrip()
        if out:
            print(self._indent(out))
        if err:
            print(self._indent(err, self.c.RED))

    def _indent(self, text: str, color: str | None = None) -> str:
        color = self.c.DIM if color is None else color
        return "\n".join(f"  {color}│{self.c.RESET} {line}"
                         for line in text.splitlines())

    def force_stop_notice(self, limit: int) -> None:
        print(f"\n  {self.c.YELLOW}⚠ hit max_fix_attempts ({limit}); "
              f"asking Claude to summarize and stop.{self.c.RESET}")

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
        print(f"\n{self.c.DIM}Stopped.{self.c.RESET}")

    def auth_error(self) -> None:
        print(f"{self.c.RED}Authentication failed.{self.c.RESET} "
              f"Set ANTHROPIC_API_KEY or run `ant auth login`.", file=sys.stderr)

    def api_error(self, detail: object) -> None:
        print(f"{self.c.RED}API error:{self.c.RESET} {detail}", file=sys.stderr)

    # -- REPL --------------------------------------------------------------
    def repl_intro(self, cwd: str) -> None:
        print(f"{self.c.DIM}Interactive mode. Type a request, or 'exit' to "
              f"quit. Working dir: {cwd}{self.c.RESET}")

    def repl_prompt(self, cwd: str) -> str:
        return input(f"\n{self.c.MAGENTA}prompter{self.c.RESET} "
                     f"{self.c.DIM}{cwd}{self.c.RESET} ➜ ").strip()
