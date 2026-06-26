"""Shell execution with working-directory tracking across calls.

Each command runs in a fresh ``bash -c``, wrapped so that any ``cd`` it performs
is reflected back into prompter's own state: the wrapper echoes ``$PWD`` on a
sentinel line, which is parsed off the output. That makes ``cd foo && ls``
behave the way a human expects across separate calls.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass

from .constants import EMPTY, NEWLINE

COMMAND_TIMEOUT_SECONDS = 600
TIMEOUT_EXIT_CODE = 124
MAX_OUTPUT_CHARS = 20000

BASH = "bash"
BASH_COMMAND_FLAG = "-c"

_CWD_SENTINEL = "__PROMPTER_CWD_a3f9__:"
_CAPTURE_TEMPLATE = (
    "cd {cwd} 2>/dev/null\n"
    "{command}\n"
    "__prompter_rc=$?\n"
    'printf "%s%s\\n" "{sentinel}" "$PWD"\n'
    "exit $__prompter_rc\n"
)
_INTERACTIVE_TEMPLATE = "cd {cwd} 2>/dev/null; {command}"

_TIMEOUT_MESSAGE = (
    f"Command timed out after {COMMAND_TIMEOUT_SECONDS}s and was killed."
)
_INTERACTIVE_STDOUT = "(interactive program, output shown directly above)"
_OMITTED_TEMPLATE = "{head}\n... [{count} characters omitted] ...\n{tail}"

_INTERACTIVE_COMMANDS = re.compile(
    r"^\s*(claude|vim|nvim|vi|nano|emacs|less|more|top|htop|ssh|tmux|screen|"
    r"python3?|node|irb|psql|mysql|sqlite3|fzf|man)\b"
)


def looks_interactive(command: str) -> bool:
    """True for programs that take over the terminal even without a flag."""
    return bool(_INTERACTIVE_COMMANDS.match(command))


def truncate(text: str) -> str:
    """Cap output handed back to the model, keeping head and tail."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return _OMITTED_TEMPLATE.format(
        head=text[:half], tail=text[-half:], count=len(text) - MAX_OUTPUT_CHARS
    )


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    cwd_changed: bool

    @property
    def failed(self) -> bool:
        return self.exit_code != 0


class Shell:
    def __init__(self, cwd: str | None = None):
        self.cwd = cwd or os.getcwd()

    def run(self, command: str, interactive: bool = False) -> CommandResult:
        if interactive:
            return self._run_interactive(command)
        return self._run_captured(command)

    def _run_captured(self, command: str) -> CommandResult:
        script = _CAPTURE_TEMPLATE.format(
            cwd=shlex.quote(self.cwd),
            command=command,
            sentinel=_CWD_SENTINEL,
        )
        try:
            proc = subprocess.run(
                [BASH, BASH_COMMAND_FLAG, script],
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(TIMEOUT_EXIT_CODE, EMPTY, _TIMEOUT_MESSAGE, False)

        stdout, changed = self._extract_cwd(proc.stdout)
        return CommandResult(
            proc.returncode, truncate(stdout), truncate(proc.stderr), changed
        )

    def _run_interactive(self, command: str) -> CommandResult:
        """Hand the real terminal to the program. Its cwd cannot be recovered."""
        script = _INTERACTIVE_TEMPLATE.format(
            cwd=shlex.quote(self.cwd), command=command
        )
        proc = subprocess.run([BASH, BASH_COMMAND_FLAG, script])
        return CommandResult(proc.returncode, _INTERACTIVE_STDOUT, EMPTY, False)

    def _extract_cwd(self, stdout: str) -> tuple[str, bool]:
        changed = False
        kept = []
        for line in stdout.splitlines():
            if line.startswith(_CWD_SENTINEL):
                new_cwd = line[len(_CWD_SENTINEL):].strip()
                if new_cwd and new_cwd != self.cwd and os.path.isdir(new_cwd):
                    self.cwd = new_cwd
                    changed = True
            else:
                kept.append(line)
        trailing = NEWLINE if stdout.endswith(NEWLINE) and kept else EMPTY
        return NEWLINE.join(kept) + trailing, changed
