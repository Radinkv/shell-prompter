#!/usr/bin/env python3
"""prompter — a natural-language shell agent powered by Claude.

You describe what you want in plain English; prompter figures out the shell
commands, shows them to you with a risk rating, and (after your approval for
anything non-trivial) runs them — keeping track of the working directory as it
goes, just like a real shell session.

    prompter "make a folder called scratch, cd into it, then run claude"
    prompter "download uv if it isn't installed, then check the version"
    prompter            # no prompt → interactive REPL

The agent's workspace is your shell, not a single repo. Claude is just one of
the tools it knows how to launch.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
import re

try:
    import anthropic
except ImportError:  # pragma: no cover - import guard
    sys.exit(
        "The 'anthropic' package is required.\n"
        "Install it with:  pip install anthropic\n"
        "(or reinstall prompter with:  pip install -e .)"
    )

MODEL = "claude-opus-4-8"

# ----------------------------------------------------------------------------
# Terminal colors
# ----------------------------------------------------------------------------


class C:
    """ANSI color codes, auto-disabled when output isn't a TTY."""

    _on = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    RESET = "\033[0m" if _on else ""
    BOLD = "\033[1m" if _on else ""
    DIM = "\033[2m" if _on else ""
    RED = "\033[31m" if _on else ""
    GREEN = "\033[32m" if _on else ""
    YELLOW = "\033[33m" if _on else ""
    BLUE = "\033[34m" if _on else ""
    MAGENTA = "\033[35m" if _on else ""
    CYAN = "\033[36m" if _on else ""


# ----------------------------------------------------------------------------
# Risk classification
# ----------------------------------------------------------------------------
#
# Every command Claude proposes is graded into one of three tiers. The tier
# decides whether prompter runs it automatically or asks you first.
#
#   safe     read-only / clearly reversible (ls, cd, mkdir, git status, ...)
#   confirm  installs, downloads, clones, moves, single-file deletes
#   danger   destructive, privileged, or "pipe the internet into a shell"

# Order matters: the first pattern that matches wins, and danger is checked
# before confirm so e.g. `sudo rm -rf` lands in danger, not confirm.

_DANGER = [
    (r"\brm\b[^|;&]*\s-[a-z]*[rf]", "recursive/forced delete"),
    (r"\bsudo\b", "runs with root privileges"),
    (r"\bdoas\b", "runs with elevated privileges"),
    (r"\bmkfs\b", "formats a filesystem"),
    (r"\bdd\b\s+.*of=", "raw disk write with dd"),
    (r">\s*/dev/(sd|disk|nvme|hd)", "writes directly to a disk device"),
    (r"\bchmod\b\s+-[a-z]*R", "recursive permission change"),
    (r"\bchown\b\s+-[a-z]*R", "recursive ownership change"),
    (r":\(\)\s*\{.*\|.*&\s*\}", "looks like a fork bomb"),
    (r"\bgit\b.*\bpush\b.*(--force|-f)\b", "force-push (rewrites remote history)"),
    (r"(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh)\b",
     "downloads and executes code from the internet"),
    (r"\beval\b", "evaluates a constructed command string"),
    (r"\brm\b[^|;&]*\s+(/|~|\$HOME)\s*$", "deletes your home or root"),
]

_CONFIRM = [
    (r"\b(brew|apt|apt-get|yum|dnf|pacman|port)\b\s+(install|add|upgrade)",
     "installs system packages"),
    (r"\b(pip|pip3|pipx|uv)\b.*\binstall\b", "installs Python packages"),
    (r"\bnpm\b\s+(install|i|add)\b", "installs npm packages"),
    (r"\b(pnpm|yarn|bun)\b\s+(install|add|i)\b", "installs JS packages"),
    (r"\bcargo\b\s+install\b", "installs a Rust crate"),
    (r"\bgo\b\s+install\b", "installs a Go binary"),
    (r"\bgit\b\s+clone\b", "clones a repository"),
    (r"\bgit\b\s+push\b", "pushes to a remote"),
    (r"\b(curl|wget)\b", "downloads from the internet"),
    (r"\bmv\b", "moves/renames files"),
    (r"\brm\b", "deletes files"),
    (r"\bkill(all)?\b", "terminates processes"),
    (r"\bgit\b\s+(reset|checkout|restore)\b.*--hard", "discards local changes"),
    (r"\bdocker\b\s+(run|rm|rmi|system\s+prune)", "modifies Docker state"),
    (r">\s*[^|&\s]", "redirects output into a file (may overwrite)"),
]

# Commands we treat as interactive even if Claude forgets the flag — these take
# over the terminal and shouldn't have their output captured.
INTERACTIVE_HINT = re.compile(
    r"^\s*(claude|vim|nvim|vi|nano|emacs|less|more|top|htop|ssh|tmux|screen|"
    r"python3?|node|irb|psql|mysql|sqlite3|fzf|man)\b"
)


def classify_risk(command: str) -> tuple[str, str]:
    """Return (tier, reason) where tier is 'safe' | 'confirm' | 'danger'."""
    for pattern, reason in _DANGER:
        if re.search(pattern, command):
            return "danger", reason
    for pattern, reason in _CONFIRM:
        if re.search(pattern, command):
            return "confirm", reason
    return "safe", "read-only or clearly reversible"


RISK_STYLE = {
    "safe": (C.GREEN, "SAFE"),
    "confirm": (C.YELLOW, "CONFIRM"),
    "danger": (C.RED, "DANGER"),
}


# ----------------------------------------------------------------------------
# Shell execution with working-directory tracking
# ----------------------------------------------------------------------------
#
# Each command runs in a fresh `bash -c`, but we wrap it so that any `cd` it
# performs is reflected back into prompter's own state. We do this by echoing
# $PWD on a sentinel line after the command, then parsing it off the output.
# That makes `cd foo && ls` behave the way a human expects across calls.

_CWD_SENTINEL = "__PROMPTER_CWD_a3f9__:"
_MAX_OUTPUT = 20000  # chars of output handed back to Claude per command


class Shell:
    def __init__(self, cwd: str | None = None):
        self.cwd = cwd or os.getcwd()

    def run(self, command: str, interactive: bool = False) -> dict:
        """Run a command. Returns {exit_code, stdout, stderr, cwd_changed}."""
        if interactive:
            return self._run_interactive(command)
        return self._run_captured(command)

    def _run_captured(self, command: str) -> dict:
        wrapped = (
            f"cd {shlex.quote(self.cwd)} 2>/dev/null\n"
            f"{command}\n"
            f"__prompter_rc=$?\n"
            f'printf "%s%s\\n" "{_CWD_SENTINEL}" "$PWD"\n'
            f"exit $__prompter_rc\n"
        )
        try:
            proc = subprocess.run(
                ["bash", "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return {
                "exit_code": 124,
                "stdout": "",
                "stderr": "Command timed out after 600s and was killed.",
                "cwd_changed": False,
            }

        stdout, changed = self._extract_cwd(proc.stdout)
        return {
            "exit_code": proc.returncode,
            "stdout": _truncate(stdout),
            "stderr": _truncate(proc.stderr),
            "cwd_changed": changed,
        }

    def _run_interactive(self, command: str) -> dict:
        # Hand the real terminal to the program (no capture) so the user can
        # actually interact with it. We can't recover its cwd afterward.
        full = f"cd {shlex.quote(self.cwd)} 2>/dev/null; {command}"
        proc = subprocess.run(["bash", "-c", full])
        return {
            "exit_code": proc.returncode,
            "stdout": "(interactive program — output shown directly above)",
            "stderr": "",
            "cwd_changed": False,
        }

    def _extract_cwd(self, stdout: str) -> tuple[str, bool]:
        lines = stdout.splitlines()
        changed = False
        kept = []
        for line in lines:
            if line.startswith(_CWD_SENTINEL):
                new_cwd = line[len(_CWD_SENTINEL):].strip()
                if new_cwd and new_cwd != self.cwd and os.path.isdir(new_cwd):
                    self.cwd = new_cwd
                    changed = True
            else:
                kept.append(line)
        return ("\n".join(kept) + ("\n" if stdout.endswith("\n") and kept else ""),
                changed)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT:
        return text
    head = text[: _MAX_OUTPUT // 2]
    tail = text[-_MAX_OUTPUT // 2:]
    omitted = len(text) - _MAX_OUTPUT
    return f"{head}\n... [{omitted} characters omitted] ...\n{tail}"


# ----------------------------------------------------------------------------
# The tool Claude calls
# ----------------------------------------------------------------------------

RUN_TOOL = {
    "name": "run_command",
    "description": (
        "Run a single shell command in the user's terminal session. The "
        "working directory persists between calls (so `cd` works as expected). "
        "Use one command per call and build up to the goal step by step. Set "
        "`interactive` to true for programs that take over the terminal and "
        "need the user to type into them (claude, vim, ssh, a REPL, etc.) — "
        "their output is shown to the user directly and not returned to you."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The exact shell command to run.",
            },
            "explanation": {
                "type": "string",
                "description": "One short sentence on what this does and why.",
            },
            "interactive": {
                "type": "boolean",
                "description": "True if the program needs an interactive terminal.",
            },
        },
        "required": ["command", "explanation"],
    },
}


SYSTEM_PROMPT = """You are prompter, a command-line agent that turns the user's \
plain-English requests into real shell actions on their machine.

Your workspace is the user's entire shell session, not a single project folder. \
You accomplish goals by calling the run_command tool, one command at a time, \
and reacting to each result before deciding the next step.

Guidelines:
- Prefer small, verifiable steps over one giant command. Check that something \
worked before relying on it.
- The working directory persists across run_command calls. Use plain `cd` to \
move around; you do not need to chain everything with && in one command.
- For programs that need an interactive terminal (claude, vim/nvim, ssh, a \
language REPL, top/htop, fzf), pass interactive=true. You will not see their \
output; assume the user is interacting with them directly.
- Be careful with destructive or privileged actions. The user's harness gates \
risky commands behind a confirmation prompt, so don't be surprised if a command \
comes back as "declined by user" — adapt and propose an alternative or ask.
- Don't fabricate results. Base what you report on actual command output.
- Keep your prose brief. A sentence of context before a command and a short \
summary at the end is plenty; the user can see the commands and their output.
- When the goal is achieved, say so concisely and stop calling tools."""


# ----------------------------------------------------------------------------
# Confirmation prompt
# ----------------------------------------------------------------------------


def confirm(command: str, tier: str, reason: str, explanation: str) -> str:
    """Ask the user. Returns 'run', 'skip', 'all', or 'quit'."""
    color, label = RISK_STYLE[tier]
    print()
    print(f"  {color}{C.BOLD}[{label}]{C.RESET} {C.DIM}{reason}{C.RESET}")
    if explanation:
        print(f"  {C.DIM}{explanation}{C.RESET}")
    print(f"  {C.BOLD}$ {command}{C.RESET}")
    while True:
        try:
            ans = input(
                f"  {color}run?{C.RESET} "
                f"[{C.GREEN}y{C.RESET}es / {C.YELLOW}n{C.RESET}o / "
                f"{C.CYAN}a{C.RESET}ll / {C.RED}q{C.RESET}uit] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "quit"
        if ans in ("y", "yes", ""):
            return "run"
        if ans in ("n", "no"):
            return "skip"
        if ans in ("a", "all"):
            return "all"
        if ans in ("q", "quit"):
            return "quit"
        print("  Please answer y, n, a, or q.")


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------


class Agent:
    def __init__(self, client, shell: Shell, mode: str, model: str):
        self.client = client
        self.shell = shell
        self.mode = mode  # 'smart' | 'ask-all' | 'yolo'
        self.model = model
        self.messages: list[dict] = []
        self._approve_all = False  # set when the user picks "all" this run

    # -- decide whether a command runs without asking ----------------------
    def _auto_ok(self, tier: str) -> bool:
        if self._approve_all or self.mode == "yolo":
            return True
        if self.mode == "ask-all":
            return False
        # smart mode: safe runs automatically; everything else asks
        return tier == "safe"

    def _execute(self, inp: dict) -> tuple[str, bool]:
        """Returns (tool_result_text, is_error)."""
        command = inp.get("command", "")
        explanation = inp.get("explanation", "")
        interactive = bool(inp.get("interactive", False))
        if INTERACTIVE_HINT.match(command):
            interactive = True

        tier, reason = classify_risk(command)

        if not self._auto_ok(tier):
            decision = confirm(command, tier, reason, explanation)
            if decision == "quit":
                raise KeyboardInterrupt
            if decision == "skip":
                return ("User declined to run this command. "
                        "Suggest an alternative or ask what they'd prefer.",
                        True)
            if decision == "all":
                self._approve_all = True
        else:
            color, label = RISK_STYLE[tier]
            print(f"\n  {color}[{label}]{C.RESET} {C.BOLD}$ {command}{C.RESET}")

        result = self.shell.run(command, interactive=interactive)

        if result["cwd_changed"]:
            print(f"  {C.DIM}↪ now in {self.shell.cwd}{C.RESET}")

        # Echo captured output to the user too, so they see what happened.
        if not interactive:
            out = (result["stdout"] or "").rstrip()
            err = (result["stderr"] or "").rstrip()
            if out:
                print(_indent(out))
            if err:
                print(_indent(err, color=C.RED))

        payload = (
            f"exit_code: {result['exit_code']}\n"
            f"cwd: {self.shell.cwd}\n"
            f"stdout:\n{result['stdout']}\n"
            f"stderr:\n{result['stderr']}"
        )
        return payload, result["exit_code"] != 0

    # -- one user turn -----------------------------------------------------
    def run_turn(self, user_text: str) -> None:
        self.messages.append({"role": "user", "content": user_text})

        while True:
            final = self._stream_once()
            self.messages.append({"role": "assistant", "content": final.content})

            if final.stop_reason != "tool_use":
                break

            tool_results = []
            for block in final.content:
                if block.type != "tool_use":
                    continue
                text, is_error = self._execute(block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": text,
                    "is_error": is_error,
                })

            self.messages.append({"role": "user", "content": tool_results})

    def _stream_once(self):
        context = (
            f"[environment] os={platform.system()} "
            f"({platform.release()}); shell={os.environ.get('SHELL', 'unknown')}; "
            f"cwd={self.shell.cwd}; home={os.path.expanduser('~')}"
        )
        printed_any = False
        with self.client.messages.stream(
            model=self.model,
            max_tokens=8000,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "text", "text": context},
            ],
            thinking={"type": "adaptive"},
            tools=[RUN_TOOL],
            messages=self.messages,
        ) as stream:
            for event in stream:
                if (event.type == "content_block_delta"
                        and event.delta.type == "text_delta"):
                    if not printed_any:
                        sys.stdout.write(f"\n{C.CYAN}")
                        printed_any = True
                    sys.stdout.write(event.delta.text)
                    sys.stdout.flush()
            if printed_any:
                sys.stdout.write(f"{C.RESET}\n")
                sys.stdout.flush()
            return stream.get_final_message()


def _indent(text: str, color: str = C.DIM) -> str:
    lines = text.splitlines()
    return "\n".join(f"  {color}│{C.RESET} {line}" for line in lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_client():
    try:
        return anthropic.Anthropic()
    except Exception as e:  # pragma: no cover
        sys.exit(f"Could not initialize the Anthropic client: {e}")


def make_agent(args) -> Agent:
    if not (os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        # The SDK can still resolve an `ant auth login` profile, so we only
        # warn rather than hard-fail.
        print(f"{C.DIM}Note: no ANTHROPIC_API_KEY set; relying on a saved "
              f"Anthropic login if you have one.{C.RESET}", file=sys.stderr)
    mode = "smart"
    if args.yolo:
        mode = "yolo"
    elif args.ask_all:
        mode = "ask-all"
    return Agent(build_client(), Shell(), mode, args.model)


def banner(mode: str):
    print(f"{C.MAGENTA}{C.BOLD}prompter{C.RESET} "
          f"{C.DIM}· natural-language shell agent · mode={mode}{C.RESET}")
    if mode == "yolo":
        print(f"{C.RED}{C.BOLD}⚠ YOLO mode: every command runs without "
              f"asking, including dangerous ones.{C.RESET}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prompter",
        description="Describe what you want; prompter runs the shell commands.",
    )
    parser.add_argument("prompt", nargs="*", help="What you want done.")
    parser.add_argument("--model", default=MODEL, help=f"Model (default {MODEL}).")
    parser.add_argument("--ask-all", action="store_true",
                        help="Confirm every command, even safe ones.")
    parser.add_argument("--yolo", action="store_true",
                        help="Run everything without confirmation (dangerous).")
    args = parser.parse_args(argv)

    agent = make_agent(args)
    banner(agent.mode)

    goal = " ".join(args.prompt).strip()
    try:
        if goal:
            agent.run_turn(goal)
        else:
            _repl(agent)
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Stopped.{C.RESET}")
        return 130
    except anthropic.AuthenticationError:
        print(f"{C.RED}Authentication failed.{C.RESET} Set ANTHROPIC_API_KEY "
              f"or run `ant auth login`.", file=sys.stderr)
        return 1
    except anthropic.APIError as e:
        print(f"{C.RED}API error:{C.RESET} {e}", file=sys.stderr)
        return 1
    return 0


def _repl(agent: Agent) -> None:
    print(f"{C.DIM}Interactive mode. Type a request, or 'exit' to quit. "
          f"Working dir: {agent.shell.cwd}{C.RESET}")
    while True:
        try:
            line = input(f"\n{C.MAGENTA}prompter{C.RESET} "
                         f"{C.DIM}{agent.shell.cwd}{C.RESET} ➜ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.lower() in ("exit", "quit", ":q"):
            return
        if not line:
            continue
        agent.run_turn(line)


if __name__ == "__main__":
    sys.exit(main())
