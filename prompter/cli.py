"""Command-line entry point: argument parsing, wiring, and dispatch."""

from __future__ import annotations

import argparse
import os
import sys

try:
    import anthropic
except ImportError:
    sys.exit(
        "The 'anthropic' package is required.\n"
        "Install it with:  pip install anthropic\n"
        "(or reinstall prompter with:  pip install -e .)"
    )

from .agent import Agent, QuitRequested
from .config import (
    ApprovalMode,
    CONFIG_PATH,
    Config,
    DEFAULT_MODEL,
    PROGRAM_NAME,
    load_config,
)
from .llm import ClaudeClient
from .shell import Shell
from .ui import Console

OK_EXIT_CODE = 0
ERROR_EXIT_CODE = 1
SIGINT_EXIT_CODE = 130

ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_AUTH_TOKEN = "ANTHROPIC_AUTH_TOKEN"

_DESCRIPTION = "Describe what you want; prompter runs the shell commands."
_EXIT_WORDS = {"exit", "quit", ":q"}
_MISSING_KEY_NOTE = (
    "Note: no ANTHROPIC_API_KEY set; relying on a saved "
    "Anthropic login if you have one."
)
_CLIENT_INIT_ERROR = "Could not initialize the Anthropic client: {error}"

_HELP_PROMPT = "What you want done."
_HELP_MODEL = f"Model override (config 'model', else {DEFAULT_MODEL})."
_HELP_WORKSPACE = "Override the default project workspace this run."
_HELP_MAX_FIX = "Max commands that may fail in a row before stopping."
_HELP_ASK_ALL = "Confirm every command, even safe ones."
_HELP_YOLO = "Run everything without confirmation (dangerous)."
_HELP_CONFIG = "Print the config file path and exit."

_ARG_PROMPT = "prompt"
_FLAG_MODEL = "--model"
_FLAG_WORKSPACE = "--workspace"
_FLAG_MAX_FIX = "--max-fix"
_FLAG_ASK_ALL = "--ask-all"
_FLAG_YOLO = "--yolo"
_FLAG_CONFIG = "--config"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=_DESCRIPTION)
    parser.add_argument(_ARG_PROMPT, nargs="*", help=_HELP_PROMPT)
    parser.add_argument(_FLAG_MODEL, default=None, help=_HELP_MODEL)
    parser.add_argument(_FLAG_WORKSPACE, help=_HELP_WORKSPACE)
    parser.add_argument(_FLAG_MAX_FIX, type=int, default=None, help=_HELP_MAX_FIX)
    parser.add_argument(_FLAG_ASK_ALL, action="store_true", help=_HELP_ASK_ALL)
    parser.add_argument(_FLAG_YOLO, action="store_true", help=_HELP_YOLO)
    parser.add_argument(_FLAG_CONFIG, action="store_true", help=_HELP_CONFIG)
    return parser


def _apply_overrides(args, config: Config) -> None:
    """CLI flags override config values for this run."""
    if args.workspace:
        config.default_workspace = args.workspace
    if args.max_fix is not None:
        config.max_fix_attempts = args.max_fix
    config.model = args.model or config.model or DEFAULT_MODEL


def _resolve_mode(args, config: Config) -> ApprovalMode:
    if args.yolo:
        return ApprovalMode.YOLO
    if args.ask_all or not config.auto_approve_safe:
        return ApprovalMode.ASK_ALL
    return ApprovalMode.SMART


def build_client():
    try:
        return anthropic.Anthropic()
    except Exception as e:
        sys.exit(_CLIENT_INIT_ERROR.format(error=e))


def make_agent(args, config: Config, console: Console) -> Agent:
    if not (os.environ.get(ENV_API_KEY) or os.environ.get(ENV_AUTH_TOKEN)):
        console.note(_MISSING_KEY_NOTE)
    _apply_overrides(args, config)
    mode = _resolve_mode(args, config)
    client = ClaudeClient(build_client(), config.model)
    return Agent(client, Shell(), console, config, mode)


def _handle_interrupt(console: Console, _exc: BaseException) -> int:
    console.stopped()
    return SIGINT_EXIT_CODE


def _handle_auth(console: Console, _exc: BaseException) -> int:
    console.auth_error()
    return ERROR_EXIT_CODE


def _handle_api(console: Console, exc: BaseException) -> int:
    console.api_error(exc)
    return ERROR_EXIT_CODE


_ERROR_HANDLERS = [
    ((KeyboardInterrupt, QuitRequested), _handle_interrupt),
    (anthropic.AuthenticationError, _handle_auth),
    (anthropic.APIError, _handle_api),
]


def _dispatch(agent: Agent, console: Console, goal: str) -> None:
    if goal:
        agent.run_turn(goal)
    else:
        _repl(agent, console)


def run(args, console: Console) -> int:
    config = load_config()
    agent = make_agent(args, config, console)
    console.banner(config, agent.mode)
    goal = " ".join(args.prompt).strip()
    try:
        _dispatch(agent, console, goal)
    except BaseException as exc:
        for exc_types, handler in _ERROR_HANDLERS:
            if isinstance(exc, exc_types):
                return handler(console, exc)
        raise
    return OK_EXIT_CODE


def _repl(agent: Agent, console: Console) -> None:
    console.repl_intro(agent.shell.cwd)
    while True:
        try:
            line = console.repl_prompt(agent.shell.cwd)
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.lower() in _EXIT_WORDS:
            return
        if not line:
            continue
        agent.run_turn(line)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config:
        load_config()
        print(CONFIG_PATH)
        return OK_EXIT_CODE
    return run(args, Console())
