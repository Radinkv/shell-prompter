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
from .config import ApprovalMode, CONFIG_PATH, Config, load_config
from .constants import (
    DEFAULT_MODEL,
    ENV_API_KEY,
    ENV_AUTH_TOKEN,
    ERROR_EXIT_CODE,
    OK_EXIT_CODE,
    SIGINT_EXIT_CODE,
)
from .llm import ClaudeClient
from .shell import Shell
from .ui import Console

_DESCRIPTION = "Describe what you want; prompter runs the shell commands."
_EXIT_WORDS = {"exit", "quit", ":q"}
_MISSING_KEY_NOTE = (
    "Note: no ANTHROPIC_API_KEY set; relying on a saved "
    "Anthropic login if you have one."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prompter", description=_DESCRIPTION)
    parser.add_argument("prompt", nargs="*", help="What you want done.")
    parser.add_argument("--model", default=None,
                        help=f"Model override (config 'model', else {DEFAULT_MODEL}).")
    parser.add_argument("--workspace",
                        help="Override the default project workspace this run.")
    parser.add_argument("--max-fix", type=int, default=None,
                        help="Max commands that may fail in a row before stopping.")
    parser.add_argument("--ask-all", action="store_true",
                        help="Confirm every command, even safe ones.")
    parser.add_argument("--yolo", action="store_true",
                        help="Run everything without confirmation (dangerous).")
    parser.add_argument("--config", action="store_true",
                        help="Print the config file path and exit.")
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
        sys.exit(f"Could not initialize the Anthropic client: {e}")


def make_agent(args, config: Config, console: Console) -> Agent:
    if not (os.environ.get(ENV_API_KEY) or os.environ.get(ENV_AUTH_TOKEN)):
        console.note(_MISSING_KEY_NOTE)
    _apply_overrides(args, config)
    mode = _resolve_mode(args, config)
    client = ClaudeClient(build_client(), config.model)
    return Agent(client, Shell(), console, config, mode)


def run(args, console: Console) -> int:
    config = load_config()
    agent = make_agent(args, config, console)
    console.banner(config, agent.mode)

    goal = " ".join(args.prompt).strip()
    try:
        if goal:
            agent.run_turn(goal)
        else:
            _repl(agent, console)
    except (KeyboardInterrupt, QuitRequested):
        console.stopped()
        return SIGINT_EXIT_CODE
    except anthropic.AuthenticationError:
        console.auth_error()
        return ERROR_EXIT_CODE
    except anthropic.APIError as e:
        console.api_error(e)
        return ERROR_EXIT_CODE
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
