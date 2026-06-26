"""Command-line entry point: argument parsing, wiring, and dispatch."""

from __future__ import annotations

import argparse
import os
import sys

from .agent import Agent, QuitRequested
from .config import (
    ApprovalMode,
    CONFIG_PATH,
    Config,
    PROGRAM_NAME,
    default_model_for,
    load_config,
)
from .providers import ProviderAuthError, ProviderError, create_provider
from .shell import Shell
from .ui import Console

OK_EXIT_CODE = 0
ERROR_EXIT_CODE = 1
SIGINT_EXIT_CODE = 130

_DESCRIPTION = "Describe what you want; prompter runs the shell commands."
_EXIT_WORDS = {"exit", "quit", ":q"}
_MISSING_KEY_TEMPLATE = (
    "Note: {env} is not set; relying on a saved login if your provider has one."
)

_HELP_PROMPT = "What you want done."
_HELP_PROVIDER = "Model provider this run (anthropic, openai, gemini)."
_HELP_MODEL = "Model override (config 'model', else the provider default)."
_HELP_BASE_URL = "OpenAI-compatible base URL (e.g. Groq, OpenRouter)."
_HELP_WORKSPACE = "Override the default project workspace this run."
_HELP_MAX_FIX = "Max commands that may fail in a row before stopping."
_HELP_ASK_ALL = "Confirm every command, even safe ones."
_HELP_YOLO = "Run everything without confirmation (dangerous)."
_HELP_CONFIG = "Print the config file path and exit."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=_DESCRIPTION)
    parser.add_argument("prompt", nargs="*", help=_HELP_PROMPT)
    parser.add_argument("--provider", help=_HELP_PROVIDER)
    parser.add_argument("--model", default=None, help=_HELP_MODEL)
    parser.add_argument("--base-url", help=_HELP_BASE_URL)
    parser.add_argument("--workspace", help=_HELP_WORKSPACE)
    parser.add_argument("--max-fix", type=int, default=None, help=_HELP_MAX_FIX)
    parser.add_argument("--ask-all", action="store_true", help=_HELP_ASK_ALL)
    parser.add_argument("--yolo", action="store_true", help=_HELP_YOLO)
    parser.add_argument("--config", action="store_true", help=_HELP_CONFIG)
    return parser


def _apply_overrides(args, config: Config) -> None:
    """CLI flags override config values for this run; resolve the model."""
    if args.provider:
        config.provider = args.provider
    if args.base_url:
        config.base_url = args.base_url
    if args.workspace:
        config.default_workspace = args.workspace
    if args.max_fix is not None:
        config.max_fix_attempts = args.max_fix
    config.model = args.model or config.model or default_model_for(config.provider)


def _resolve_mode(args, config: Config) -> ApprovalMode:
    if args.yolo:
        return ApprovalMode.YOLO
    if args.ask_all or not config.auto_approve_safe:
        return ApprovalMode.ASK_ALL
    return ApprovalMode.SMART


def make_agent(args, config: Config, console: Console) -> Agent:
    _apply_overrides(args, config)
    if config.key_env and not os.environ.get(config.key_env):
        console.note(_MISSING_KEY_TEMPLATE.format(env=config.key_env))
    mode = _resolve_mode(args, config)
    try:
        provider = create_provider(config)
    except ProviderError as e:
        sys.exit(str(e))
    return Agent(provider, Shell(), console, config, mode)


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
    (ProviderAuthError, _handle_auth),
    (ProviderError, _handle_api),
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
