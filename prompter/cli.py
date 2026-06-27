"""Command-line entry point: a command dispatcher with actionable errors.

Management is done with subcommands (keys, use, status, config). Flags only
modify a single run. No command ever opens an interactive prompt. Every failure
prints a problem with the exact command to fix it.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

from .agent import Agent, QuitRequested
from .config import (
    ApprovalMode,
    CONFIG_PATH,
    Config,
    PROGRAM_NAME,
    PROVIDER_ALIASES,
    PROVIDER_OPENAI,
    default_api_key_env,
    default_model_for,
    load_config,
    normalize_provider,
    program_version,
    save_config,
    source_version,
)
from .completion import (
    CommandSurface,
    SUPPORTED_SHELLS,
    render as render_completion,
)
from .constants import COMMA_SPACE, EMPTY, SPACE
from .keys import KEYS_PATH, clear_key, set_key, stored_key
from .providers import (
    ModelProvider,
    ProviderAuthError,
    ProviderError,
    create_provider,
    known_providers,
)
from .shell import Shell
from .ui import Console

OK_EXIT_CODE = 0
ERROR_EXIT_CODE = 1
SIGINT_EXIT_CODE = 130

_CMD_KEYS = "keys"
_CMD_USE = "use"
_CMD_STATUS = "status"
_CMD_CONFIG = "config"
_CMD_RUN = "run"
_CMD_HELP = "help"
_CMD_COMPLETIONS = "completions"
_CMD_VERSION = "version"
_HELP_FLAGS = ("-h", "--help")
_VERSION_FLAGS = ("-V", "--version")
_VERSION_LINE = "{program} {version}"
_VERSION_DRIFT = (
    "note: pyproject.toml declares {source}, but the installed package is "
    "{installed}. Run `pip install -e .` to refresh."
)

_KEYS_LIST = "list"
_KEYS_ADD = "add"
_KEYS_REMOVE = ("remove", "rm", "clear")

_EXIT_WORDS = {"exit", "quit", ":q"}

_ARG_PROMPT = "prompt"
_ARG_NARGS_ANY = "*"
_FLAG_PROVIDER = "--provider"
_FLAG_MODEL = "--model"
_FLAG_BASE_URL = "--base-url"
_FLAG_WORKSPACE = "--workspace"
_FLAG_MAX_FIX = "--max-fix"
_FLAG_ASK_ALL = "--ask-all"
_FLAG_YOLO = "--yolo"
_FLAG_CONCISE = "--concise"
_FLAG_VERBOSE = "--verbose"
_ACTION_STORE_TRUE = "store_true"

_LABEL_ADD_IT = "add it"
_LABEL_OR_EXPORT = "or export"
_LABEL_DID_YOU_MEAN = "did you mean"
_LABEL_AVAILABLE = "available"
_LABEL_SWITCH_TO_IT = "switch to it"
_LABEL_FOR_ONE_RUN = "for one run"
_LABEL_SET_A_MODEL = "set a model"
_LABEL_USAGE = "usage"
_LABEL_EXAMPLE = "example"

_FIELD_PROVIDER = "provider"
_FIELD_MODEL = "model"
_FIELD_WORKSPACE = "workspace"
_FIELD_MAX_FIX = "max-fix"
_FIELD_CONFIG = "config"

_BASE_URL_IGNORED = (
    "Note: base_url only applies to the openai provider. Ignoring it for {provider}."
)

_TITLE_NO_KEY = "No API key for {provider}"
_CMD_ADD_KEY = "prompter keys add {provider} <key>"
_ENV_EXPORT = "{env}=<key>"

_TITLE_UNKNOWN_PROVIDER = 'Unknown provider "{name}"'
_HINT_USE = "prompter use {provider}"

_TITLE_MODEL_IS_PROVIDER = '"{value}" is a provider, not a model'
_CMD_RUN_PROVIDER = 'prompter --provider {provider} "..."'

_TITLE_BAD_MAX_FIX = "--max-fix must be at least 1 (got {value})"

_MODEL_WORD = "model"
_TITLE_NO_MODEL = '{provider} has no model "{model}"'
_CMD_USE_MODEL = "prompter use {provider} <model>"
_TITLE_PROVIDER_FAILED = "{provider} request failed"
_MODEL_ERROR_MARKERS = (
    "not found", "does not exist", "is not supported", "no model",
    "invalid model", "not a valid model",
)

_TITLE_UNKNOWN_KEYS_ACTION = 'Unknown "keys" command: "{action}"'
_USAGE_KEYS = "prompter keys [list | add <provider> <key> | remove <provider>]"
_TITLE_KEYS_ADD_USAGE = "Usage: prompter keys add <provider> <key>"
_EXAMPLE_KEYS_ADD = "prompter keys add gemini AIza..."
_TITLE_KEYS_REMOVE_USAGE = "Usage: prompter keys remove <provider>"
_KEY_SAVED = "Saved the {provider} key to {path}"
_KEY_REMOVED = "Removed the stored {provider} key"
_NO_STORED_KEY = "No stored key for {provider}"

_SOURCE_ENV = "set via {env}"
_SOURCE_STORED = "stored ({masked})"
_SOURCE_NOT_SET = "not set"
_MASK_TEMPLATE = "...{tail}"
_MASK_FALLBACK = "set"
_MASK_TAIL_LENGTH = 4

_TITLE_USE_USAGE = "Usage: prompter use <provider> [model]"
_EXAMPLE_USE = "prompter use gemini"
_DEFAULT_SET = "Default provider set to {provider} (model: {model})"

_TITLE_COMPLETIONS_USAGE = "Usage: prompter completions <shell>"
_EXAMPLE_COMPLETIONS = "prompter completions zsh"
_TITLE_UNKNOWN_SHELL = 'Unknown shell "{shell}"'
_LABEL_SUPPORTED = "supported"

_KEYS_HEADER = "keys:"

_HELP_TEXT = """\
prompter is a natural-language shell agent.

Run:
  prompter "<goal>"                 run a one-off goal
  prompter                          interactive chat (REPL)

Run flags (modify one run):
  --provider NAME                   anthropic, openai, or gemini
  --model ID                        model override
  --base-url URL                    OpenAI-compatible endpoint (Groq, OpenRouter)
  --workspace PATH                  where new projects go
  --max-fix N                       failures in a row before stopping
  --ask-all                         confirm every command
  --yolo                            run everything with no confirmation
  --concise                         no code comments, minimal explanation
  --verbose                         full comments and explanation (overrides config)

Manage:
  prompter keys add <provider> <key>    store an API key
  prompter keys list                    show key status
  prompter keys remove <provider>       delete a stored key
  prompter use <provider> [model]       set your default provider/model
  prompter status                       show the current setup
  prompter config                       print the config file path
  prompter completions <shell>          print a tab-completion script (zsh, bash)
  prompter version                      print the installed version
  prompter help                         show this help

Providers: anthropic, openai, gemini
"""


def _missing_key_problem(config: Config):
    return (
        _TITLE_NO_KEY.format(provider=config.provider),
        [(_LABEL_ADD_IT, _CMD_ADD_KEY.format(provider=config.provider)),
         (_LABEL_OR_EXPORT, _ENV_EXPORT.format(env=config.key_env))],
        None,
    )


def _unknown_provider_problem(name: str):
    """Build the unknown-provider problem, matching aliases for a suggestion.

    Candidates include the aliases (claude, gpt, google) so a near miss like
    "claud" suggests anthropic, normalized back to its canonical name.
    """
    known = known_providers()
    hints = []
    candidates = list(known) + list(PROVIDER_ALIASES)
    match = difflib.get_close_matches(name.lower(), candidates, n=1)
    if match:
        hints.append((_LABEL_DID_YOU_MEAN,
                      _HINT_USE.format(provider=normalize_provider(match[0]))))
    hints.append((_LABEL_AVAILABLE, COMMA_SPACE.join(known)))
    return (_TITLE_UNKNOWN_PROVIDER.format(name=name), hints, None)


def _model_is_provider_problem(value: str, provider: str):
    return (
        _TITLE_MODEL_IS_PROVIDER.format(value=value),
        [(_LABEL_SWITCH_TO_IT, _HINT_USE.format(provider=provider)),
         (_LABEL_FOR_ONE_RUN, _CMD_RUN_PROVIDER.format(provider=provider))],
        None,
    )


def _looks_like_model_error(message: str) -> bool:
    low = message.lower()
    return _MODEL_WORD in low and any(m in low for m in _MODEL_ERROR_MARKERS)


def _provider_request_problem(config: Config, exc: ProviderError):
    detail = str(exc)
    if _looks_like_model_error(detail):
        return (
            _TITLE_NO_MODEL.format(provider=config.provider, model=config.resolved_model),
            [(_LABEL_SET_A_MODEL, _CMD_USE_MODEL.format(provider=config.provider))],
            None,
        )
    return (_TITLE_PROVIDER_FAILED.format(provider=config.provider), [], detail)


class _CommandError(Exception):
    def __init__(self, title, hints=(), detail=None):
        super().__init__(title)
        self.title = title
        self.hints = hints
        self.detail = detail


def _fail(console: Console, title, hints=(), detail=None) -> int:
    console.problem(title, hints, detail)
    return ERROR_EXIT_CODE


def _mask(key: str) -> str:
    if len(key) >= _MASK_TAIL_LENGTH:
        return _MASK_TEMPLATE.format(tail=key[-_MASK_TAIL_LENGTH:])
    return _MASK_FALLBACK


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, add_help=False)
    parser.add_argument(_ARG_PROMPT, nargs=_ARG_NARGS_ANY)
    parser.add_argument(_FLAG_PROVIDER)
    parser.add_argument(_FLAG_MODEL, default=None)
    parser.add_argument(_FLAG_BASE_URL)
    parser.add_argument(_FLAG_WORKSPACE)
    parser.add_argument(_FLAG_MAX_FIX, type=int, default=None)
    parser.add_argument(_FLAG_ASK_ALL, action=_ACTION_STORE_TRUE)
    parser.add_argument(_FLAG_YOLO, action=_ACTION_STORE_TRUE)
    parser.add_argument(_FLAG_CONCISE, action=_ACTION_STORE_TRUE)
    parser.add_argument(_FLAG_VERBOSE, action=_ACTION_STORE_TRUE)
    return parser


def _reject_provider_as_model(args) -> None:
    if not args.model:
        return
    provider = normalize_provider(args.model)
    if provider in known_providers():
        raise _CommandError(*_model_is_provider_problem(args.model, provider))


def _apply_overrides(args, config: Config) -> None:
    """Apply CLI flag overrides for this run, then resolve the model.

    A model pinned in the config belongs to the provider it was set for.
    Switching providers falls back to the new provider's default model.
    """
    config_provider = normalize_provider(config.provider)
    if args.provider:
        config.provider = args.provider
    config.provider = normalize_provider(config.provider)
    if args.base_url:
        config.base_url = args.base_url
    if args.workspace:
        config.default_workspace = args.workspace
    if args.max_fix is not None:
        config.max_fix_attempts = args.max_fix
    if args.concise:
        config.concise = True
    if args.verbose:
        config.concise = False
    pinned = config.model if config_provider == config.provider else EMPTY
    config.model = args.model or pinned or default_model_for(config.provider)


def _resolve_mode(args, config: Config) -> ApprovalMode:
    if args.yolo:
        return ApprovalMode.YOLO
    if args.ask_all or not config.auto_approve_safe:
        return ApprovalMode.ASK_ALL
    return ApprovalMode.SMART


def _validate_config(config: Config, console: Console) -> None:
    if config.max_fix_attempts < 1:
        raise _CommandError(_TITLE_BAD_MAX_FIX.format(value=config.max_fix_attempts))
    if config.base_url and config.provider != PROVIDER_OPENAI:
        console.note(_BASE_URL_IGNORED.format(provider=config.provider))


def _create_provider_checked(config: Config) -> ModelProvider:
    if config.provider not in known_providers():
        raise _CommandError(*_unknown_provider_problem(config.provider))
    try:
        return create_provider(config)
    except ProviderError:
        raise _CommandError(*_missing_key_problem(config))


def _prepare(args, console: Console):
    config = load_config()
    _reject_provider_as_model(args)
    _apply_overrides(args, config)
    _validate_config(config, console)
    provider = _create_provider_checked(config)
    mode = _resolve_mode(args, config)
    return Agent(provider, Shell(), console, config, mode), config, mode


def _dispatch(agent: Agent, goal: str) -> None:
    if goal:
        agent.run_turn(goal)
    else:
        _repl(agent, agent.console)


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


def _run_agent(agent: Agent, config: Config, console: Console, goal: str) -> int:
    try:
        _dispatch(agent, goal)
    except (KeyboardInterrupt, QuitRequested):
        console.stopped()
        return SIGINT_EXIT_CODE
    except ProviderAuthError:
        return _fail(console, *_missing_key_problem(config))
    except ProviderError as exc:
        return _fail(console, *_provider_request_problem(config, exc))
    return OK_EXIT_CODE


def cmd_run(argv: list[str], console: Console) -> int:
    args = _run_parser().parse_args(argv)
    try:
        agent, config, mode = _prepare(args, console)
    except _CommandError as e:
        return _fail(console, e.title, e.hints, e.detail)
    console.banner(config, mode)
    return _run_agent(agent, config, console, SPACE.join(args.prompt).strip())


def _keys_list(console: Console) -> int:
    for provider in known_providers():
        env = default_api_key_env(provider)
        stored = stored_key(provider)
        if env and os.environ.get(env):
            console.key_status(provider, _SOURCE_ENV.format(env=env), True)
        elif stored:
            console.key_status(
                provider, _SOURCE_STORED.format(masked=_mask(stored)), True)
        else:
            console.key_status(provider, _SOURCE_NOT_SET, False)
    return OK_EXIT_CODE


def _keys_add(rest: list[str], console: Console) -> int:
    if len(rest) < 2:
        return _fail(console, _TITLE_KEYS_ADD_USAGE, [(_LABEL_EXAMPLE, _EXAMPLE_KEYS_ADD)])
    provider = normalize_provider(rest[0])
    if provider not in known_providers():
        return _fail(console, *_unknown_provider_problem(rest[0]))
    set_key(provider, rest[1])
    console.success(_KEY_SAVED.format(provider=provider, path=KEYS_PATH))
    return OK_EXIT_CODE


def _keys_remove(rest: list[str], console: Console) -> int:
    if not rest:
        return _fail(console, _TITLE_KEYS_REMOVE_USAGE)
    provider = normalize_provider(rest[0])
    if clear_key(provider):
        console.success(_KEY_REMOVED.format(provider=provider))
    else:
        console.info(_NO_STORED_KEY.format(provider=provider))
    return OK_EXIT_CODE


def cmd_keys(argv: list[str], console: Console) -> int:
    action = argv[0] if argv else _KEYS_LIST
    rest = argv[1:]
    if action == _KEYS_LIST:
        return _keys_list(console)
    if action == _KEYS_ADD:
        return _keys_add(rest, console)
    if action in _KEYS_REMOVE:
        return _keys_remove(rest, console)
    return _fail(console, _TITLE_UNKNOWN_KEYS_ACTION.format(action=action),
                 [(_LABEL_USAGE, _USAGE_KEYS)])


def cmd_use(argv: list[str], console: Console) -> int:
    if not argv:
        return _fail(console, _TITLE_USE_USAGE, [(_LABEL_EXAMPLE, _EXAMPLE_USE)])
    provider = normalize_provider(argv[0])
    if provider not in known_providers():
        return _fail(console, *_unknown_provider_problem(argv[0]))
    config = load_config()
    config.provider = provider
    config.model = argv[1] if len(argv) > 1 else EMPTY
    save_config(config)
    console.success(_DEFAULT_SET.format(provider=provider, model=config.resolved_model))
    return OK_EXIT_CODE


def cmd_status(argv: list[str], console: Console) -> int:
    config = load_config()
    console.field(_FIELD_PROVIDER, config.provider)
    console.field(_FIELD_MODEL, config.resolved_model)
    console.field(_FIELD_WORKSPACE, config.workspace_path)
    console.field(_FIELD_MAX_FIX, str(config.max_fix_attempts))
    console.field(_FIELD_CONFIG, CONFIG_PATH)
    console.info(_KEYS_HEADER)
    return _keys_list(console)


def cmd_config(argv: list[str], console: Console) -> int:
    load_config()
    console.info(CONFIG_PATH)
    return OK_EXIT_CODE


def cmd_completions(argv: list[str], console: Console) -> int:
    if not argv:
        return _fail(console, _TITLE_COMPLETIONS_USAGE,
                     [(_LABEL_EXAMPLE, _EXAMPLE_COMPLETIONS)])
    shell = argv[0].lower()
    if shell not in SUPPORTED_SHELLS:
        return _fail(console, _TITLE_UNKNOWN_SHELL.format(shell=argv[0]),
                     [(_LABEL_SUPPORTED, COMMA_SPACE.join(SUPPORTED_SHELLS))])
    console.info(render_completion(shell, command_surface()))
    return OK_EXIT_CODE


def cmd_help(console: Console) -> int:
    console.info(_HELP_TEXT)
    return OK_EXIT_CODE


def cmd_version(console: Console) -> int:
    installed = program_version()
    console.info(_VERSION_LINE.format(program=PROGRAM_NAME, version=installed))
    source = source_version()
    if source is not None and source != installed:
        console.note(_VERSION_DRIFT.format(source=source, installed=installed))
    return OK_EXIT_CODE


_COMMANDS = {
    _CMD_KEYS: cmd_keys,
    _CMD_USE: cmd_use,
    _CMD_STATUS: cmd_status,
    _CMD_CONFIG: cmd_config,
    _CMD_COMPLETIONS: cmd_completions,
}


def command_surface() -> CommandSurface:
    """Describe the command surface for shell completion, read from the live
    dispatcher and argument parser so completion never drifts from the CLI.

    Flags are split by whether the parser has them consume a value; the two that
    take a known kind of value are named so completion can suggest it.
    """
    bool_flags, value_flags = [], []
    for action in _run_parser()._actions:
        if not action.option_strings:
            continue
        (bool_flags if action.nargs == 0 else value_flags).extend(action.option_strings)
    return CommandSurface(
        commands=tuple(_COMMANDS) + (_CMD_HELP,),
        bool_flags=tuple(bool_flags),
        value_flags=tuple(value_flags),
        provider_flags=(_FLAG_PROVIDER,),
        dir_flags=(_FLAG_WORKSPACE,),
        keys_actions=(_KEYS_LIST, _KEYS_ADD, _KEYS_REMOVE[0]),
        keys_provider_actions=(_KEYS_ADD,) + _KEYS_REMOVE,
        providers=tuple(known_providers()),
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    console = Console()
    if not argv:
        return cmd_run(argv, console)
    head = argv[0]
    if head in _HELP_FLAGS or head == _CMD_HELP:
        return cmd_help(console)
    if head in _VERSION_FLAGS or head == _CMD_VERSION:
        return cmd_version(console)
    if head == _CMD_RUN:
        return cmd_run(argv[1:], console)
    handler = _COMMANDS.get(head)
    if handler is not None:
        return handler(argv[1:], console)
    return cmd_run(argv, console)
