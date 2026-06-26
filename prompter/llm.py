"""Everything we send to Claude: the tool schema, system prompt, per-turn
environment context, and a thin streaming client wrapper.

The wrapper keeps the Anthropic request shape in one place and reports streamed
text through an ``on_text`` callback, so the agent never touches API field names
or printing directly.
"""

from __future__ import annotations

import os
import platform
from typing import Callable

from .config import Config
from .constants import (
    DELTA_TEXT,
    EVENT_CONTENT_BLOCK_DELTA,
    FIELD_TEXT,
    FIELD_TYPE,
    INPUT_COMMAND,
    INPUT_EXPLANATION,
    INPUT_INTERACTIVE,
    BLOCK_TEXT,
    THINKING_ADAPTIVE,
    TOOL_CHOICE_NONE,
    TOOL_NAME_RUN_COMMAND,
)

MAX_TOKENS = 8000
ENV_SHELL = "SHELL"

_TOOL_DESCRIPTION = (
    "Run a single shell command in the user's terminal session. The working "
    "directory persists between calls (so `cd` works as expected). Use one "
    "command per call and build up to the goal step by step. Set `interactive` "
    "to true for programs that take over the terminal and need the user to type "
    "into them (claude, vim, ssh, a REPL, etc.) — their output is shown to the "
    "user directly and not returned to you."
)
_DESC_COMMAND = "The exact shell command to run."
_DESC_EXPLANATION = "One short sentence on what this does and why."
_DESC_INTERACTIVE = "True if the program needs an interactive terminal."

_SCHEMA_TYPE = "type"
_SCHEMA_OBJECT = "object"
_SCHEMA_STRING = "string"
_SCHEMA_BOOLEAN = "boolean"
_SCHEMA_DESCRIPTION = "description"
_SCHEMA_PROPERTIES = "properties"
_SCHEMA_REQUIRED = "required"
_TOOL_KEY_NAME = "name"
_TOOL_KEY_DESCRIPTION = "description"
_TOOL_KEY_INPUT_SCHEMA = "input_schema"


def _string_property(description: str) -> dict:
    return {_SCHEMA_TYPE: _SCHEMA_STRING, _SCHEMA_DESCRIPTION: description}


def _boolean_property(description: str) -> dict:
    return {_SCHEMA_TYPE: _SCHEMA_BOOLEAN, _SCHEMA_DESCRIPTION: description}


RUN_TOOL = {
    _TOOL_KEY_NAME: TOOL_NAME_RUN_COMMAND,
    _TOOL_KEY_DESCRIPTION: _TOOL_DESCRIPTION,
    _TOOL_KEY_INPUT_SCHEMA: {
        _SCHEMA_TYPE: _SCHEMA_OBJECT,
        _SCHEMA_PROPERTIES: {
            INPUT_COMMAND: _string_property(_DESC_COMMAND),
            INPUT_EXPLANATION: _string_property(_DESC_EXPLANATION),
            INPUT_INTERACTIVE: _boolean_property(_DESC_INTERACTIVE),
        },
        _SCHEMA_REQUIRED: [INPUT_COMMAND, INPUT_EXPLANATION],
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
- When the user asks to create a project or folder and doesn't say where, put \
it under the default workspace shown in the environment block (create the \
workspace with `mkdir -p` first if needed) — not the current directory.
- Follow the user preferences in the environment block (e.g. preferred compiler \
or language standard) when they apply.
- If a command fails, read the actual error and try a sensible different fix; \
do not re-run the identical failing command. If the same step keeps failing \
after a few attempts, stop and explain what's wrong rather than looping forever.
- Be careful with destructive or privileged actions. The user's harness gates \
risky commands behind a confirmation prompt, so don't be surprised if a command \
comes back as "declined by user" — adapt and propose an alternative or ask.
- Don't fabricate results. Base what you report on actual command output.
- Keep your prose brief. A sentence of context before a command and a short \
summary at the end is plenty; the user can see the commands and their output.
- When the goal is achieved, say so concisely and stop calling tools."""

_UNKNOWN_SHELL = "unknown"
_NO_PREFERENCES = "(none)"
_PREFERENCE_LINE = "- {item}"
_ENVIRONMENT_TEMPLATE = (
    "[environment] os={system} ({release}); shell={shell}; "
    "cwd={cwd}; home={home}\n"
    "[default workspace] {workspace}  (use for new projects when the user "
    "doesn't say where; mkdir -p it if missing)\n"
    "[user preferences]\n{preferences}"
)


def build_environment_context(config: Config, cwd: str) -> str:
    """The per-turn context block: OS, shell, cwd, workspace, preferences."""
    prefs = config.preferences or []
    prefs_text = (
        "\n".join(_PREFERENCE_LINE.format(item=p) for p in prefs)
        if prefs else _NO_PREFERENCES
    )
    return _ENVIRONMENT_TEMPLATE.format(
        system=platform.system(),
        release=platform.release(),
        shell=os.environ.get(ENV_SHELL, _UNKNOWN_SHELL),
        cwd=cwd,
        home=os.path.expanduser("~"),
        workspace=config.workspace_path,
        preferences=prefs_text,
    )


def build_system_blocks(context: str) -> list[dict]:
    return [
        {FIELD_TYPE: BLOCK_TEXT, FIELD_TEXT: SYSTEM_PROMPT},
        {FIELD_TYPE: BLOCK_TEXT, FIELD_TEXT: context},
    ]


class ClaudeClient:
    """Thin wrapper over the Anthropic client for the streaming agent loop."""

    def __init__(self, client, model: str):
        self._client = client
        self.model = model

    def stream(
        self,
        messages: list[dict],
        system_blocks: list[dict],
        disable_tools: bool,
        on_text: Callable[[str], None],
    ):
        """Stream one assistant turn, reporting text deltas via on_text."""
        kwargs = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system_blocks,
            "thinking": THINKING_ADAPTIVE,
            "tools": [RUN_TOOL],
            "messages": messages,
        }
        if disable_tools:
            kwargs["tool_choice"] = TOOL_CHOICE_NONE

        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                if (event.type == EVENT_CONTENT_BLOCK_DELTA
                        and event.delta.type == DELTA_TEXT):
                    on_text(event.delta.text)
            return stream.get_final_message()
