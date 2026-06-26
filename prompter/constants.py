"""Named constants for the whole package.

Anything with semantic meaning that the code compares against, keys into, or
limits by lives here, so no module hard-codes a bare literal. Anthropic
Messages API protocol strings, runtime limits (the "magic numbers"), process
exit codes, and environment-variable names are grouped below.
"""

from __future__ import annotations

DEFAULT_MODEL = "claude-sonnet-4-6"

# -- Anthropic Messages API protocol ----------------------------------------
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

BLOCK_TEXT = "text"
BLOCK_TOOL_USE = "tool_use"
BLOCK_TOOL_RESULT = "tool_result"

STOP_REASON_TOOL_USE = "tool_use"

EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
DELTA_TEXT = "text_delta"

THINKING_ADAPTIVE = {"type": "adaptive"}
TOOL_CHOICE_NONE = {"type": "none"}

TOOL_NAME_RUN_COMMAND = "run_command"
INPUT_COMMAND = "command"
INPUT_EXPLANATION = "explanation"
INPUT_INTERACTIVE = "interactive"

# -- Runtime limits ("magic numbers") ---------------------------------------
COMMAND_TIMEOUT_SECONDS = 600
TIMEOUT_EXIT_CODE = 124
MAX_OUTPUT_CHARS = 20000
MAX_TOKENS = 8000
DEFAULT_MAX_FIX_ATTEMPTS = 3

# -- Process exit codes ------------------------------------------------------
OK_EXIT_CODE = 0
ERROR_EXIT_CODE = 1
SIGINT_EXIT_CODE = 130

# -- Environment variables ---------------------------------------------------
ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_AUTH_TOKEN = "ANTHROPIC_AUTH_TOKEN"
ENV_SHELL = "SHELL"
ENV_NO_COLOR = "NO_COLOR"

# -- Shell-execution internals ----------------------------------------------
CWD_SENTINEL = "__PROMPTER_CWD_a3f9__:"
