"""Anthropic Messages API vocabulary.

The exact field names, roles, block types, event types, and payload shapes the
API defines. The only strings shared across modules. Operational values (limits,
timeouts, exit codes, env-var names) and human-facing text live locally in the
module that owns them, not here.
"""

from __future__ import annotations

# -- message roles -----------------------------------------------------------
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# -- message / block field keys ---------------------------------------------
FIELD_ROLE = "role"
FIELD_CONTENT = "content"
FIELD_TYPE = "type"
FIELD_TEXT = "text"
FIELD_TOOL_USE_ID = "tool_use_id"
FIELD_IS_ERROR = "is_error"

# -- content block types -----------------------------------------------------
BLOCK_TEXT = "text"
BLOCK_TOOL_USE = "tool_use"
BLOCK_TOOL_RESULT = "tool_result"

# -- stop reasons ------------------------------------------------------------
STOP_REASON_TOOL_USE = "tool_use"

# -- streaming event / delta types -------------------------------------------
EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
DELTA_TEXT = "text_delta"

# -- request payload fragments -----------------------------------------------
THINKING_ADAPTIVE = {"type": "adaptive"}
TOOL_CHOICE_NONE = {"type": "none"}

# -- the run_command tool ----------------------------------------------------
TOOL_NAME_RUN_COMMAND = "run_command"
INPUT_COMMAND = "command"
INPUT_EXPLANATION = "explanation"
INPUT_INTERACTIVE = "interactive"
