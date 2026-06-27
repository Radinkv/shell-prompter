"""Anthropic (Claude) provider adapter.

Supplies the two provider-specific steps of the template method: render neutral
history into the Messages API shape, and stream a turn with the run_command tool
and adaptive thinking. The Anthropic wire vocabulary lives here. No other module
needs it.
"""

from __future__ import annotations

import anthropic

from ..config import PROVIDER_ANTHROPIC, Config
from ..keys import resolve_api_key
from .base import (
    PARAM_COMMAND,
    PARAM_EXPLANATION,
    PARAM_INTERACTIVE,
    RUN_COMMAND_PARAMETERS,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    AssistantMessage,
    HistoryItem,
    ModelProvider,
    ProviderAuthError,
    ProviderError,
    TurnCollector,
    ToolResultsMessage,
    Usage,
    UserMessage,
    register,
)

MAX_TOKENS = 8000

_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"
_FIELD_ROLE = "role"
_FIELD_CONTENT = "content"
_FIELD_TYPE = "type"
_FIELD_TEXT = "text"
_FIELD_NAME = "name"
_FIELD_INPUT = "input"
_FIELD_ID = "id"
_FIELD_TOOL_USE_ID = "tool_use_id"
_FIELD_IS_ERROR = "is_error"
_BLOCK_TEXT = "text"
_BLOCK_TOOL_USE = "tool_use"
_BLOCK_TOOL_RESULT = "tool_result"
_EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
_DELTA_TEXT = "text_delta"
_THINKING_VALUE_ADAPTIVE = "adaptive"
_TOOL_CHOICE_VALUE_NONE = "none"
_THINKING_ADAPTIVE = {_FIELD_TYPE: _THINKING_VALUE_ADAPTIVE}
_TOOL_CHOICE_NONE = {_FIELD_TYPE: _TOOL_CHOICE_VALUE_NONE}

_TOOL_KEY_NAME = "name"
_TOOL_KEY_DESCRIPTION = "description"
_TOOL_KEY_INPUT_SCHEMA = "input_schema"

_CACHE_CONTROL = "cache_control"
_CACHE_EPHEMERAL = {"type": "ephemeral"}

_ATTR_USAGE = "usage"
_ATTR_INPUT_TOKENS = "input_tokens"
_ATTR_OUTPUT_TOKENS = "output_tokens"
_ATTR_CACHE_READ = "cache_read_input_tokens"
_ATTR_CACHE_CREATION = "cache_creation_input_tokens"

_REQ_MODEL = "model"
_REQ_MAX_TOKENS = "max_tokens"
_REQ_SYSTEM = "system"
_REQ_THINKING = "thinking"
_REQ_TOOLS = "tools"
_REQ_MESSAGES = "messages"
_REQ_TOOL_CHOICE = "tool_choice"

_CLIENT_INIT_ERROR = "Could not initialize the Anthropic client: {error}"
_MISSING_AUTH_HINT = "authentication"

_RUN_TOOL = {
    _TOOL_KEY_NAME: TOOL_NAME,
    _TOOL_KEY_DESCRIPTION: TOOL_DESCRIPTION,
    _TOOL_KEY_INPUT_SCHEMA: RUN_COMMAND_PARAMETERS,
}


def _system_blocks(system_texts: list[str]) -> list[dict]:
    """Render system texts as blocks, caching the stable prefix.

    build_system_texts puts the stable prompt first and the volatile per-turn
    environment last. A cache_control breakpoint on the last stable block marks
    that prefix. Since tools precede system in the request (the tool
    definitions too), only the small environment block is re-sent uncached.


    Anthropic only caches a prefix past a minimum size (~1024 tokens for Sonnet).
    Today's prefix is under that, so this is a correctly-placed no-op that starts
    paying off if the system prompt or tool set grows. It stays compaction-safe
    because it caches only the fixed system prefix, never the churning history.
    """
    blocks = [{_FIELD_TYPE: _BLOCK_TEXT, _FIELD_TEXT: t} for t in system_texts]
    if len(blocks) >= 2:
        blocks[-2][_CACHE_CONTROL] = _CACHE_EPHEMERAL
    return blocks


def _usage_from(usage) -> Usage:
    """Anthropic reports cached and freshly-read input separately; sum them for
    the full prompt size and keep the cache-read portion."""
    fresh = getattr(usage, _ATTR_INPUT_TOKENS, 0) or 0
    cached = getattr(usage, _ATTR_CACHE_READ, 0) or 0
    created = getattr(usage, _ATTR_CACHE_CREATION, 0) or 0
    return Usage(
        input_tokens=fresh + cached + created,
        output_tokens=getattr(usage, _ATTR_OUTPUT_TOKENS, 0) or 0,
        cached_tokens=cached,
    )


def _to_messages(history: list[HistoryItem]) -> list[dict]:
    messages = []
    for item in history:
        if isinstance(item, UserMessage):
            messages.append({_FIELD_ROLE: _ROLE_USER, _FIELD_CONTENT: item.text})
        elif isinstance(item, AssistantMessage):
            messages.append({_FIELD_ROLE: _ROLE_ASSISTANT,
                             _FIELD_CONTENT: _assistant_content(item)})
        elif isinstance(item, ToolResultsMessage):
            messages.append({_FIELD_ROLE: _ROLE_USER,
                             _FIELD_CONTENT: [_result_block(r) for r in item.results]})
    return messages


def _assistant_content(item: AssistantMessage) -> list[dict]:
    blocks = []
    if item.text:
        blocks.append({_FIELD_TYPE: _BLOCK_TEXT, _FIELD_TEXT: item.text})
    for call in item.tool_calls:
        blocks.append({
            _FIELD_TYPE: _BLOCK_TOOL_USE,
            _FIELD_ID: call.call_id,
            _FIELD_NAME: TOOL_NAME,
            _FIELD_INPUT: {
                PARAM_COMMAND: call.command,
                PARAM_EXPLANATION: call.explanation,
                PARAM_INTERACTIVE: call.interactive,
            },
        })
    return blocks or [{_FIELD_TYPE: _BLOCK_TEXT, _FIELD_TEXT: item.text}]


def _result_block(result) -> dict:
    return {
        _FIELD_TYPE: _BLOCK_TOOL_RESULT,
        _FIELD_TOOL_USE_ID: result.call_id,
        _FIELD_CONTENT: result.content,
        _FIELD_IS_ERROR: result.is_error,
    }


class AnthropicProvider(ModelProvider):
    name = PROVIDER_ANTHROPIC
    auth_errors = (anthropic.AuthenticationError,)
    api_errors = (anthropic.APIError,)
    transient_errors = (anthropic.APITimeoutError, anthropic.APIConnectionError)

    def __init__(self, client, model: str):
        super().__init__(model)
        self._client = client

    def build_request(self, history, system_texts, disable_tools) -> dict:
        request = {
            _REQ_MODEL: self.model,
            _REQ_MAX_TOKENS: MAX_TOKENS,
            _REQ_SYSTEM: _system_blocks(system_texts),
            _REQ_THINKING: _THINKING_ADAPTIVE,
            _REQ_TOOLS: [_RUN_TOOL],
            _REQ_MESSAGES: _to_messages(history),
        }
        if disable_tools:
            request[_REQ_TOOL_CHOICE] = _TOOL_CHOICE_NONE
        return request

    def run_stream(self, request, collector: TurnCollector) -> None:
        """Stream a turn into the collector.

        The Anthropic client constructs lazily and raises a client-side
        TypeError at request time when no key, token, or profile can be
        resolved. That case is treated as an auth failure.
        """
        try:
            with self._client.messages.stream(**request) as stream:
                for event in stream:
                    if (event.type == _EVENT_CONTENT_BLOCK_DELTA
                            and event.delta.type == _DELTA_TEXT):
                        collector.add_text(event.delta.text)
                final = stream.get_final_message()
        except TypeError as e:
            if _MISSING_AUTH_HINT in str(e).lower():
                raise ProviderAuthError(str(e)) from e
            raise
        for block in final.content:
            if block.type == _BLOCK_TOOL_USE:
                collector.add_tool_call(block.id, block.input or {})
        usage = getattr(final, _ATTR_USAGE, None)
        if usage is not None:
            collector.set_usage(_usage_from(usage))


@register(PROVIDER_ANTHROPIC)
def build(config: Config) -> AnthropicProvider:
    api_key = resolve_api_key(config)
    try:
        client = (anthropic.Anthropic(api_key=api_key) if api_key
                  else anthropic.Anthropic())
    except Exception as e:
        raise ProviderError(_CLIENT_INIT_ERROR.format(error=e)) from e
    return AnthropicProvider(client, config.model)
