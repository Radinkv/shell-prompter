"""Cohere (Command) provider adapter.

Uses Cohere's v2 Chat API with function tools and streaming. Supplies the
template method's two steps: build_request renders neutral history into the v2
``messages`` shape, and run_stream accumulates streamed text and tool-call
deltas into the collector.

Cohere's v2 wire vocabulary is close to OpenAI's -- chat messages with
system/user/assistant/tool roles, and tools described as standard JSON Schema --
so the shared RUN_COMMAND_PARAMETERS shape is reused unchanged. The difference
lives in the stream: v2 emits typed events (content-delta, tool-call-start,
tool-call-delta, message-end) carrying a nested ``delta.message``, and each tool
call is keyed by an event ``index`` whose argument string arrives in fragments.

Usage rides on the final message-end event: token counts under ``usage.tokens``
and the cache-served portion in ``usage.cached_tokens``.
"""

from __future__ import annotations

import json

from ..config import PROVIDER_COHERE, Config
from ..constants import EMPTY
from ..keys import resolve_api_key
from .base import (
    PARAM_COMMAND,
    PARAM_EXPLANATION,
    PARAM_INTERACTIVE,
    RUN_COMMAND_PARAMETERS,
    SYSTEM_TEXT_SEPARATOR,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    AssistantMessage,
    HistoryItem,
    ModelProvider,
    ProviderError,
    TurnCollector,
    ToolResultsMessage,
    Usage,
    UserMessage,
    fallback_call_id,
    import_optional,
    register,
)

_ROLE_SYSTEM = "system"
_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"
_ROLE_TOOL = "tool"
_FUNCTION_TYPE = "function"

_TOOL_KEY_TYPE = "type"
_TOOL_KEY_FUNCTION = "function"
_TOOL_KEY_NAME = "name"
_TOOL_KEY_DESCRIPTION = "description"
_TOOL_KEY_PARAMETERS = "parameters"

_FIELD_ROLE = "role"
_FIELD_CONTENT = "content"
_FIELD_TOOL_CALL_ID = "tool_call_id"
_FIELD_TOOL_CALLS = "tool_calls"
_FIELD_ID = "id"
_FIELD_ARGUMENTS = "arguments"

_SLOT_ID = "id"
_SLOT_ARGS = "args"

_REQ_MODEL = "model"
_REQ_MESSAGES = "messages"
_REQ_TOOLS = "tools"

_EVENT_CONTENT_DELTA = "content-delta"
_EVENT_TOOL_CALL_START = "tool-call-start"
_EVENT_TOOL_CALL_DELTA = "tool-call-delta"
_EVENT_MESSAGE_END = "message-end"

_ATTR_TYPE = "type"
_ATTR_INDEX = "index"
_ATTR_DELTA = "delta"
_ATTR_MESSAGE = "message"
_ATTR_CONTENT = "content"
_ATTR_TEXT = "text"
_ATTR_TOOL_CALLS = "tool_calls"
_ATTR_FUNCTION = "function"
_ATTR_USAGE = "usage"
_ATTR_TOKENS = "tokens"
_ATTR_INPUT_TOKENS = "input_tokens"
_ATTR_OUTPUT_TOKENS = "output_tokens"
_ATTR_CACHED_TOKENS = "cached_tokens"

_MODULE_COHERE = "cohere"
_KW_API_KEY = "api_key"

_CLIENT_INIT_ERROR = "Could not initialize the Cohere client: {error}"

_FUNCTION_TOOL = {
    _TOOL_KEY_TYPE: _FUNCTION_TYPE,
    _TOOL_KEY_FUNCTION: {
        _TOOL_KEY_NAME: TOOL_NAME,
        _TOOL_KEY_DESCRIPTION: TOOL_DESCRIPTION,
        _TOOL_KEY_PARAMETERS: RUN_COMMAND_PARAMETERS,
    },
}


def _to_messages(system_texts: list[str], history: list[HistoryItem]) -> list[dict]:
    messages = [{_FIELD_ROLE: _ROLE_SYSTEM,
                 _FIELD_CONTENT: SYSTEM_TEXT_SEPARATOR.join(system_texts)}]
    for item in history:
        if isinstance(item, UserMessage):
            messages.append({_FIELD_ROLE: _ROLE_USER, _FIELD_CONTENT: item.text})
        elif isinstance(item, AssistantMessage):
            messages.append(_assistant_message(item))
        elif isinstance(item, ToolResultsMessage):
            for result in item.results:
                messages.append({
                    _FIELD_ROLE: _ROLE_TOOL,
                    _FIELD_TOOL_CALL_ID: result.call_id,
                    _FIELD_CONTENT: result.content,
                })
    return messages


def _assistant_message(item: AssistantMessage) -> dict:
    message = {_FIELD_ROLE: _ROLE_ASSISTANT, _FIELD_CONTENT: item.text or None}
    if item.tool_calls:
        message[_FIELD_TOOL_CALLS] = [{
            _FIELD_ID: call.call_id,
            _TOOL_KEY_TYPE: _FUNCTION_TYPE,
            _TOOL_KEY_FUNCTION: {
                _TOOL_KEY_NAME: TOOL_NAME,
                _FIELD_ARGUMENTS: json.dumps({
                    PARAM_COMMAND: call.command,
                    PARAM_EXPLANATION: call.explanation,
                    PARAM_INTERACTIVE: call.interactive,
                }),
            },
        } for call in item.tool_calls]
    return message


def _usage_from(usage) -> Usage:
    """v2 reports counts as floats under usage.tokens (input is the whole prompt),
    with the cache-served portion alongside in usage.cached_tokens."""
    tokens = getattr(usage, _ATTR_TOKENS, None)
    return Usage(
        input_tokens=int(getattr(tokens, _ATTR_INPUT_TOKENS, 0) or 0),
        output_tokens=int(getattr(tokens, _ATTR_OUTPUT_TOKENS, 0) or 0),
        cached_tokens=int(getattr(usage, _ATTR_CACHED_TOKENS, 0) or 0),
    )


def _parse_args(raw: str) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {}


def _event_message(event):
    """The ``delta.message`` an event carries, or None."""
    delta = getattr(event, _ATTR_DELTA, None)
    return getattr(delta, _ATTR_MESSAGE, None)


def _content_text(message) -> str | None:
    content = getattr(message, _ATTR_CONTENT, None)
    return getattr(content, _ATTR_TEXT, None)


def _accumulate_tool_event(event, message, accumulated: dict) -> None:
    """Fold a tool-call-start or tool-call-delta into the per-index slot.

    The start event carries the call id and function name; subsequent delta
    events stream the JSON argument string in fragments keyed by the same index.
    """
    call = getattr(message, _ATTR_TOOL_CALLS, None)
    if call is None:
        return
    index = getattr(event, _ATTR_INDEX, 0) or 0
    slot = accumulated.setdefault(index, {_SLOT_ID: EMPTY, _SLOT_ARGS: EMPTY})
    if getattr(call, _FIELD_ID, None):
        slot[_SLOT_ID] = call.id
    function = getattr(call, _ATTR_FUNCTION, None)
    if function and getattr(function, _FIELD_ARGUMENTS, None):
        slot[_SLOT_ARGS] += function.arguments


class CohereProvider(ModelProvider):
    name = PROVIDER_COHERE

    def __init__(self, sdk, client, model: str):
        super().__init__(model)
        self._client = client
        self.auth_errors = (sdk.UnauthorizedError,)
        self.api_errors = (sdk.core.ApiError,)

    def build_request(self, history, system_texts, disable_tools) -> dict:
        request = {
            _REQ_MODEL: self.model,
            _REQ_MESSAGES: _to_messages(system_texts, history),
        }
        if not disable_tools:
            request[_REQ_TOOLS] = [_FUNCTION_TOOL]
        return request

    def run_stream(self, request, collector: TurnCollector) -> None:
        accumulated: dict[int, dict] = {}
        for event in self._client.chat_stream(**request):
            kind = getattr(event, _ATTR_TYPE, None)
            if kind == _EVENT_MESSAGE_END:
                usage = getattr(getattr(event, _ATTR_DELTA, None), _ATTR_USAGE, None)
                if usage is not None:
                    collector.set_usage(_usage_from(usage))
                continue
            message = _event_message(event)
            if message is None:
                continue
            if kind == _EVENT_CONTENT_DELTA:
                collector.add_text(_content_text(message))
            elif kind in (_EVENT_TOOL_CALL_START, _EVENT_TOOL_CALL_DELTA):
                _accumulate_tool_event(event, message, accumulated)
        for index, slot in accumulated.items():
            call_id = slot[_SLOT_ID] or fallback_call_id(index)
            collector.add_tool_call(call_id, _parse_args(slot[_SLOT_ARGS]))


@register(PROVIDER_COHERE)
def build(config: Config) -> CohereProvider:
    sdk = import_optional(_MODULE_COHERE)
    kwargs = {}
    api_key = resolve_api_key(config)
    if api_key:
        kwargs[_KW_API_KEY] = api_key
    try:
        client = sdk.ClientV2(**kwargs)
    except Exception as e:
        raise ProviderError(_CLIENT_INIT_ERROR.format(error=e)) from e
    return CohereProvider(sdk, client, config.model)
