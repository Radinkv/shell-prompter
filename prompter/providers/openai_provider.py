"""OpenAI (and OpenAI-compatible) provider adapter.

Uses the Chat Completions API with function tools and streaming. Because Groq
and OpenRouter implement the same API, pointing ``base_url`` at them reuses this
adapter unchanged.

Supplies the template method's two steps. build_request renders neutral history
into Chat Completions messages. run_stream accumulates streamed text and
tool-call deltas into the collector.

Prompt caching needs no code here: OpenAI caches identical prompt prefixes over
1024 tokens automatically, and our stable system prefix is sent unchanged each
turn, so it benefits for free. (OpenAI-compatible endpoints apply their own
rules.)
"""

from __future__ import annotations

import json

from ..config import PROVIDER_OPENAI, Config
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
_ATTR_FUNCTION = "function"
_RESP_CHOICES = "choices"

_SLOT_ID = "id"
_SLOT_ARGS = "args"

_REQ_MODEL = "model"
_REQ_MESSAGES = "messages"
_REQ_STREAM = "stream"
_REQ_TOOLS = "tools"

_MODULE_OPENAI = "openai"
_KW_API_KEY = "api_key"
_KW_BASE_URL = "base_url"

_FUNCTION_TOOL = {
    _TOOL_KEY_TYPE: _FUNCTION_TYPE,
    _TOOL_KEY_FUNCTION: {
        _TOOL_KEY_NAME: TOOL_NAME,
        _TOOL_KEY_DESCRIPTION: TOOL_DESCRIPTION,
        _TOOL_KEY_PARAMETERS: RUN_COMMAND_PARAMETERS,
    },
}

_CLIENT_INIT_ERROR = "Could not initialize the OpenAI client: {error}"


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


def _parse_args(raw: str) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {}


def _accumulate_tool_deltas(delta, accumulated: dict) -> None:
    for call in getattr(delta, _FIELD_TOOL_CALLS, None) or []:
        slot = accumulated.setdefault(call.index, {_SLOT_ID: EMPTY, _SLOT_ARGS: EMPTY})
        if getattr(call, _FIELD_ID, None):
            slot[_SLOT_ID] = call.id
        function = getattr(call, _ATTR_FUNCTION, None)
        if function and getattr(function, _FIELD_ARGUMENTS, None):
            slot[_SLOT_ARGS] += function.arguments


class OpenAIProvider(ModelProvider):
    name = PROVIDER_OPENAI

    def __init__(self, sdk, client, model: str):
        super().__init__(model)
        self._client = client
        self.auth_errors = (sdk.AuthenticationError,)
        self.api_errors = (sdk.OpenAIError,)

    def build_request(self, history, system_texts, disable_tools) -> dict:
        request = {
            _REQ_MODEL: self.model,
            _REQ_MESSAGES: _to_messages(system_texts, history),
            _REQ_STREAM: True,
        }
        if not disable_tools:
            request[_REQ_TOOLS] = [_FUNCTION_TOOL]
        return request

    def run_stream(self, request, collector: TurnCollector) -> None:
        accumulated: dict[int, dict] = {}
        for chunk in self._client.chat.completions.create(**request):
            choices = getattr(chunk, _RESP_CHOICES, None) or []
            if not choices:
                continue
            delta = choices[0].delta
            collector.add_text(getattr(delta, _FIELD_CONTENT, None))
            _accumulate_tool_deltas(delta, accumulated)
        for index, slot in accumulated.items():
            call_id = slot[_SLOT_ID] or fallback_call_id(index)
            collector.add_tool_call(call_id, _parse_args(slot[_SLOT_ARGS]))


@register(PROVIDER_OPENAI)
def build(config: Config) -> OpenAIProvider:
    sdk = import_optional(_MODULE_OPENAI)
    kwargs = {}
    api_key = resolve_api_key(config)
    if api_key:
        kwargs[_KW_API_KEY] = api_key
    if config.base_url:
        kwargs[_KW_BASE_URL] = config.base_url
    try:
        client = sdk.OpenAI(**kwargs)
    except Exception as e:
        raise ProviderError(_CLIENT_INIT_ERROR.format(error=e)) from e
    return OpenAIProvider(sdk, client, config.model)
