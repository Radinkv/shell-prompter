"""Anthropic (Claude) provider adapter.

Supplies the two provider-specific steps of the template method: render neutral
history into the Messages API shape, and stream a turn with the run_command tool
and adaptive thinking. The Anthropic wire vocabulary lives here — no other
module needs it.
"""

from __future__ import annotations

import os

import anthropic

from ..config import PROVIDER_ANTHROPIC, Config
from .base import (
    COMMAND_DESCRIPTION,
    EXPLANATION_DESCRIPTION,
    INTERACTIVE_DESCRIPTION,
    PARAM_COMMAND,
    PARAM_EXPLANATION,
    PARAM_INTERACTIVE,
    REQUIRED_PARAMS,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    AssistantMessage,
    HistoryItem,
    ModelProvider,
    ProviderError,
    TurnCollector,
    ToolResultsMessage,
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
_THINKING_ADAPTIVE = {"type": "adaptive"}
_TOOL_CHOICE_NONE = {"type": "none"}

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

_CLIENT_INIT_ERROR = "Could not initialize the Anthropic client: {error}"


def _string_property(description: str) -> dict:
    return {_SCHEMA_TYPE: _SCHEMA_STRING, _SCHEMA_DESCRIPTION: description}


def _boolean_property(description: str) -> dict:
    return {_SCHEMA_TYPE: _SCHEMA_BOOLEAN, _SCHEMA_DESCRIPTION: description}


_RUN_TOOL = {
    _TOOL_KEY_NAME: TOOL_NAME,
    _TOOL_KEY_DESCRIPTION: TOOL_DESCRIPTION,
    _TOOL_KEY_INPUT_SCHEMA: {
        _SCHEMA_TYPE: _SCHEMA_OBJECT,
        _SCHEMA_PROPERTIES: {
            PARAM_COMMAND: _string_property(COMMAND_DESCRIPTION),
            PARAM_EXPLANATION: _string_property(EXPLANATION_DESCRIPTION),
            PARAM_INTERACTIVE: _boolean_property(INTERACTIVE_DESCRIPTION),
        },
        _SCHEMA_REQUIRED: REQUIRED_PARAMS,
    },
}


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

    def __init__(self, client, model: str):
        super().__init__(model)
        self._client = client

    def build_request(self, history, system_texts, disable_tools) -> dict:
        request = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": [{_FIELD_TYPE: _BLOCK_TEXT, _FIELD_TEXT: t} for t in system_texts],
            "thinking": _THINKING_ADAPTIVE,
            "tools": [_RUN_TOOL],
            "messages": _to_messages(history),
        }
        if disable_tools:
            request["tool_choice"] = _TOOL_CHOICE_NONE
        return request

    def run_stream(self, request, collector: TurnCollector) -> None:
        with self._client.messages.stream(**request) as stream:
            for event in stream:
                if (event.type == _EVENT_CONTENT_BLOCK_DELTA
                        and event.delta.type == _DELTA_TEXT):
                    collector.add_text(event.delta.text)
            final = stream.get_final_message()
        for block in final.content:
            if block.type == _BLOCK_TOOL_USE:
                collector.add_tool_call(block.id, block.input or {})


@register(PROVIDER_ANTHROPIC)
def build(config: Config) -> AnthropicProvider:
    api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
    try:
        client = (anthropic.Anthropic(api_key=api_key) if api_key
                  else anthropic.Anthropic())
    except Exception as e:
        raise ProviderError(_CLIENT_INIT_ERROR.format(error=e)) from e
    return AnthropicProvider(client, config.model)
