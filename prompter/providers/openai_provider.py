"""OpenAI (and OpenAI-compatible) provider adapter.

Uses the Chat Completions API with function tools and streaming. Because Groq
and OpenRouter implement the same API, pointing ``base_url`` at them reuses this
adapter unchanged.

Supplies the template method's two steps: build_request renders neutral history
into Chat Completions messages; run_stream accumulates streamed text and
tool-call deltas into the collector.
"""

from __future__ import annotations

import json
import os

from ..config import PROVIDER_OPENAI, Config
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
    import_optional,
    register,
)

_ROLE_SYSTEM = "system"
_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"
_ROLE_TOOL = "tool"
_FUNCTION_TYPE = "function"
_SYSTEM_JOIN = "\n\n"

_FUNCTION_TOOL = {
    "type": _FUNCTION_TYPE,
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                PARAM_COMMAND: {"type": "string", "description": COMMAND_DESCRIPTION},
                PARAM_EXPLANATION: {"type": "string", "description": EXPLANATION_DESCRIPTION},
                PARAM_INTERACTIVE: {"type": "boolean", "description": INTERACTIVE_DESCRIPTION},
            },
            "required": REQUIRED_PARAMS,
        },
    },
}

_CLIENT_INIT_ERROR = "Could not initialize the OpenAI client: {error}"


def _to_messages(system_texts: list[str], history: list[HistoryItem]) -> list[dict]:
    messages = [{"role": _ROLE_SYSTEM, "content": _SYSTEM_JOIN.join(system_texts)}]
    for item in history:
        if isinstance(item, UserMessage):
            messages.append({"role": _ROLE_USER, "content": item.text})
        elif isinstance(item, AssistantMessage):
            messages.append(_assistant_message(item))
        elif isinstance(item, ToolResultsMessage):
            for result in item.results:
                messages.append({
                    "role": _ROLE_TOOL,
                    "tool_call_id": result.call_id,
                    "content": result.content,
                })
    return messages


def _assistant_message(item: AssistantMessage) -> dict:
    message = {"role": _ROLE_ASSISTANT, "content": item.text or None}
    if item.tool_calls:
        message["tool_calls"] = [{
            "id": call.call_id,
            "type": _FUNCTION_TYPE,
            "function": {
                "name": TOOL_NAME,
                "arguments": json.dumps({
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
    for call in getattr(delta, "tool_calls", None) or []:
        slot = accumulated.setdefault(call.index, {"id": "", "args": ""})
        if getattr(call, "id", None):
            slot["id"] = call.id
        function = getattr(call, "function", None)
        if function and getattr(function, "arguments", None):
            slot["args"] += function.arguments


class OpenAIProvider(ModelProvider):
    name = PROVIDER_OPENAI

    def __init__(self, sdk, client, model: str):
        super().__init__(model)
        self._client = client
        self.auth_errors = (sdk.AuthenticationError,)
        self.api_errors = (sdk.OpenAIError,)

    def build_request(self, history, system_texts, disable_tools) -> dict:
        request = {
            "model": self.model,
            "messages": _to_messages(system_texts, history),
            "stream": True,
        }
        if not disable_tools:
            request["tools"] = [_FUNCTION_TOOL]
        return request

    def run_stream(self, request, collector: TurnCollector) -> None:
        accumulated: dict[int, dict] = {}
        for chunk in self._client.chat.completions.create(**request):
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            collector.add_text(getattr(delta, "content", None))
            _accumulate_tool_deltas(delta, accumulated)
        for index, slot in accumulated.items():
            call_id = slot["id"] or f"call_{index}"
            collector.add_tool_call(call_id, _parse_args(slot["args"]))


@register(PROVIDER_OPENAI)
def build(config: Config) -> OpenAIProvider:
    sdk = import_optional("openai", "openai")
    kwargs = {}
    api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
    if api_key:
        kwargs["api_key"] = api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    try:
        client = sdk.OpenAI(**kwargs)
    except Exception as e:
        raise ProviderError(_CLIENT_INIT_ERROR.format(error=e)) from e
    return OpenAIProvider(sdk, client, config.model)
