"""Google Gemini provider adapter (google-genai SDK).

Supplies the template method's two steps: build_request renders neutral history
into Gemini ``contents`` + a GenerateContentConfig; run_stream streams the
response, feeding text and function calls into the collector.

Gemini matches tool calls by function name (and an optional id) rather than an
opaque id like the other providers, so the round-trip uses the shared TOOL_NAME.
This adapter is written against the documented google-genai surface; verify the
function-call round-trip with a live smoke test.
"""

from __future__ import annotations

import os

from ..config import PROVIDER_GEMINI, Config
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
    ProviderAuthError,
    ProviderError,
    TurnCollector,
    ToolResultsMessage,
    UserMessage,
    import_optional,
    register,
)

_ROLE_USER = "user"
_ROLE_MODEL = "model"
_SYSTEM_JOIN = "\n\n"
_RESPONSE_KEY = "output"
_AUTH_STATUS_CODES = {401, 403}
_AUTH_HINT = "api key"


def _function_declaration(types):
    return types.FunctionDeclaration(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        parameters=types.Schema(
            type="OBJECT",
            properties={
                PARAM_COMMAND: types.Schema(type="STRING", description=COMMAND_DESCRIPTION),
                PARAM_EXPLANATION: types.Schema(type="STRING", description=EXPLANATION_DESCRIPTION),
                PARAM_INTERACTIVE: types.Schema(type="BOOLEAN", description=INTERACTIVE_DESCRIPTION),
            },
            required=REQUIRED_PARAMS,
        ),
    )


def _chunk_text(chunk) -> str | None:
    try:
        return chunk.text
    except (ValueError, AttributeError):
        return None


def _chunk_function_calls(chunk) -> list:
    direct = getattr(chunk, "function_calls", None)
    if direct:
        return list(direct)
    calls = []
    for candidate in getattr(chunk, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call is not None:
                calls.append(call)
    return calls


def _is_auth_error(error) -> bool:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if code in _AUTH_STATUS_CODES:
        return True
    return _AUTH_HINT in str(error).lower()


class GeminiProvider(ModelProvider):
    name = PROVIDER_GEMINI

    def __init__(self, errors, types, client, model: str):
        super().__init__(model)
        self._errors = errors
        self._types = types
        self._client = client

    def build_request(self, history, system_texts, disable_tools) -> dict:
        types = self._types
        config_kwargs = {"system_instruction": _SYSTEM_JOIN.join(system_texts)}
        if not disable_tools:
            tool = types.Tool(function_declarations=[_function_declaration(types)])
            config_kwargs["tools"] = [tool]
        return {
            "contents": self._to_contents(history),
            "config": types.GenerateContentConfig(**config_kwargs),
        }

    def run_stream(self, request, collector: TurnCollector) -> None:
        try:
            stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=request["contents"],
                config=request["config"],
            )
            for chunk in stream:
                collector.add_text(_chunk_text(chunk))
                for index, call in enumerate(_chunk_function_calls(chunk)):
                    call_id = getattr(call, "id", None) or f"call_{index}"
                    collector.add_tool_call(call_id, dict(call.args or {}))
        except self._errors.APIError as e:
            if _is_auth_error(e):
                raise ProviderAuthError(str(e)) from e
            raise ProviderError(str(e)) from e

    def _to_contents(self, history: list[HistoryItem]) -> list:
        types = self._types
        contents = []
        for item in history:
            if isinstance(item, UserMessage):
                contents.append(types.Content(
                    role=_ROLE_USER, parts=[types.Part(text=item.text)]))
            elif isinstance(item, AssistantMessage):
                contents.append(types.Content(
                    role=_ROLE_MODEL, parts=self._assistant_parts(item)))
            elif isinstance(item, ToolResultsMessage):
                contents.append(types.Content(
                    role=_ROLE_USER, parts=self._result_parts(item)))
        return contents

    def _assistant_parts(self, item: AssistantMessage) -> list:
        types = self._types
        parts = []
        if item.text:
            parts.append(types.Part(text=item.text))
        for call in item.tool_calls:
            parts.append(types.Part(function_call=types.FunctionCall(
                name=TOOL_NAME,
                args={
                    PARAM_COMMAND: call.command,
                    PARAM_EXPLANATION: call.explanation,
                    PARAM_INTERACTIVE: call.interactive,
                },
            )))
        return parts or [types.Part(text=item.text)]

    def _result_parts(self, item: ToolResultsMessage) -> list:
        types = self._types
        return [
            types.Part(function_response=types.FunctionResponse(
                name=TOOL_NAME,
                response={_RESPONSE_KEY: result.content},
            ))
            for result in item.results
        ]


@register(PROVIDER_GEMINI)
def build(config: Config) -> GeminiProvider:
    genai = import_optional("google.genai", "gemini")
    types = import_optional("google.genai.types", "gemini")
    errors = import_optional("google.genai.errors", "gemini")
    api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
    try:
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
    except Exception as e:
        raise ProviderError(str(e)) from e
    return GeminiProvider(errors, types, client, config.model)
