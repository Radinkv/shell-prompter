"""Google Gemini provider adapter (google-genai SDK).

Supplies the template method's two steps. build_request renders neutral history
into Gemini ``contents`` and a GenerateContentConfig. run_stream streams the
response, feeding text and function calls into the collector.

Gemini matches tool calls by function name (and an optional id) rather than an
opaque id like the other providers, so the round-trip uses the shared TOOL_NAME.
This adapter is written against the documented google-genai surface. Verify the
function-call round-trip with a live smoke test.

Prompt caching needs no code here: Gemini 2.5/3.x apply implicit caching to a
repeated prompt prefix automatically, and our stable system prefix is sent
unchanged each turn, so it benefits without an explicit CachedContent resource.
"""

from __future__ import annotations

import httpx

from ..config import PROVIDER_GEMINI, Config
from ..constants import EMPTY
from ..keys import resolve_api_key
from .base import (
    COMMAND_DESCRIPTION,
    EXPLANATION_DESCRIPTION,
    INTERACTIVE_DESCRIPTION,
    PARAM_COMMAND,
    PARAM_EXPLANATION,
    PARAM_INTERACTIVE,
    REQUIRED_PARAMS,
    SYSTEM_TEXT_SEPARATOR,
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
    fallback_call_id,
    import_optional,
    register,
)

_ROLE_USER = "user"
_ROLE_MODEL = "model"
_ROLE_TOOL = "tool"
_RESPONSE_KEY = "result"
_GEMINI_OBJECT = "OBJECT"
_GEMINI_STRING = "STRING"
_GEMINI_BOOLEAN = "BOOLEAN"
_AUTH_STATUS_CODES = {401, 403}
_AUTH_HINT = "api key"

_ATTR_TEXT = "text"
_ATTR_FUNCTION_CALLS = "function_calls"
_ATTR_CANDIDATES = "candidates"
_ATTR_CONTENT = "content"
_ATTR_PARTS = "parts"
_ATTR_FUNCTION_CALL = "function_call"
_ATTR_THOUGHT_SIGNATURE = "thought_signature"
_ATTR_CODE = "code"
_ATTR_STATUS_CODE = "status_code"
_ATTR_ID = "id"

_ATTR_USAGE_METADATA = "usage_metadata"
_ATTR_PROMPT_TOKEN_COUNT = "prompt_token_count"
_ATTR_CANDIDATES_TOKEN_COUNT = "candidates_token_count"
_ATTR_THOUGHTS_TOKEN_COUNT = "thoughts_token_count"
_ATTR_CACHED_TOKEN_COUNT = "cached_content_token_count"

_REQ_CONTENTS = "contents"
_REQ_CONFIG = "config"
_CFG_SYSTEM_INSTRUCTION = "system_instruction"
_CFG_TOOLS = "tools"

_MODULE_GENAI = "google.genai"
_MODULE_GENAI_TYPES = "google.genai.types"
_MODULE_GENAI_ERRORS = "google.genai.errors"


def _function_declaration(types):
    return types.FunctionDeclaration(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        parameters=types.Schema(
            type=_GEMINI_OBJECT,
            properties={
                PARAM_COMMAND: types.Schema(type=_GEMINI_STRING, description=COMMAND_DESCRIPTION),
                PARAM_EXPLANATION: types.Schema(type=_GEMINI_STRING, description=EXPLANATION_DESCRIPTION),
                PARAM_INTERACTIVE: types.Schema(type=_GEMINI_BOOLEAN, description=INTERACTIVE_DESCRIPTION),
            },
            required=REQUIRED_PARAMS,
        ),
    )


def _chunk_text(chunk) -> str | None:
    """Concatenate the chunk's text parts.

    Reading them directly avoids chunk.text, whose convenience accessor logs a
    warning on every chunk that also carries a function_call part -- i.e. every
    tool-calling turn. When a chunk exposes no candidate parts (a plain text
    response, or a test stub), fall back to the .text accessor.
    """
    candidates = getattr(chunk, _ATTR_CANDIDATES, None)
    if candidates:
        texts = [
            text
            for candidate in candidates
            for part in getattr(getattr(candidate, _ATTR_CONTENT, None),
                                _ATTR_PARTS, None) or []
            if (text := getattr(part, _ATTR_TEXT, None))
        ]
        return EMPTY.join(texts)
    try:
        return chunk.text
    except (ValueError, AttributeError):
        return None


def _chunk_calls(chunk) -> list:
    """Return (function_call, thought_signature) for each call in a chunk.

    The thought signature lives on the Part, and the convenient
    chunk.function_calls accessor drops it, so walk the parts first and fall back
    to that accessor (signature-less) only when there are no candidate parts.
    """
    pairs = []
    for candidate in getattr(chunk, _ATTR_CANDIDATES, None) or []:
        content = getattr(candidate, _ATTR_CONTENT, None)
        for part in getattr(content, _ATTR_PARTS, None) or []:
            call = getattr(part, _ATTR_FUNCTION_CALL, None)
            if call is not None:
                pairs.append((call, getattr(part, _ATTR_THOUGHT_SIGNATURE, None)))
    if pairs:
        return pairs
    return [(call, None) for call in getattr(chunk, _ATTR_FUNCTION_CALLS, None) or []]


def _chunk_usage(chunk) -> Usage | None:
    """The cumulative usage_metadata rides on the streamed chunks (the final one
    carries the totals); absent on chunks that don't report it."""
    metadata = getattr(chunk, _ATTR_USAGE_METADATA, None)
    if metadata is None:
        return None
    answer = getattr(metadata, _ATTR_CANDIDATES_TOKEN_COUNT, 0) or 0
    thoughts = getattr(metadata, _ATTR_THOUGHTS_TOKEN_COUNT, 0) or 0
    return Usage(
        input_tokens=getattr(metadata, _ATTR_PROMPT_TOKEN_COUNT, 0) or 0,
        output_tokens=answer + thoughts,   # thinking models bill thoughts as output
        cached_tokens=getattr(metadata, _ATTR_CACHED_TOKEN_COUNT, 0) or 0,
    )


def _is_auth_error(error) -> bool:
    code = getattr(error, _ATTR_CODE, None) or getattr(error, _ATTR_STATUS_CODE, None)
    if isinstance(code, int):
        return code in _AUTH_STATUS_CODES
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
        config_kwargs = {
            _CFG_SYSTEM_INSTRUCTION: SYSTEM_TEXT_SEPARATOR.join(system_texts)
        }
        if not disable_tools:
            tool = types.Tool(function_declarations=[_function_declaration(types)])
            config_kwargs[_CFG_TOOLS] = [tool]
        return {
            _REQ_CONTENTS: self._to_contents(history),
            _REQ_CONFIG: types.GenerateContentConfig(**config_kwargs),
        }

    def run_stream(self, request, collector: TurnCollector) -> None:
        try:
            stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=request[_REQ_CONTENTS],
                config=request[_REQ_CONFIG],
            )
            for chunk in stream:
                collector.add_text(_chunk_text(chunk))
                for index, (call, signature) in enumerate(_chunk_calls(chunk)):
                    call_id = getattr(call, _ATTR_ID, None) or fallback_call_id(index)
                    collector.add_tool_call(
                        call_id, dict(call.args or {}), signature=signature)
                collector.set_usage(_chunk_usage(chunk))
        except self._errors.APIError as e:
            if _is_auth_error(e):
                raise ProviderAuthError(str(e)) from e
            raise ProviderError(str(e), retryable=self.is_transient(e)) from e
        except httpx.TransportError as e:
            raise ProviderError(str(e), retryable=True) from e

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
                    role=_ROLE_TOOL, parts=self._result_parts(item)))
        return contents

    def _assistant_parts(self, item: AssistantMessage) -> list:
        types = self._types
        parts = []
        if item.text:
            parts.append(types.Part(text=item.text))
        for call in item.tool_calls:
            parts.append(types.Part(
                function_call=types.FunctionCall(
                    name=TOOL_NAME,
                    args={
                        PARAM_COMMAND: call.command,
                        PARAM_EXPLANATION: call.explanation,
                        PARAM_INTERACTIVE: call.interactive,
                    },
                ),
                thought_signature=call.signature,
            ))
        return parts or [types.Part(text=item.text)]

    def _result_parts(self, item: ToolResultsMessage) -> list:
        types = self._types
        return [
            types.Part.from_function_response(
                name=TOOL_NAME,
                response={_RESPONSE_KEY: result.content},
            )
            for result in item.results
        ]


@register(PROVIDER_GEMINI)
def build(config: Config) -> GeminiProvider:
    genai = import_optional(_MODULE_GENAI)
    types = import_optional(_MODULE_GENAI_TYPES)
    errors = import_optional(_MODULE_GENAI_ERRORS)
    api_key = resolve_api_key(config)
    try:
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
    except Exception as e:
        raise ProviderError(str(e)) from e
    return GeminiProvider(errors, types, client, config.model)
