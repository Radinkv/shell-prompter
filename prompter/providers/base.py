"""Provider-neutral model interface.

The agent speaks only in the neutral types here. Each provider adapter
translates them to and from its own API.

The abstraction is a template method. ModelProvider.complete() owns the
invariant algorithm. It builds a request, streams it into a TurnCollector, wraps
provider errors into ProviderError or ProviderAuthError, and returns an
AssistantTurn. A concrete provider supplies only the two steps that genuinely
differ: build_request, which turns neutral history plus system text into a
provider request, and run_stream, which opens the stream and feeds text and
tool calls into the collector. Each adapter also declares which of its SDK
exceptions mean auth versus other. Nothing in the agent or the loop changes
between providers.

A registry maps a provider name to a factory so config selects the backend with
one string.

The built-in adapters are Anthropic, OpenAI, Gemini, and Cohere; each registers
itself under its provider name (see config.PROVIDER_* and DEFAULT_MODELS).

The run_command tool and its JSON-Schema parameters are described once here.
Anthropic's input_schema, OpenAI's function.parameters, and Cohere's v2
function.parameters are all standard JSON Schema, so the parameter shape is
defined exactly once and reused. Gemini uses its own typed Schema objects and
builds them locally. SYSTEM_TEXT_SEPARATOR is
how providers that take a single system string join the system texts, and
fallback_call_id synthesizes a call id when the model did not supply one.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Union

from ..constants import COMMA_SPACE, EMPTY

if TYPE_CHECKING:
    from ..config import Config

_UNKNOWN_PROVIDER = "Unknown provider '{name}'. Known providers: {known}."
_NONE_REGISTERED = "(none registered)"
_MISSING_PACKAGE = (
    "The '{module}' package is missing. Reinstall prompter with:  pip install -e ."
)

TOOL_NAME = "run_command"
PARAM_COMMAND = "command"
PARAM_EXPLANATION = "explanation"
PARAM_INTERACTIVE = "interactive"
REQUIRED_PARAMS = [PARAM_COMMAND, PARAM_EXPLANATION]

TOOL_DESCRIPTION = (
    "Run a single shell command in the user's terminal session. The working "
    "directory persists between calls (so `cd` works as expected). Use one "
    "command per call and build up to the goal step by step. Set `interactive` "
    "to true for programs that take over the terminal and need the user to type "
    "into them (a coding agent like claude, vim, ssh, a REPL, etc.). Their "
    "output is shown to the user directly and not returned to you."
)
COMMAND_DESCRIPTION = "The exact shell command to run."
EXPLANATION_DESCRIPTION = "One short sentence on what this does and why."
INTERACTIVE_DESCRIPTION = "True if the program needs an interactive terminal."

JSON_SCHEMA_TYPE = "type"
JSON_SCHEMA_OBJECT = "object"
JSON_SCHEMA_STRING = "string"
JSON_SCHEMA_BOOLEAN = "boolean"
JSON_SCHEMA_DESCRIPTION = "description"
JSON_SCHEMA_PROPERTIES = "properties"
JSON_SCHEMA_REQUIRED = "required"


def _string_property(description: str) -> dict:
    return {JSON_SCHEMA_TYPE: JSON_SCHEMA_STRING, JSON_SCHEMA_DESCRIPTION: description}


def _boolean_property(description: str) -> dict:
    return {JSON_SCHEMA_TYPE: JSON_SCHEMA_BOOLEAN, JSON_SCHEMA_DESCRIPTION: description}


RUN_COMMAND_PARAMETERS = {
    JSON_SCHEMA_TYPE: JSON_SCHEMA_OBJECT,
    JSON_SCHEMA_PROPERTIES: {
        PARAM_COMMAND: _string_property(COMMAND_DESCRIPTION),
        PARAM_EXPLANATION: _string_property(EXPLANATION_DESCRIPTION),
        PARAM_INTERACTIVE: _boolean_property(INTERACTIVE_DESCRIPTION),
    },
    JSON_SCHEMA_REQUIRED: REQUIRED_PARAMS,
}

SYSTEM_TEXT_SEPARATOR = "\n\n"
_CALL_ID_TEMPLATE = "call_{index}"
_BASE_PROVIDER_NAME = "model"


def fallback_call_id(index: object) -> str:
    """Synthesize a call id when a provider did not supply one."""
    return _CALL_ID_TEMPLATE.format(index=index)


@dataclass
class ToolInvocation:
    """A single tool call the model wants to make.

    ``signature`` is an opaque, provider-supplied token that must be replayed
    verbatim when this call is sent back in history. Gemini's thinking models
    attach a thought signature to each call and reject a follow-up turn that
    omits it; other providers leave this None.
    """

    call_id: str
    command: str
    explanation: str = EMPTY
    interactive: bool = False
    signature: bytes | None = None


@dataclass
class Usage:
    """Token counts for one model turn.

    input_tokens is the whole prompt (every provider reports it inclusive of any
    cached portion); cached_tokens is the part of it served from the prompt cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_tokens + other.cached_tokens,
        )


@dataclass
class AssistantTurn:
    """One model turn: any text plus any tool calls. No calls means a final turn."""

    text: str
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    usage: Usage | None = None


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool


@dataclass
class UserMessage:
    text: str


@dataclass
class AssistantMessage:
    text: str
    tool_calls: list[ToolInvocation] = field(default_factory=list)


@dataclass
class ToolResultsMessage:
    results: list[ToolResult]


HistoryItem = Union[UserMessage, AssistantMessage, ToolResultsMessage]


def tool_invocation_from_args(
    call_id: str, args: dict | None, signature: bytes | None = None
) -> ToolInvocation:
    """Build a neutral ToolInvocation from a provider's parsed tool arguments."""
    args = args or {}
    return ToolInvocation(
        call_id=call_id,
        command=args.get(PARAM_COMMAND, EMPTY),
        explanation=args.get(PARAM_EXPLANATION, EMPTY),
        interactive=bool(args.get(PARAM_INTERACTIVE, False)),
        signature=signature,
    )


class ProviderError(Exception):
    """A model request failed (network, server, bad request).

    retryable marks a transient failure (rate limit, timeout, 5xx) that the
    agent may retry with backoff; a bad request or auth failure is not.
    """

    def __init__(self, message: str = EMPTY, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class ProviderAuthError(ProviderError):
    """The provider rejected the credentials."""


class ProviderNotInstalled(ProviderError):
    """The provider's SDK isn't installed."""


class TurnCollector:
    """Accumulates streamed text and tool calls into one AssistantTurn.

    add_text streams a chunk to the console as it arrives. add_tool_call records
    a call. finish() produces the neutral turn. Every adapter feeds this, so the
    logic that assembles a turn exists exactly once.
    """

    def __init__(self, on_text: Callable[[str], None]):
        self._on_text = on_text
        self._text_parts: list[str] = []
        self._tool_calls: list[ToolInvocation] = []
        self._usage: Usage | None = None

    def add_text(self, text: str | None) -> None:
        if text:
            self._on_text(text)
            self._text_parts.append(text)

    def add_tool_call(self, call_id: str, args: dict | None,
                      signature: bytes | None = None) -> None:
        self._tool_calls.append(tool_invocation_from_args(call_id, args, signature))

    def set_usage(self, usage: Usage | None) -> None:
        if usage is not None:
            self._usage = usage

    def finish(self) -> AssistantTurn:
        return AssistantTurn(EMPTY.join(self._text_parts), self._tool_calls, self._usage)


_ATTR_STATUS_CODE = "status_code"
_ATTR_ERROR_CODE = "code"
# Request timeout, rate limit, and server errors -- the codes that are actually
# safe to replay. Not 409/425, which signal client-side state, not a blip.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class ModelProvider(ABC):
    name: str = _BASE_PROVIDER_NAME
    auth_errors: tuple = ()
    api_errors: tuple = ()
    transient_errors: tuple = ()

    def __init__(self, model: str):
        self.model = model

    def is_transient(self, error: Exception) -> bool:
        """Whether a failed request is worth retrying: a declared transient SDK
        error (timeout, connection drop) or a retryable HTTP status."""
        if self.transient_errors and isinstance(error, self.transient_errors):
            return True
        code = (getattr(error, _ATTR_STATUS_CODE, None)
                or getattr(error, _ATTR_ERROR_CODE, None))
        # .code can be a string app-code on some SDKs; only an int HTTP status
        # is meaningful here.
        return isinstance(code, int) and code in RETRYABLE_STATUS_CODES

    def complete(
        self,
        history: list[HistoryItem],
        system_texts: list[str],
        disable_tools: bool,
        on_text: Callable[[str], None],
    ) -> AssistantTurn:
        """The invariant algorithm. Subclasses do not override this."""
        collector = TurnCollector(on_text)
        try:
            request = self.build_request(history, system_texts, disable_tools)
            self.run_stream(request, collector)
        except self.auth_errors as e:
            raise ProviderAuthError(str(e)) from e
        except self.api_errors as e:
            raise ProviderError(str(e), retryable=self.is_transient(e)) from e
        return collector.finish()

    @abstractmethod
    def build_request(self, history: list[HistoryItem], system_texts: list[str],
                      disable_tools: bool) -> Any:
        """Translate neutral history and system text into a provider request."""

    @abstractmethod
    def run_stream(self, request: Any, collector: TurnCollector) -> None:
        """Open the stream and feed text and tool calls into the collector."""


ProviderFactory = Callable[..., ModelProvider]
_FACTORIES: dict[str, ProviderFactory] = {}


def register(name: str) -> Callable[[ProviderFactory], ProviderFactory]:
    def decorator(factory: ProviderFactory) -> ProviderFactory:
        _FACTORIES[name] = factory
        return factory

    return decorator


def known_providers() -> list[str]:
    return sorted(_FACTORIES)


def unknown_provider_message(name: str) -> str:
    known = COMMA_SPACE.join(known_providers()) or _NONE_REGISTERED
    return _UNKNOWN_PROVIDER.format(name=name, known=known)


def create_provider(config: Config) -> ModelProvider:
    try:
        factory = _FACTORIES[config.provider]
    except KeyError as e:
        raise ProviderError(unknown_provider_message(config.provider)) from e
    return factory(config)


def import_optional(module_name: str, package: str | None = None):
    """Import a provider SDK, with an actionable error if it's missing.

    The provider SDKs ship as core dependencies, so this only fires on a broken
    or partial install.
    """
    try:
        return importlib.import_module(module_name, package)
    except ImportError as e:
        raise ProviderNotInstalled(_MISSING_PACKAGE.format(module=module_name)) from e
