"""Provider-neutral model interface.

The agent speaks only in the neutral types here; each provider adapter
translates them to and from its own API.

The abstraction is a template method. ModelProvider.complete() owns the
invariant algorithm — build a request, stream it into a TurnCollector, wrap
provider errors into ProviderError/ProviderAuthError, return an AssistantTurn.
A concrete provider supplies only the two steps that genuinely differ:

    build_request(history, system_texts, disable_tools) -> request
    run_stream(request, collector) -> None   # feed collector.add_text / add_tool_call

and declares which of its SDK's exceptions mean "auth" vs "other". Nothing in
the agent or the loop changes between providers.

A registry maps a provider name to a factory so config selects the backend with
one string.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Union

if TYPE_CHECKING:
    from ..config import Config

# -- the run_command tool, described once for every adapter ------------------
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
    "into them (a coding agent like claude, vim, ssh, a REPL, etc.) — their "
    "output is shown to the user directly and not returned to you."
)
COMMAND_DESCRIPTION = "The exact shell command to run."
EXPLANATION_DESCRIPTION = "One short sentence on what this does and why."
INTERACTIVE_DESCRIPTION = "True if the program needs an interactive terminal."

# -- shared JSON-Schema for the tool parameters ------------------------------
# Anthropic's input_schema and OpenAI's function.parameters are both standard
# JSON Schema, so the parameter shape is defined exactly once here. Gemini uses
# its own typed Schema objects and builds them locally.
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

# How providers that take a single system string join the system texts, and how
# they synthesize a call id when the model didn't supply one.
SYSTEM_TEXT_SEPARATOR = "\n\n"


def fallback_call_id(index: object) -> str:
    return f"call_{index}"


# -- neutral data types ------------------------------------------------------
@dataclass
class ToolInvocation:
    """A single tool call the model wants to make."""

    call_id: str
    command: str
    explanation: str = ""
    interactive: bool = False


@dataclass
class AssistantTurn:
    """One model turn: any text plus any tool calls. Empty calls => final."""

    text: str
    tool_calls: list[ToolInvocation] = field(default_factory=list)


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


def tool_invocation_from_args(call_id: str, args: dict | None) -> ToolInvocation:
    """Build a neutral ToolInvocation from a provider's parsed tool arguments."""
    args = args or {}
    return ToolInvocation(
        call_id=call_id,
        command=args.get(PARAM_COMMAND, ""),
        explanation=args.get(PARAM_EXPLANATION, ""),
        interactive=bool(args.get(PARAM_INTERACTIVE, False)),
    )


# -- errors ------------------------------------------------------------------
class ProviderError(Exception):
    """A model request failed (network, server, bad request)."""


class ProviderAuthError(ProviderError):
    """The provider rejected the credentials."""


class ProviderNotInstalled(ProviderError):
    """The provider's SDK isn't installed."""


# -- the turn accumulator shared by every adapter ----------------------------
class TurnCollector:
    """Accumulates streamed text and tool calls into one AssistantTurn.

    add_text streams a chunk to the console as it arrives; add_tool_call records
    a call. finish() produces the neutral turn. Every adapter feeds this, so the
    "assemble a turn" logic exists exactly once.
    """

    def __init__(self, on_text: Callable[[str], None]):
        self._on_text = on_text
        self._text_parts: list[str] = []
        self._tool_calls: list[ToolInvocation] = []

    def add_text(self, text: str | None) -> None:
        if text:
            self._on_text(text)
            self._text_parts.append(text)

    def add_tool_call(self, call_id: str, args: dict | None) -> None:
        self._tool_calls.append(tool_invocation_from_args(call_id, args))

    def finish(self) -> AssistantTurn:
        return AssistantTurn("".join(self._text_parts), self._tool_calls)


# -- the template-method base every adapter extends --------------------------
class ModelProvider(ABC):
    name: str = "model"
    auth_errors: tuple = ()
    api_errors: tuple = ()

    def __init__(self, model: str):
        self.model = model

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
            raise ProviderError(str(e)) from e
        return collector.finish()

    @abstractmethod
    def build_request(self, history: list[HistoryItem], system_texts: list[str],
                      disable_tools: bool) -> Any:
        """Translate neutral history + system text into a provider request."""

    @abstractmethod
    def run_stream(self, request: Any, collector: TurnCollector) -> None:
        """Open the stream and feed text + tool calls into the collector."""


# -- registry ----------------------------------------------------------------
ProviderFactory = Callable[["Config"], ModelProvider]
_FACTORIES: dict[str, ProviderFactory] = {}


def register(name: str) -> Callable[[ProviderFactory], ProviderFactory]:
    def decorator(factory: ProviderFactory) -> ProviderFactory:
        _FACTORIES[name] = factory
        return factory

    return decorator


def create_provider(config: "Config") -> ModelProvider:
    try:
        factory = _FACTORIES[config.provider]
    except KeyError as e:
        known = ", ".join(sorted(_FACTORIES)) or "(none registered)"
        raise ProviderError(
            f"Unknown provider '{config.provider}'. Known providers: {known}."
        ) from e
    return factory(config)


def known_providers() -> list[str]:
    return sorted(_FACTORIES)


def import_optional(module_name: str):
    """Import a provider SDK, with an actionable error if it's missing.

    The provider SDKs ship as core dependencies, so this only fires on a broken
    or partial install.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ProviderNotInstalled(
            f"The '{module_name}' package is missing. "
            f"Reinstall prompter with:  pip install -e ."
        ) from e
