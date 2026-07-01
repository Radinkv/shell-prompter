"""Unit tests for the Cohere (Command) provider adapter.

The neutral round-trip (history -> v2 wire, stream -> AssistantTurn) is exercised
with a fake SDK client, so no network call or API key is ever made. The fake
mirrors only the slice of the cohere v2 surface the adapter touches: a
``chat_stream`` that yields typed events, and the ``UnauthorizedError`` /
``core.ApiError`` classes used for error mapping.
"""

from __future__ import annotations

import types

import pytest

from prompter.cli import _missing_key_problem
from prompter.config import Config, normalize_provider
from prompter.providers import base
from prompter.providers.base import (
    AssistantMessage,
    ToolInvocation,
    ToolResult,
    ToolResultsMessage,
    UserMessage,
    known_providers,
)
from prompter.providers import cohere_provider as cp


def _ns(**kw):
    return types.SimpleNamespace(**kw)


# --- fake cohere SDK -------------------------------------------------------

class _FakeCohereSdk:
    class core:
        class ApiError(Exception):
            def __init__(self, message="", status_code=None):
                super().__init__(message)
                self.status_code = status_code

    class UnauthorizedError(core.ApiError):
        pass


class _CohereClient:
    def __init__(self, events):
        self._events = events
        self.last_kwargs = None

    def chat_stream(self, **kwargs):
        self.last_kwargs = kwargs
        return iter(self._events)


class _RaisingCohereClient:
    def __init__(self, error):
        self._error = error

    def chat_stream(self, **kwargs):
        raise self._error


# --- fake v2 stream events -------------------------------------------------

def _content_delta(text):
    return _ns(type="content-delta",
               delta=_ns(message=_ns(content=_ns(text=text))))


def _tool_call_start(index, call_id, name="run_command", arguments=""):
    function = _ns(name=name, arguments=arguments)
    return _ns(type="tool-call-start", index=index,
               delta=_ns(message=_ns(tool_calls=_ns(id=call_id, function=function))))


def _tool_call_delta(index, arguments):
    function = _ns(name=None, arguments=arguments)
    return _ns(type="tool-call-delta", index=index,
               delta=_ns(message=_ns(tool_calls=_ns(id=None, function=function))))


def _message_end(input_tokens=0, output_tokens=0, cached_tokens=0):
    return _ns(type="message-end",
               delta=_ns(usage=_ns(
                   tokens=_ns(input_tokens=input_tokens, output_tokens=output_tokens),
                   cached_tokens=cached_tokens)))


def _provider(client, model="command-test"):
    return cp.CohereProvider(_FakeCohereSdk, client, model)


# --- registration / aliases ------------------------------------------------

def test_cohere_registered():
    assert "cohere" in known_providers()


def test_cohere_aliases_normalize():
    assert normalize_provider("Cohere") == "cohere"
    assert normalize_provider("command") == "cohere"


def test_cohere_default_model_and_key_env():
    config = Config(provider="cohere")
    assert config.resolved_model == "command-a-03-2025"
    assert config.key_env == "COHERE_API_KEY"


# --- build_request translates to the v2 wire format ------------------------

def test_cohere_to_messages_round_trip():
    history = [
        UserMessage("hi"),
        AssistantMessage("", [ToolInvocation("c1", "ls", "list", False)]),
        ToolResultsMessage([ToolResult("c1", "out", False)]),
    ]
    messages = cp._to_messages(["sysA", "sysB"], history)
    assert messages[0]["role"] == "system"
    assert "sysA" in messages[0]["content"] and "sysB" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "hi"}
    assistant = messages[2]
    assert assistant["role"] == "assistant"
    call = assistant["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "run_command"
    # arguments are serialized JSON carrying the neutral command fields
    import json
    assert json.loads(call["function"]["arguments"])["command"] == "ls"
    assert messages[3] == {"role": "tool", "tool_call_id": "c1", "content": "out"}


def test_cohere_build_request_includes_tools():
    provider = _provider(_CohereClient([]), "command-r-plus")
    request = provider.build_request([UserMessage("hi")], ["sys"], disable_tools=False)
    assert request["model"] == "command-r-plus"
    assert request["messages"][0]["role"] == "system"
    assert request["tools"][0]["function"]["name"] == "run_command"
    assert request["tools"][0]["function"]["parameters"] == base.RUN_COMMAND_PARAMETERS


def test_cohere_disable_tools_omits_tools():
    provider = _provider(_CohereClient([]))
    request = provider.build_request([], ["sys"], disable_tools=True)
    assert "tools" not in request


# --- run_stream handles the streaming response format ----------------------

def test_cohere_complete_accumulates_tool_call():
    events = [
        _content_delta("th"),
        _content_delta("inking"),
        _tool_call_start(0, "call_1", arguments=""),
        _tool_call_delta(0, '{"command": "l'),
        _tool_call_delta(0, 's -la"}'),
        _message_end(input_tokens=11, output_tokens=4),
    ]
    client = _CohereClient(events)
    provider = _provider(client)
    streamed = []
    turn = provider.complete([UserMessage("go")], ["sys"], False, streamed.append)
    assert streamed == ["th", "inking"]
    assert turn.text == "thinking"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].call_id == "call_1"
    assert turn.tool_calls[0].command == "ls -la"
    assert "tools" in client.last_kwargs


def test_cohere_captures_usage():
    events = [_content_delta("hi"),
              _message_end(input_tokens=20, output_tokens=8, cached_tokens=6)]
    turn = _provider(_CohereClient(events)).complete(
        [UserMessage("go")], ["sys"], False, lambda _t: None)
    assert turn.usage.input_tokens == 20
    assert turn.usage.output_tokens == 8
    assert turn.usage.cached_tokens == 6   # usage.cached_tokens, not nested in tokens


def test_cohere_tool_call_without_id_uses_fallback():
    events = [_tool_call_start(2, None, arguments='{"command": "ls"}')]
    turn = _provider(_CohereClient(events)).complete(
        [UserMessage("go")], ["sys"], False, lambda _t: None)
    assert turn.tool_calls[0].call_id == base.fallback_call_id(2)
    assert turn.tool_calls[0].command == "ls"


def test_cohere_parse_args_bad_json():
    assert cp._parse_args("{bad") == {}
    assert cp._parse_args("") == {}


# --- error mapping ---------------------------------------------------------

def test_cohere_unauthorized_becomes_auth_error():
    client = _RaisingCohereClient(_FakeCohereSdk.UnauthorizedError("invalid api key"))
    with pytest.raises(base.ProviderAuthError):
        _provider(client).complete([UserMessage("go")], ["sys"], False, lambda _t: None)


def test_cohere_server_error_is_retryable():
    error = _FakeCohereSdk.core.ApiError("overloaded", status_code=503)
    with pytest.raises(base.ProviderError) as exc:
        _provider(_RaisingCohereClient(error)).complete(
            [UserMessage("go")], ["sys"], False, lambda _t: None)
    assert exc.value.retryable is True


def test_cohere_bad_request_not_retryable():
    error = _FakeCohereSdk.core.ApiError("bad request", status_code=400)
    with pytest.raises(base.ProviderError) as exc:
        _provider(_RaisingCohereClient(error)).complete(
            [UserMessage("go")], ["sys"], False, lambda _t: None)
    assert exc.value.retryable is False


def test_cohere_build_without_key_raises_provider_error(monkeypatch):
    # With no resolvable key, the real ClientV2 refuses to construct; build()
    # wraps that as ProviderError, which the CLI turns into the missing-key
    # message. No network is touched -- the failure is at construction time.
    monkeypatch.setattr(cp, "resolve_api_key", lambda config: None)
    with pytest.raises(base.ProviderError):
        cp.build(Config(provider="cohere"))


# --- missing-key message matches the other providers -----------------------

def test_cohere_missing_key_message_is_actionable():
    title, hints, _ = _missing_key_problem(Config(provider="cohere"))
    assert "cohere" in title
    commands = [command for _label, command in hints]
    assert "prompter keys add cohere <key>" in commands
    assert any("COHERE_API_KEY" in command for command in commands)


def test_cohere_missing_key_message_shape_matches_anthropic():
    _, cohere_hints, _ = _missing_key_problem(Config(provider="cohere"))
    _, anthropic_hints, _ = _missing_key_problem(Config(provider="anthropic"))
    assert [label for label, _ in cohere_hints] == [label for label, _ in anthropic_hints]
