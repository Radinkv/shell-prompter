"""Unit tests for the provider abstraction and the three adapters.

Each adapter's neutral round-trip (history -> wire, stream -> AssistantTurn) is
exercised with a fake SDK client, so no network or API key is needed.
"""

from __future__ import annotations

import types

import pytest

from prompter.config import Config
from prompter.providers import base
from prompter.providers.base import (
    AssistantMessage,
    ToolInvocation,
    ToolResult,
    ToolResultsMessage,
    TurnCollector,
    UserMessage,
    create_provider,
    known_providers,
    tool_invocation_from_args,
)
from prompter.providers import anthropic_provider as ap
from prompter.providers import gemini_provider as gp
from prompter.providers import openai_provider as op


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def test_known_providers_registered():
    assert set(known_providers()) >= {"anthropic", "openai", "gemini"}


def test_create_unknown_provider_errors():
    with pytest.raises(base.ProviderError):
        create_provider(Config(provider="nope"))


def test_tool_invocation_from_args():
    inv = tool_invocation_from_args("id1", {"command": "ls", "interactive": True})
    assert inv.call_id == "id1"
    assert inv.command == "ls"
    assert inv.interactive is True
    assert inv.explanation == ""


def test_tool_invocation_from_empty_args():
    inv = tool_invocation_from_args("id", None)
    assert inv.command == "" and inv.interactive is False


def test_turn_collector():
    streamed = []
    collector = TurnCollector(streamed.append)
    collector.add_text("hello ")
    collector.add_text(None)
    collector.add_text("world")
    collector.add_tool_call("c1", {"command": "ls"})
    turn = collector.finish()
    assert streamed == ["hello ", "world"]
    assert turn.text == "hello world"
    assert turn.tool_calls[0].command == "ls"


def test_import_optional_missing():
    with pytest.raises(base.ProviderNotInstalled):
        base.import_optional("definitely_not_a_real_module_xyz")


class _AnthropicStream:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


class _AnthropicClient:
    def __init__(self, events, final):
        self._events = events
        self._final = final
        self.last_kwargs = None
        self.messages = _ns(stream=self._stream)

    def _stream(self, **kwargs):
        self.last_kwargs = kwargs
        return _AnthropicStream(self._events, self._final)


def _text_delta(text):
    return _ns(type="content_block_delta", delta=_ns(type="text_delta", text=text))


def test_anthropic_to_messages_round_trip():
    history = [
        UserMessage("hi"),
        AssistantMessage("ok", [ToolInvocation("c1", "ls", "list", False)]),
        ToolResultsMessage([ToolResult("c1", "out", False)]),
    ]
    messages = ap._to_messages(history)
    assert messages[0] == {"role": "user", "content": "hi"}
    assistant = messages[1]
    assert assistant["role"] == "assistant"
    assert any(b["type"] == "tool_use" and b["id"] == "c1"
               for b in assistant["content"])
    result_block = messages[2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "c1"
    assert result_block["is_error"] is False


def test_anthropic_complete_parses_text_and_tool():
    final = _ns(content=[
        _ns(type="text", text="working"),
        _ns(type="tool_use", id="t1",
            input={"command": "ls", "explanation": "x", "interactive": False}),
    ])
    client = _AnthropicClient([_text_delta("work"), _text_delta("ing")], final)
    provider = ap.AnthropicProvider(client, "claude-test")
    streamed = []
    turn = provider.complete([UserMessage("go")], ["sys"], False, streamed.append)
    assert streamed == ["work", "ing"]
    assert turn.text == "working"
    assert turn.tool_calls[0].call_id == "t1"
    assert turn.tool_calls[0].command == "ls"
    assert "tool_choice" not in client.last_kwargs


def test_anthropic_disable_tools_sets_choice():
    final = _ns(content=[_ns(type="text", text="done")])
    client = _AnthropicClient([], final)
    ap.AnthropicProvider(client, "m").complete([], ["sys"], True, lambda _t: None)
    assert client.last_kwargs["tool_choice"] == {"type": "none"}


@pytest.mark.parametrize("system_texts, cached_index", [
    (["stable", "volatile-env"], 0),            # concise off: cache the base prompt
    (["base", "concise", "volatile-env"], 1),   # concise on: cache through concise
])
def test_anthropic_caches_stable_system_prefix(system_texts, cached_index):
    request = ap.AnthropicProvider(None, "m").build_request(
        [], system_texts, disable_tools=False)
    system = request["system"]
    assert system[cached_index]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system[-1]   # volatile env block re-sent fresh


class _RaisingClient:
    def __init__(self, error):
        self._error = error

        def stream(**kwargs):
            raise self._error
        self.messages = _ns(stream=stream)


def test_anthropic_missing_auth_becomes_auth_error():
    client = _RaisingClient(TypeError("Could not resolve authentication method"))
    provider = ap.AnthropicProvider(client, "m")
    with pytest.raises(base.ProviderAuthError):
        provider.complete([UserMessage("go")], ["sys"], False, lambda _t: None)


def test_anthropic_other_typeerror_propagates():
    client = _RaisingClient(TypeError("some unrelated bug"))
    provider = ap.AnthropicProvider(client, "m")
    with pytest.raises(TypeError):
        provider.complete([UserMessage("go")], ["sys"], False, lambda _t: None)


class _FakeOpenAIErrors:
    class AuthenticationError(Exception):
        pass

    class OpenAIError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass


class _OpenAIClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_params = None
        self.chat = _ns(completions=_ns(create=self._create))

    def _create(self, **params):
        self.last_params = params
        return iter(self._chunks)


def _openai_chunk(content=None, tool_calls=None):
    return _ns(choices=[_ns(delta=_ns(content=content, tool_calls=tool_calls))])


def _tc_delta(index, call_id=None, arguments=None):
    function = _ns(name="run_command", arguments=arguments) if arguments else None
    return _ns(index=index, id=call_id, function=function)


def test_openai_to_messages_round_trip():
    history = [
        UserMessage("hi"),
        AssistantMessage("", [ToolInvocation("c1", "ls", "list", False)]),
        ToolResultsMessage([ToolResult("c1", "out", False)]),
    ]
    messages = op._to_messages(["sysA", "sysB"], history)
    assert messages[0]["role"] == "system"
    assert "sysA" in messages[0]["content"] and "sysB" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["id"] == "c1"
    assert messages[3] == {"role": "tool", "tool_call_id": "c1", "content": "out"}


def test_openai_complete_accumulates_tool_call():
    chunks = [
        _openai_chunk(content="th"),
        _openai_chunk(content="inking"),
        _openai_chunk(tool_calls=[_tc_delta(0, "call_1", '{"command": "l')]),
        _openai_chunk(tool_calls=[_tc_delta(0, None, 's -la"}')]),
    ]
    client = _OpenAIClient(chunks)
    provider = op.OpenAIProvider(_FakeOpenAIErrors, client, "gpt-test")
    streamed = []
    turn = provider.complete([UserMessage("go")], ["sys"], False, streamed.append)
    assert turn.text == "thinking"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].call_id == "call_1"
    assert turn.tool_calls[0].command == "ls -la"
    assert "tools" in client.last_params


def test_openai_disable_tools_omits_tools():
    client = _OpenAIClient([_openai_chunk(content="hi")])
    op.OpenAIProvider(_FakeOpenAIErrors, client, "m").complete(
        [], ["sys"], True, lambda _t: None)
    assert "tools" not in client.last_params


def test_openai_parse_args_bad_json():
    assert op._parse_args("{bad") == {}
    assert op._parse_args("") == {}


class _FakePart:
    def __init__(self, text=None, function_call=None, function_response=None,
                 thought_signature=None):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response
        self.thought_signature = thought_signature

    @staticmethod
    def from_function_response(name=None, response=None):
        return _FakePart(function_response=_ns(name=name, response=response))


def _fake_genai_types():
    return _ns(
        Content=lambda role=None, parts=None: _ns(role=role, parts=parts),
        Part=_FakePart,
        FunctionCall=lambda name=None, args=None, id=None: _ns(
            name=name, args=args, id=id),
        Tool=lambda function_declarations=None: _ns(
            function_declarations=function_declarations),
        FunctionDeclaration=lambda name=None, description=None, parameters=None: _ns(
            name=name, description=description, parameters=parameters),
        Schema=lambda type=None, properties=None, description=None, required=None: _ns(
            type=type, properties=properties, description=description, required=required),
        GenerateContentConfig=lambda **kw: _ns(**kw),
    )


class _FakeGenaiErrors:
    class APIError(Exception):
        pass


class _GeminiClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_contents = None
        self.models = _ns(generate_content_stream=self._stream)

    def _stream(self, model, contents, config):
        self.last_contents = contents
        return iter(self._chunks)


def test_gemini_chunk_text_handles_raise():
    class Raising:
        @property
        def text(self):
            raise ValueError("no text part")

    assert gp._chunk_text(Raising()) is None
    assert gp._chunk_text(_ns(text="hi")) == "hi"


def test_gemini_chunk_text_reads_parts_not_accessor():
    # Text lives in parts; .text would warn (here, raise). We must not call it.
    class Boom:
        candidates = [_ns(content=_ns(parts=[_ns(text="hello", function_call=None)]))]

        @property
        def text(self):
            raise AssertionError(".text must not be accessed when parts exist")

    assert gp._chunk_text(Boom()) == "hello"


def test_gemini_chunk_text_skips_function_call_parts():
    chunk = _ns(candidates=[_ns(content=_ns(parts=[
        _ns(text="answer", function_call=None),
        _ns(text=None, function_call=_ns(name="run_command", args={})),
    ]))])
    assert gp._chunk_text(chunk) == "answer"


def test_gemini_chunk_calls_from_candidates():
    call = _ns(name="run_command", args={"command": "ls"})
    chunk = _ns(function_calls=None, candidates=[
        _ns(content=_ns(parts=[_ns(function_call=call, thought_signature=b"sig")]))])
    assert gp._chunk_calls(chunk) == [(call, b"sig")]


def test_gemini_chunk_calls_fallback_accessor_has_no_signature():
    call = _ns(name="run_command", args={"command": "ls"})
    chunk = _ns(function_calls=[call], candidates=None)
    assert gp._chunk_calls(chunk) == [(call, None)]


def test_gemini_thought_signature_round_trips():
    """A thinking model's signature is captured on the way in and replayed on
    the way out, so the follow-up turn is not rejected with a 400."""
    call = _ns(name="run_command", id="g1",
               args={"command": "ls", "explanation": "x", "interactive": False})
    part = _ns(function_call=call, thought_signature=b"sig-xyz")
    chunks = [_ns(text=None, function_calls=None,
                  candidates=[_ns(content=_ns(parts=[part]))])]
    provider = gp.GeminiProvider(
        _FakeGenaiErrors, _fake_genai_types(), _GeminiClient(chunks), "gemini-test")

    turn = provider.complete([UserMessage("go")], ["sys"], False, lambda _t: None)
    assert turn.tool_calls[0].signature == b"sig-xyz"

    contents = provider._to_contents([AssistantMessage("", turn.tool_calls)])
    assert contents[0].parts[0].thought_signature == b"sig-xyz"


def test_gemini_is_auth_error():
    assert gp._is_auth_error(_ns(code=403)) is True
    assert gp._is_auth_error(Exception("Invalid API key")) is True
    assert gp._is_auth_error(Exception("server boom")) is False


def test_gemini_to_contents():
    provider = gp.GeminiProvider(_FakeGenaiErrors, _fake_genai_types(), None, "m")
    contents = provider._to_contents([
        UserMessage("hi"),
        AssistantMessage("", [ToolInvocation("c1", "ls", "list", False)]),
        ToolResultsMessage([ToolResult("c1", "out", False)]),
    ])
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "hi"
    assert contents[1].role == "model"
    assert contents[1].parts[0].function_call.name == "run_command"
    assert contents[2].role == "tool"
    assert contents[2].parts[0].function_response.response == {"result": "out"}


def test_gemini_complete_round_trip():
    fc = _ns(name="run_command", id="g1",
             args={"command": "ls", "explanation": "x", "interactive": False})
    chunks = [
        _ns(text="hi", function_calls=[]),
        _ns(text=None, function_calls=[fc]),
    ]
    client = _GeminiClient(chunks)
    provider = gp.GeminiProvider(
        _FakeGenaiErrors, _fake_genai_types(), client, "gemini-test")
    streamed = []
    turn = provider.complete([UserMessage("go")], ["sys"], False, streamed.append)
    assert turn.text == "hi"
    assert turn.tool_calls[0].call_id == "g1"
    assert turn.tool_calls[0].command == "ls"


def test_anthropic_captures_usage():
    final = _ns(
        content=[_ns(type="text", text="ok")],
        usage=_ns(input_tokens=10, output_tokens=5,
                  cache_read_input_tokens=4, cache_creation_input_tokens=2),
    )
    provider = ap.AnthropicProvider(_AnthropicClient([], final), "m")
    turn = provider.complete([], ["sys"], False, lambda _t: None)
    assert turn.usage.input_tokens == 16   # 10 fresh + 4 read + 2 created
    assert turn.usage.output_tokens == 5
    assert turn.usage.cached_tokens == 4


def test_openai_captures_usage():
    chunks = [
        _openai_chunk(content="hi"),
        _ns(choices=[], usage=_ns(prompt_tokens=20, completion_tokens=8,
                                  prompt_tokens_details=_ns(cached_tokens=6))),
    ]
    provider = op.OpenAIProvider(_FakeOpenAIErrors, _OpenAIClient(chunks), "m")
    turn = provider.complete([UserMessage("go")], ["sys"], False, lambda _t: None)
    assert turn.usage.input_tokens == 20
    assert turn.usage.output_tokens == 8
    assert turn.usage.cached_tokens == 6
    assert provider._client.last_params["stream_options"] == {"include_usage": True}


def test_gemini_captures_usage():
    chunks = [_ns(text="hi", function_calls=None, candidates=None,
                  usage_metadata=_ns(prompt_token_count=30, candidates_token_count=12,
                                     cached_content_token_count=9))]
    provider = gp.GeminiProvider(
        _FakeGenaiErrors, _fake_genai_types(), _GeminiClient(chunks), "m")
    turn = provider.complete([UserMessage("go")], ["sys"], False, lambda _t: None)
    assert turn.usage.input_tokens == 30
    assert turn.usage.output_tokens == 12
    assert turn.usage.cached_tokens == 9


class _BoomTransient(Exception):
    status_code = 503


class _BoomPermanent(Exception):
    status_code = 400


class _FlakyProvider(base.ModelProvider):
    api_errors = (_BoomTransient, _BoomPermanent)

    def __init__(self, error):
        super().__init__("m")
        self._error = error

    def build_request(self, history, system_texts, disable_tools):
        return {}

    def run_stream(self, request, collector):
        raise self._error


@pytest.mark.parametrize("status, retryable", [(503, True), (429, True), (400, False)])
def test_is_transient_by_status_code(status, retryable):
    error = _ns(status_code=status)
    assert _FlakyProvider(None).is_transient(error) is retryable


class _Timeout(Exception):
    pass


def test_is_transient_by_declared_error_type():
    provider = _FlakyProvider(None)
    provider.transient_errors = (_Timeout,)
    assert provider.is_transient(_Timeout()) is True
    assert provider.is_transient(ValueError("nope")) is False


def test_complete_marks_transient_error_retryable():
    with pytest.raises(base.ProviderError) as exc:
        _FlakyProvider(_BoomTransient("overloaded")).complete(
            [], ["s"], False, lambda _t: None)
    assert exc.value.retryable is True


def test_complete_marks_permanent_error_not_retryable():
    with pytest.raises(base.ProviderError) as exc:
        _FlakyProvider(_BoomPermanent("bad request")).complete(
            [], ["s"], False, lambda _t: None)
    assert exc.value.retryable is False
