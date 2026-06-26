"""Unit tests for the prompt/context builders and the ClaudeClient wrapper."""

from __future__ import annotations

import types

from prompter.config import Config
from prompter.constants import (
    DELTA_TEXT,
    EVENT_CONTENT_BLOCK_DELTA,
    TOOL_CHOICE_NONE,
)
from prompter.llm import (
    RUN_TOOL,
    SYSTEM_PROMPT,
    ClaudeClient,
    build_environment_context,
    build_system_blocks,
)


def test_environment_context_includes_workspace_and_prefs():
    cfg = Config(default_workspace="~/Proj", preferences=["use pnpm"])
    ctx = build_environment_context(cfg, "/some/dir")
    assert "/some/dir" in ctx
    assert cfg.workspace_path in ctx
    assert "- use pnpm" in ctx


def test_environment_context_no_prefs():
    ctx = build_environment_context(Config(preferences=[]), "/d")
    assert "(none)" in ctx


def test_system_blocks_shape():
    blocks = build_system_blocks("CTX")
    assert len(blocks) == 2
    assert blocks[0]["text"] == SYSTEM_PROMPT
    assert blocks[1]["text"] == "CTX"
    assert all(b["type"] == "text" for b in blocks)


def test_run_tool_schema():
    assert RUN_TOOL["name"] == "run_command"
    schema = RUN_TOOL["input_schema"]
    assert set(schema["properties"]) == {"command", "explanation", "interactive"}
    assert schema["required"] == ["command", "explanation"]


class _FakeStream:
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


class _FakeMessages:
    def __init__(self):
        self.last_kwargs = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        event = types.SimpleNamespace(
            type=EVENT_CONTENT_BLOCK_DELTA,
            delta=types.SimpleNamespace(type=DELTA_TEXT, text="hi "),
        )
        final = types.SimpleNamespace(stop_reason="end_turn", content=[])
        return _FakeStream([event, event], final)


class _FakeAnthropic:
    def __init__(self):
        self.messages = _FakeMessages()


def test_stream_reports_text_and_returns_final():
    fake = _FakeAnthropic()
    client = ClaudeClient(fake, "claude-test")
    collected = []
    final = client.stream([], [], disable_tools=False, on_text=collected.append)
    assert collected == ["hi ", "hi "]
    assert final.stop_reason == "end_turn"
    assert fake.messages.last_kwargs["model"] == "claude-test"
    assert "tool_choice" not in fake.messages.last_kwargs


def test_stream_disables_tools():
    fake = _FakeAnthropic()
    ClaudeClient(fake, "m").stream([], [], disable_tools=True, on_text=lambda _: None)
    assert fake.messages.last_kwargs["tool_choice"] == TOOL_CHOICE_NONE
