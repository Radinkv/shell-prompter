"""Unit tests for context compaction (history.compact)."""

from __future__ import annotations

from prompter.history import COLLAPSE_THRESHOLD, _ELISION, compact
from prompter.providers.base import (
    AssistantMessage,
    ToolInvocation,
    ToolResult,
    ToolResultsMessage,
    UserMessage,
)


def _big(prefix: str = "exit_code: 0", last: str = "Successfully installed x") -> str:
    middle = "\n".join(f"line {i}" for i in range(200))
    return f"{prefix}\n{middle}\n{last}"


def _results(content: str, is_error: bool = False) -> ToolResultsMessage:
    return ToolResultsMessage([ToolResult("c1", content, is_error)])


def test_recent_results_kept_verbatim():
    big = _big()
    history = [_results(big), _results(big)]
    out = compact(history, keep_recent=2)
    assert out[0].results[0].content == big
    assert out[1].results[0].content == big


def test_old_large_success_is_collapsed():
    history = [_results(_big()), _results("recent"), _results("recent")]
    out = compact(history, keep_recent=2)
    collapsed = out[0].results[0].content
    assert collapsed == "exit_code: 0\n" + _ELISION  # header kept, body elided
    assert "line 100" not in collapsed
    assert "Successfully installed x" not in collapsed


def test_collapse_on_real_payload_shape():
    """The agent's payload ends with a (often empty) stderr section, so the
    collapse must not depend on the last line being meaningful."""
    from prompter.agent import _PAYLOAD_TEMPLATE
    stdout = "\n".join(["Collecting pkg"] * 60 + ["Successfully installed pkg"])
    payload = _PAYLOAD_TEMPLATE.format(exit_code=0, cwd="/x", stdout=stdout, stderr="")
    out = compact([_results(payload), _results("a"), _results("b")], keep_recent=2)
    collapsed = out[0].results[0].content
    assert collapsed == "exit_code: 0\n" + _ELISION
    assert len(collapsed) < len(payload)             # actually shrank
    assert "stderr:" not in collapsed                # no useless trailing label


def test_old_failure_kept_verbatim():
    failure = _big(prefix="exit_code: 1", last="error: boom")
    history = [_results(failure, is_error=True), _results("a"), _results("b")]
    out = compact(history, keep_recent=2)
    assert out[0].results[0].content == failure       # self-repair needs failures


def test_short_success_left_alone():
    short = "exit_code: 0\nok"
    assert len(short) <= COLLAPSE_THRESHOLD
    history = [_results(short), _results("a"), _results("b")]
    out = compact(history, keep_recent=2)
    assert out[0].results[0].content == short


def test_non_tool_messages_pass_through():
    sig_call = ToolInvocation("c1", "ls", signature=b"sig")
    history = [
        UserMessage("goal"),
        AssistantMessage("thinking", [sig_call]),
        _results(_big()),
        _results("recent"),
        _results("recent"),
    ]
    out = compact(history, keep_recent=2)
    assert out[0] == history[0]
    assert out[1] is history[1]                        # assistant turn untouched
    assert out[1].tool_calls[0].signature == b"sig"    # gemini replay intact


def test_does_not_mutate_input():
    big = _big()
    history = [_results(big), _results("a"), _results("b")]
    compact(history, keep_recent=2)
    assert history[0].results[0].content == big        # original record preserved


def test_empty_history():
    assert compact([]) == []
