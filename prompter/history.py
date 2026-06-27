"""Context compaction: shrink the history re-sent to the model each turn.

A multi-step task resends the whole conversation on every turn, so old command
outputs pile up as input tokens. compact() keeps the goal, the most recent tool
results, and any failures verbatim, and collapses older *successful* outputs to
their first line (the exit-code header) with the body elided. The model
already reacted to the full output when the command ran; a later turn only needs
to know the step succeeded.

It is a pure transform applied just before a provider call. The stored
conversation is never mutated, so nothing is lost from the program's own record;
only the copy handed to the model is slimmed. Failures survive untouched because
self-repair reads their error text, and short outputs survive because collapsing
them would save nothing.
"""

from __future__ import annotations

from .constants import NEWLINE
from .providers.base import HistoryItem, ToolResult, ToolResultsMessage

# Tool-result messages kept verbatim, counting back from the latest turn.
DEFAULT_KEEP_RECENT = 2
# Successful outputs at or under this length are left alone; collapsing a short
# output would replace it with a marker of about the same size.
COLLAPSE_THRESHOLD = 240
_ELISION = "... [output elided to save context] ..."


def compact(history: list[HistoryItem],
            keep_recent: int = DEFAULT_KEEP_RECENT) -> list[HistoryItem]:
    """Return a copy of history with old, large, successful outputs collapsed."""
    total = sum(isinstance(item, ToolResultsMessage) for item in history)
    seen = 0
    out: list[HistoryItem] = []
    for item in history:
        if isinstance(item, ToolResultsMessage):
            seen += 1
            recent = seen > total - keep_recent
            out.append(item if recent else _collapse_message(item))
        else:
            out.append(item)
    return out


def _collapse_message(message: ToolResultsMessage) -> ToolResultsMessage:
    return ToolResultsMessage([_collapse_result(r) for r in message.results])


def _collapse_result(result: ToolResult) -> ToolResult:
    if result.is_error or len(result.content) <= COLLAPSE_THRESHOLD:
        return result
    return ToolResult(result.call_id, _shorten(result.content), result.is_error)


def _shorten(content: str) -> str:
    """Keep the first line (the exit-code header) and elide the rest.

    A single-line payload (no header/body split) has no body to elide, so cap it
    by character budget instead of growing it with a marker.
    """
    head, separator, _body = content.partition(NEWLINE)
    if not separator:
        return content[:COLLAPSE_THRESHOLD]
    return head + NEWLINE + _ELISION
