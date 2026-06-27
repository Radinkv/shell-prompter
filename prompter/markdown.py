"""Render a useful subset of Markdown to ANSI as the model's answer streams in.

The answer arrives a few characters at a time through Console.stream_text. This
module releases styled output as soon as it is sure of it, rather than waiting
for the end of a line.

Two facts decide when output is safe to release.

The first fact is about block prefixes. A heading, a list item, a quote, and a
fenced code block are all decided by the characters at the very start of a line,
and they change how the whole line looks. So a line that turns out to be one of
these is held until its newline arrives and then rendered in one piece. These
lines are short, so the wait is not visible.

The second fact is about inline emphasis. Bold, italic, and inline code each
need a closing marker before their styling is known. An opening star with no
closing star on the same line is just a literal star. So plain text is released
the moment it arrives, and a run that sits inside an open marker is held until
that marker closes or the line ends. When emphasis is nested, the inner run can
not commit its final style until the outer marker closes, so the hold extends to
the outermost marker.

The renderer never asks how wide the terminal is and never inserts its own line
breaks. The terminal wraps the styled lines on its own, so resizing the window
in the middle of a stream stays correct.
"""

from __future__ import annotations

import re

from .colors import Palette
from .constants import EMPTY, NEWLINE

_BOLD = "**"
_ITALIC = "*"
_BACKTICK = "`"

_TOK_TEXT = "text"
_TOK_CODE = "code"
_TOK_DELIM = "delim"

_KIND_BLOCK = "block"
_KIND_PARAGRAPH = "paragraph"
_KIND_UNDECIDED = "more"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_QUOTE_RE = re.compile(r"^(\s*)>\s?(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d+\.)\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*```")
_FENCE_MARK = "```"

_MAX_HEADING_HASHES = 6
_BULLET = "• "
_QUOTE_GUTTER = "│ "


def _is_space(char: str) -> bool:
    return char == EMPTY or char.isspace()


def _tokenize(text: str) -> list:
    """Split one finished line into text, code spans, and emphasis markers.

    The scan walks the line one character at a time. Most characters are plain
    text, so they collect in a buffer that is flushed as a single text token
    whenever a special character interrupts it.

    A backtick begins a code span. The scan looks ahead for the next backtick on
    the same line. When it finds one, everything between the two backticks
    becomes a single code token, and the scan jumps past the closing backtick.
    Code is taken as one atomic piece on purpose, because emphasis markers inside
    code are meant to be literal. A backtick with no partner left on the line is
    just an ordinary character.

    A star begins an emphasis marker. Two stars in a row mean bold and one star
    means italic. Each marker token also records two yes or no facts about the
    characters that touch it. It can open only when the character on its right is
    not a space, and it can close only when the character on its left is not a
    space. These two facts are what keep an expression like a star b star from
    turning into emphasis, because those stars are surrounded by spaces and so
    they can neither open nor close.

    The underscore is left out of emphasis on purpose. It shows up constantly
    inside names and paths such as my_var and src_tauri, where emphasis is never
    intended, so treating it as a marker would do more harm than good.
    """
    tokens: list = []
    buf: list = []
    index, length = 0, len(text)

    def flush():
        if buf:
            tokens.append((_TOK_TEXT, EMPTY.join(buf)))
            buf.clear()

    while index < length:
        char = text[index]
        if char == _BACKTICK:
            close = text.find(_BACKTICK, index + 1)
            if close != -1:
                flush()
                tokens.append((_TOK_CODE, text[index + 1:close]))
                index = close + 1
                continue
        elif char == _ITALIC:
            marker = _BOLD if text[index:index + 2] == _BOLD else _ITALIC
            span = len(marker)
            left = text[index - 1] if index > 0 else EMPTY
            right = text[index + span] if index + span < length else EMPTY
            flush()
            tokens.append((_TOK_DELIM, marker,
                           not _is_space(right), not _is_space(left)))
            index += span
            continue
        buf.append(char)
        index += 1
    flush()
    return tokens


def _match_delimiters(tokens: list) -> set:
    """Return the indices of the markers that pair up into real emphasis.

    The scan keeps a stack of markers that have opened but not yet closed. A
    marker that can close and matches the kind on top of the stack pairs with it,
    and both indices are recorded as matched. A marker that can open is pushed to
    wait for its partner. Any marker that does neither is left alone and will be
    shown as a literal character. This is the nearest match rule, which is simple
    and is correct for the common cases.
    """
    matched: set = set()
    stack: list = []
    for index, token in enumerate(tokens):
        if token[0] != _TOK_DELIM:
            continue
        _, kind, can_open, can_close = token
        if can_close and stack and stack[-1][0] == kind:
            matched.add(stack.pop()[1])
            matched.add(index)
        elif can_open:
            stack.append((kind, index))
    return matched


def _style_inline(text: str, c: Palette) -> str:
    """Turn one finished, self contained run of text into styled output."""
    if not text:
        return EMPTY
    tokens = _tokenize(text)
    matched = _match_delimiters(tokens)
    style_for = {_BOLD: c.BOLD, _ITALIC: c.ITALIC}
    active: list = []
    out: list = []

    def restyle() -> str:
        return c.RESET + EMPTY.join(active)

    for index, token in enumerate(tokens):
        kind = token[0]
        if kind == _TOK_TEXT:
            out.append(token[1])
        elif kind == _TOK_CODE:
            out.append(c.CYAN + token[1] + restyle())
        elif index in matched:
            code = style_for[token[1]]
            active.remove(code) if code in active else active.append(code)
            out.append(restyle())
        else:
            out.append(token[1])
    out.append(c.RESET)
    return EMPTY.join(out)


def _held_from(line: str) -> int:
    """Find where a partial line stops being safe to release.

    Everything before the returned index is final and can be shown now.
    Everything from that index onward depends on characters that have not
    arrived yet, so it must wait.

    The scan mirrors the marker logic but works on a line that is still growing.
    It keeps a stack of markers that have opened but not closed. The first marker
    still on the stack at the end is the earliest point that is not yet settled,
    because more text could still close it and change its styling. If a code span
    has opened but its closing backtick has not arrived, that opening backtick is
    the unsettled point. If the line ends right on a star, that star is held too,
    because the next character decides whether it can open at all.
    """
    stack: list = []
    index, length = 0, len(line)
    while index < length:
        char = line[index]
        if char == _BACKTICK:
            close = line.find(_BACKTICK, index + 1)
            if close == -1:
                return stack[0][1] if stack else index
            index = close + 1
            continue
        if char == _ITALIC:
            marker = _BOLD if line[index:index + 2] == _BOLD else _ITALIC
            span = len(marker)
            if index + span >= length:
                return stack[0][1] if stack else index
            can_open = not _is_space(line[index + span])
            can_close = not _is_space(line[index - 1] if index > 0 else EMPTY)
            if can_close and stack and stack[-1][0] == marker:
                stack.pop()
            elif can_open:
                stack.append((marker, index))
            index += span
            continue
        index += 1
    if stack:
        return stack[0][1]
    return length


def _classify(line: str) -> str:
    """Decide whether a line so far is a block, a paragraph, or still unknown.

    A block is a line whose meaning comes from a marker at its start, such as a
    heading, a quote, a list item, or a fence. A paragraph is ordinary text. The
    answer is unknown when the characters seen so far could still go either way,
    in which case the caller waits for more.

    The leading spaces are skipped first. An empty remainder is unknown. After
    that the first character chooses a small family of possibilities and a short
    look at the next characters settles it. A hash can lead to a heading once a
    space follows, but a hash glued to a word is a paragraph. A greater than sign
    is always a quote. A dash, plus, or star followed by a space is a list item,
    while the same character followed by anything else is a paragraph. Digits
    that lead into a dot and a space are a numbered item. Three backticks are a
    fence, while one or two backticks alone are still unknown because a third
    could follow.
    """
    stripped = line.lstrip(" ")
    if stripped == EMPTY:
        return _KIND_UNDECIDED
    head = stripped[0]
    if head == ">":
        return _KIND_BLOCK
    if head == "#":
        hashes = len(stripped) - len(stripped.lstrip("#"))
        if hashes > _MAX_HEADING_HASHES:
            return _KIND_PARAGRAPH
        rest = stripped[hashes:]
        if rest == EMPTY:
            return _KIND_UNDECIDED
        return _KIND_BLOCK if rest[0] == " " else _KIND_PARAGRAPH
    if head in "-+*":
        if len(stripped) == 1:
            return _KIND_UNDECIDED
        return _KIND_BLOCK if stripped[1] == " " else _KIND_PARAGRAPH
    if head == _BACKTICK:
        if stripped.startswith(_FENCE_MARK):
            return _KIND_BLOCK
        if stripped == _BACKTICK or stripped == "``":
            return _KIND_UNDECIDED
        return _KIND_PARAGRAPH
    if head.isdigit():
        digits = len(stripped) - len(stripped.lstrip("0123456789"))
        rest = stripped[digits:]
        if rest == EMPTY or rest == ".":
            return _KIND_UNDECIDED
        if rest.startswith(". "):
            return _KIND_BLOCK
        return _KIND_PARAGRAPH
    return _KIND_PARAGRAPH


class MarkdownStream:
    """Holds the state needed to render Markdown as it streams in.

    The renderer remembers the text of the line it is on, how much of that line
    it has already shown, what kind of line it decided it is, and whether it is
    inside a fenced code block.
    """

    def __init__(self, palette: Palette):
        self.c = palette
        self._line = EMPTY
        self._shown = 0
        self._kind: str | None = None
        self._in_fence = False

    def feed(self, chunk: str | None) -> str:
        if not chunk:
            return EMPTY
        out = []
        while True:
            newline = chunk.find(NEWLINE)
            if newline == -1:
                self._line += chunk
                break
            self._line += chunk[:newline]
            chunk = chunk[newline + 1:]
            out.append(self._finish_line())
        out.append(self._release())
        return EMPTY.join(out)

    def flush(self) -> str:
        if self._kind is None:
            self._kind = self._settle()
        rest = self._line[self._shown:]
        rendered = (_style_inline(rest, self.c)
                    if self._kind == _KIND_PARAGRAPH else self._render(self._line))
        self._reset()
        return rendered

    def _settle(self) -> str:
        if self._in_fence:
            return _KIND_BLOCK
        return _KIND_BLOCK if _classify(self._line) == _KIND_BLOCK else _KIND_PARAGRAPH

    def _release(self) -> str:
        """Show whatever part of the current unfinished line is now safe."""
        if self._in_fence:
            self._kind = _KIND_BLOCK
        if self._kind is None:
            verdict = _classify(self._line)
            if verdict == _KIND_UNDECIDED:
                return EMPTY
            self._kind = _KIND_BLOCK if verdict == _KIND_BLOCK else _KIND_PARAGRAPH
        if self._kind == _KIND_BLOCK:
            return EMPTY
        boundary = _held_from(self._line)
        if boundary <= self._shown:
            return EMPTY
        ready = self._line[self._shown:boundary]
        self._shown = boundary
        return _style_inline(ready, self.c)

    def _finish_line(self) -> str:
        if self._kind is None:
            self._kind = self._settle()
        rest = self._line[self._shown:]
        rendered = (_style_inline(rest, self.c)
                    if self._kind == _KIND_PARAGRAPH else self._render(self._line))
        self._reset()
        return rendered + NEWLINE

    def _reset(self) -> None:
        self._line = EMPTY
        self._shown = 0
        self._kind = None

    def _render(self, line: str) -> str:
        c = self.c
        if _FENCE_RE.match(line):
            self._in_fence = not self._in_fence
            return c.DIM + line + c.RESET
        if self._in_fence:
            return c.DIM + line + c.RESET

        heading = _HEADING_RE.match(line)
        if heading:
            return c.BOLD + _style_inline(heading.group(2), c) + c.RESET
        quote = _QUOTE_RE.match(line)
        if quote:
            return (quote.group(1) + c.DIM + _QUOTE_GUTTER
                    + _style_inline(quote.group(2), c) + c.RESET)
        bullet = _BULLET_RE.match(line)
        if bullet:
            return bullet.group(1) + _BULLET + _style_inline(bullet.group(2), c)
        ordered = _ORDERED_RE.match(line)
        if ordered:
            return (ordered.group(1) + ordered.group(2) + " "
                    + _style_inline(ordered.group(3), c))
        return _style_inline(line, c)
