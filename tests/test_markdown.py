"""Unit tests for the streaming Markdown-to-ANSI renderer."""

from __future__ import annotations

import pytest

from prompter.colors import Palette
from prompter.markdown import MarkdownStream, _style_inline

C = Palette(enabled=True)
OFF = Palette(enabled=False)


def test_bold_and_italic_and_code():
    assert C.BOLD in _style_inline("**hi**", C)
    assert C.ITALIC in _style_inline("*hi*", C)
    code = _style_inline("`x`", C)
    assert C.CYAN in code and "x" in code


def test_unmatched_delimiter_is_literal():
    out = _style_inline("*hi", C)
    assert C.ITALIC not in out
    assert "*hi" in out


def test_flanking_keeps_spaced_stars_literal():
    out = _style_inline("a * b", C)
    assert C.ITALIC not in out
    assert "a * b" in out


def test_nested_emphasis():
    out = _style_inline("**a *b* c**", C)
    assert C.BOLD in out and C.ITALIC in out


def test_code_span_protects_inner_stars():
    out = _style_inline("`a*b*c`", C)
    assert C.ITALIC not in out
    assert C.CYAN in out


def test_color_off_strips_to_plain_text():
    assert _style_inline("**hi**", OFF) == "hi"


def _render(line: str) -> str:
    return MarkdownStream(C)._render(line)


def test_heading_is_bold():
    out = _render("## Key Insights")
    assert C.BOLD in out and "Key Insights" in out and "#" not in out


def test_bullet_marker():
    assert _render("- item").startswith("• ") or "• item" in _render("- item")


def test_ordered_keeps_number():
    assert "1. item" in _render("1. item")


def test_blockquote_gutter():
    out = _render("> quoted")
    assert "│" in out and "quoted" in out


def test_plain_text_releases_immediately():
    md = MarkdownStream(C)
    assert "plain " in md.feed("plain ")


def test_open_span_is_held_then_flushed():
    md = MarkdownStream(C)
    md.feed("plain ")
    assert "held" not in md.feed("*held")
    assert "held" in md.flush()


def test_span_releases_when_it_closes_before_the_newline():
    md = MarkdownStream(C)
    md.feed("see ")
    assert C.BOLD not in md.feed("**bo")
    assert C.BOLD in md.feed("ld** ")


def test_bold_assembles_across_chunks_and_ends_line():
    md = MarkdownStream(C)
    assert md.feed("**bo") == ""
    out = md.feed("ld**\n")
    assert C.BOLD in out and out.endswith("\n")


def test_fenced_code_block_hides_markers_and_keeps_content_plain():
    md = MarkdownStream(C)
    out = md.feed("```text\nx = a *b* c\n```\n")
    assert "```" not in out
    assert "text" not in out
    assert "x = a *b* c" in out
    assert C.ITALIC not in out
    assert C.DIM not in out


def test_fenced_block_keeps_markdown_source_literal():
    md = MarkdownStream(C)
    out = md.feed("```\n# Not a heading\n**not bold**\n```\n")
    assert "# Not a heading" in out
    assert "**not bold**" in out
    assert C.BOLD not in out


def test_horizontal_rule_becomes_a_divider():
    md = MarkdownStream(C)
    out = md.feed("---\n")
    assert "---" not in out
    assert "─" in out and C.DIM in out


def test_dashes_with_text_stay_literal():
    md = MarkdownStream(C)
    out = md.feed("-- a note\n")
    assert "-- a note" in out
    assert "─" not in out
