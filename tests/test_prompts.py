"""Unit tests for system-prompt assembly and the concise toggle."""

from __future__ import annotations

from prompter.config import Config
from prompter.prompts import CONCISE_INSTRUCTION, SYSTEM_PROMPT, build_system_texts


def test_system_texts_start_with_base_prompt():
    texts = build_system_texts(Config(), "/work")
    assert texts[0] == SYSTEM_PROMPT


def test_concise_off_by_default():
    assert CONCISE_INSTRUCTION not in build_system_texts(Config(), "/work")


def test_concise_on_adds_instruction_before_environment():
    texts = build_system_texts(Config(concise=True), "/work")
    assert CONCISE_INSTRUCTION in texts
    # The instruction is stable, so it sits ahead of the volatile env block (last).
    assert texts.index(CONCISE_INSTRUCTION) < len(texts) - 1
