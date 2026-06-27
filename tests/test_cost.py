"""Unit tests for cost estimation."""

from __future__ import annotations

from prompter.cost import estimate
from prompter.providers.base import Usage


def test_none_without_configured_price():
    assert estimate("some-model", Usage(1000, 1000, 0), {}) is None


def test_estimate_uses_configured_rates():
    pricing = {"m": {"input": 3.0, "output": 15.0}}
    # 1M input @ $3 + 1M output @ $15 = $18
    cost = estimate("m", Usage(1_000_000, 1_000_000, 0), pricing)
    assert cost == 18.0


def test_missing_rate_defaults_to_zero():
    cost = estimate("m", Usage(1_000_000, 1_000_000, 0), {"m": {"input": 2.0}})
    assert cost == 2.0  # output rate absent -> treated as 0
