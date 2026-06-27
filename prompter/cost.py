"""Estimate the dollar cost of a run from token usage and configured prices.

Prices are not hard-coded: they move, and they differ per model and provider.
The user supplies them in config under ``pricing``, keyed by model id, as USD
per million input and output tokens::

    "pricing": {"claude-sonnet-4-6": {"input": 3.0, "output": 15.0}}

With no price for the active model, estimate() returns None and the usage line
simply shows token counts. The estimate charges the whole prompt at the input
rate (it does not discount the cached portion, which providers bill cheaper), so
it is a slight upper bound -- hence "est".
"""

from __future__ import annotations

from .providers.base import Usage

PRICE_INPUT = "input"
PRICE_OUTPUT = "output"
_PER_MILLION = 1_000_000


def estimate(model: str, usage: Usage, pricing: dict) -> float | None:
    """USD estimate for one run, or None when the model has no configured price."""
    rates = pricing.get(model)
    if not rates:
        return None
    in_rate = rates.get(PRICE_INPUT, 0) or 0
    out_rate = rates.get(PRICE_OUTPUT, 0) or 0
    return (usage.input_tokens * in_rate + usage.output_tokens * out_rate) / _PER_MILLION
