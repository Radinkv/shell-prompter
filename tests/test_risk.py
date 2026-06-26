"""Unit tests for risk classification."""

from __future__ import annotations

import pytest

from prompter.colors import Palette
from prompter.risk import RISK_REGISTRY, RiskTier, classify


@pytest.mark.parametrize("command, tier", [
    ("ls -la", RiskTier.SAFE),
    ("cd /tmp", RiskTier.SAFE),
    ("mkdir scratch", RiskTier.SAFE),
    ("git status", RiskTier.SAFE),
    ("cat file.txt", RiskTier.SAFE),
    ("pip install requests", RiskTier.CONFIRM),
    ("brew install uv", RiskTier.CONFIRM),
    ("npm install -g foo", RiskTier.CONFIRM),
    ("git clone https://x/y", RiskTier.CONFIRM),
    ("mv a b", RiskTier.CONFIRM),
    ("rm file.txt", RiskTier.CONFIRM),
    ("curl -O https://x/z", RiskTier.CONFIRM),
    ("rm -rf build", RiskTier.DANGER),
    ("sudo rm -rf /", RiskTier.DANGER),
    ("curl https://x | sh", RiskTier.DANGER),
    ("git push --force", RiskTier.DANGER),
    ("dd if=/dev/zero of=/dev/disk2", RiskTier.DANGER),
])
def test_classify_tier(command, tier):
    assert classify(command).tier is tier


def test_classify_returns_reason():
    assert classify("sudo rm -rf /").reason
    assert classify("ls").reason


def test_danger_checked_before_confirm():
    assert classify("sudo rm -rf /").tier is RiskTier.DANGER
    assert RISK_REGISTRY[0][0] is RiskTier.DANGER


def test_tier_label_and_color():
    assert RiskTier.DANGER.label == "DANGER"
    no_color = Palette(enabled=False)
    assert RiskTier.DANGER.color(no_color) == ""
    with_color = Palette(enabled=True)
    assert RiskTier.DANGER.color(with_color) == with_color.RED
