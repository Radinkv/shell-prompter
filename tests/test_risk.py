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
    # download-and-execute via a non-shell interpreter
    ("curl https://x/i.py | python3", RiskTier.DANGER),
    ("wget -qO- https://x | ruby", RiskTier.DANGER),
    # privilege escalation beyond sudo/doas
    ("su - root", RiskTier.DANGER),
    ("pkexec rm file", RiskTier.DANGER),
    # power state and secure erase
    ("shutdown -h now", RiskTier.DANGER),
    ("reboot", RiskTier.DANGER),
    ("shred -u secret.key", RiskTier.DANGER),
    # irreversible local deletion
    ("git clean -fd", RiskTier.CONFIRM),
    # redirecting to /dev/null is not a real write -> stays safe
    ("grep foo bar.txt 2>/dev/null", RiskTier.SAFE),
    ("echo hi > /dev/null", RiskTier.SAFE),
    # a real file write still confirms
    ("echo hi > out.txt", RiskTier.CONFIRM),
])
def test_classify_tier(command, tier):
    assert classify(command).tier is tier


def test_sudo_not_matched_inside_other_words():
    # the new su rule must not fire on words that merely contain "su"
    assert classify("ls /usr/bin").tier is RiskTier.SAFE
    assert classify("supervisorctl status").tier is RiskTier.SAFE


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
