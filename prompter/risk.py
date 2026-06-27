"""Risk classification: grade each proposed command safe / confirm / danger.

The tier decides whether prompter runs a command automatically or asks first.
RISK_REGISTRY lists the tiers in precedence order so the loop in classify() is
trivial and the precedence lives in data, not control flow. The first matching
rule wins (danger before confirm).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .colors import Palette

_COLOR_GREEN = "GREEN"
_COLOR_YELLOW = "YELLOW"
_COLOR_RED = "RED"

SAFE_REASON = "read-only or clearly reversible"


class RiskTier(Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    DANGER = "danger"

    @property
    def label(self) -> str:
        return self.name

    def color(self, palette: Palette) -> str:
        return getattr(palette, _TIER_COLOR_ATTR[self])


_TIER_COLOR_ATTR = {
    RiskTier.SAFE: _COLOR_GREEN,
    RiskTier.CONFIRM: _COLOR_YELLOW,
    RiskTier.DANGER: _COLOR_RED,
}


@dataclass(frozen=True)
class RiskRule:
    pattern: re.Pattern
    reason: str


@dataclass(frozen=True)
class RiskAssessment:
    tier: RiskTier
    reason: str


def _rule(regex: str, reason: str) -> RiskRule:
    return RiskRule(re.compile(regex), reason)


DANGER_RULES = [
    _rule(r"\brm\b[^|;&]*(\s-[a-z]*[rf]|\s--(recursive|force)\b)",
          "recursive/forced delete"),
    _rule(r"\bfind\b[^|;&]*\s-delete\b", "find -delete removes matched files"),
    _rule(r"\bfind\b[^|;&]*-exec\b[^|;&]*\brm\b", "find -exec rm deletes files"),
    _rule(r"\bsudo\b", "runs with root privileges"),
    _rule(r"\b(doas|su|pkexec)\b", "runs with elevated privileges"),
    _rule(r"\b(shutdown|reboot|halt|poweroff)\b", "powers off or restarts the machine"),
    _rule(r"\bshred\b", "irrecoverably destroys a file"),
    _rule(r"\bmkfs\b", "formats a filesystem"),
    _rule(r"\bdd\b\s+.*of=", "raw disk write with dd"),
    _rule(r">\s*/dev/(sd|disk|nvme|hd|vd|mmcblk)", "writes directly to a disk device"),
    _rule(r"\bchmod\b[^|;&]*(\s-[a-z]*R\b|\s--recursive\b)", "recursive permission change"),
    _rule(r"\bchown\b[^|;&]*(\s-[a-z]*R\b|\s--recursive\b)", "recursive ownership change"),
    _rule(r":\(\)\s*\{.*\|.*&\s*\}", "looks like a fork bomb"),
    _rule(r"\bgit\b.*\bpush\b.*(--force|-f)\b",
          "force-push (rewrites remote history)"),
    _rule(r"(curl|wget)\b[^|]*\|\s*(sudo\s+)?"
          r"(sh|bash|zsh|fish|python3?|perl|ruby|node)\b",
          "downloads and executes code from the internet"),
    _rule(r"\|\s*(sudo\s+)?(sh|bash|zsh|fish)\b", "pipes data into a shell to execute"),
    _rule(r"\beval\b", "evaluates a constructed command string"),
    _rule(r"\brm\b[^|;&]*\s+(/|~|\$HOME)\s*$", "deletes your home or root"),
]

CONFIRM_RULES = [
    _rule(r"\b(brew|apt|apt-get|yum|dnf|pacman|port)\b\s+(install|add|upgrade)",
          "installs system packages"),
    _rule(r"\b(pip|pip3|pipx|uv)\b.*\binstall\b", "installs Python packages"),
    _rule(r"\bnpm\b\s+(install|i|add)\b", "installs npm packages"),
    _rule(r"\b(pnpm|yarn|bun)\b\s+(install|add|i)\b", "installs JS packages"),
    _rule(r"\bcargo\b\s+install\b", "installs a Rust crate"),
    _rule(r"\bgo\b\s+install\b", "installs a Go binary"),
    _rule(r"\bgit\b\s+clone\b", "clones a repository"),
    _rule(r"\bgit\b\s+push\b", "pushes to a remote"),
    _rule(r"\b(curl|wget)\b", "downloads from the internet"),
    _rule(r"\bmv\b", "moves/renames files"),
    _rule(r"\brm\b", "deletes files"),
    _rule(r"\bkill(all)?\b", "terminates processes"),
    _rule(r"\bgit\b\s+(reset|checkout|restore)\b.*--hard",
          "discards local changes"),
    _rule(r"\bgit\b\s+clean\b[^|;&]*\s-[a-z]*f", "deletes untracked files"),
    _rule(r"\bdocker\b\s+(run|rm|rmi|system\s+prune)", "modifies Docker state"),
    _rule(r"\btee\b\s+(-[a-z]+\s+)*[^|&\s]", "writes/overwrites a file via tee"),
    _rule(r">\s*(?!/dev/(null|stdout|stderr)\b)[^|&\s]",
          "redirects output into a file (may overwrite)"),
]

RISK_REGISTRY = [
    (RiskTier.DANGER, DANGER_RULES),
    (RiskTier.CONFIRM, CONFIRM_RULES),
]


def classify(command: str) -> RiskAssessment:
    """Grade a command into its risk tier with a short reason."""
    for tier, rules in RISK_REGISTRY:
        for rule in rules:
            if rule.pattern.search(command):
                return RiskAssessment(tier, rule.reason)
    return RiskAssessment(RiskTier.SAFE, SAFE_REASON)
