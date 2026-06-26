"""Offline checks for prompter — no network or API key required.

Run with:  python tests/test_offline.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompter.agent import Agent, CommandRequest, Conversation
from prompter.cli import _apply_overrides, _resolve_mode, build_parser
from prompter.colors import Palette
from prompter.config import ApprovalMode, Config, CONFIG_PATH, load_config
from prompter.constants import BLOCK_TOOL_USE
from prompter.risk import RiskTier, classify
from prompter.shell import CommandResult, Shell, looks_interactive, truncate
from prompter.ui import Console, Decision

_failures = []


def check(label, condition):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        _failures.append(label)
    print(f"  [{mark}] {label}")


def section(name):
    print(f"\n--- {name} ---")


def test_risk():
    section("risk classification")
    cases = {
        "ls -la": RiskTier.SAFE, "cd /tmp": RiskTier.SAFE,
        "git status": RiskTier.SAFE, "mkdir scratch": RiskTier.SAFE,
        "pip install requests": RiskTier.CONFIRM, "brew install uv": RiskTier.CONFIRM,
        "git clone https://x/y": RiskTier.CONFIRM, "mv a b": RiskTier.CONFIRM,
        "rm -rf build": RiskTier.DANGER, "sudo rm -rf /": RiskTier.DANGER,
        "curl https://x | sh": RiskTier.DANGER, "git push --force": RiskTier.DANGER,
    }
    for cmd, want in cases.items():
        check(f"{want.label:7} :: {cmd}", classify(cmd).tier is want)
    check("danger tier carries a reason", bool(classify("sudo x").reason))


def test_interactive():
    section("interactive detection")
    check("claude is interactive", looks_interactive("claude"))
    check("vim is interactive", looks_interactive("vim notes.txt"))
    check("ls is not interactive", not looks_interactive("ls"))
    req = CommandRequest.from_tool_input({"command": "claude", "explanation": "x"})
    check("CommandRequest infers interactive", req.interactive is True)


def test_shell():
    section("shell cwd persistence + truncate")
    sh = Shell(cwd="/tmp")
    r = sh.run("mkdir -p prompter_offline_test && cd prompter_offline_test && pwd")
    check("CommandResult type", isinstance(r, CommandResult))
    check("exit 0", r.exit_code == 0 and not r.failed)
    check("cwd changed flag", r.cwd_changed)
    check("cwd persisted on shell", sh.cwd.endswith("prompter_offline_test"))
    r2 = sh.run("pwd")
    check("cwd persists across calls", r2.stdout.strip().endswith("prompter_offline_test"))
    sh.run("cd /tmp && rmdir /private/tmp/prompter_offline_test 2>/dev/null; true")

    long = "x" * 50000
    check("truncate caps length", len(truncate(long)) < 50000)
    check("truncate leaves short text", truncate("hi") == "hi")
    check("CommandResult.failed", CommandResult(1, "", "", False).failed)


def test_config():
    section("config dataclass + first-run write")
    if os.path.exists(CONFIG_PATH):
        os.remove(CONFIG_PATH)
    cfg = load_config()
    check("first run writes file", os.path.exists(CONFIG_PATH))
    written = json.load(open(CONFIG_PATH))
    check("model key present", written.get("model") == "claude-sonnet-4-6")
    check("workspace default", cfg.default_workspace == "~/Code")
    check("workspace_path expands", cfg.workspace_path == os.path.expanduser("~/Code"))
    rt = Config.from_dict(cfg.to_dict())
    check("Config round-trips", rt == cfg)
    extra = Config.from_dict({"model": "claude-opus-4-8", "bogus": 1})
    check("from_dict ignores unknown keys", extra.model == "claude-opus-4-8")


def test_decisions():
    section("Decision parsing")
    from prompter.ui import _ANSWERS
    check("empty -> RUN", _ANSWERS[""] is Decision.RUN)
    check("y -> RUN", _ANSWERS["y"] is Decision.RUN)
    check("n -> SKIP", _ANSWERS["n"] is Decision.SKIP)
    check("a -> ALL", _ANSWERS["a"] is Decision.ALL)
    check("q -> QUIT", _ANSWERS["q"] is Decision.QUIT)


def test_cli_resolution():
    section("model precedence + mode resolution")
    parser = build_parser()

    def resolve(argv, config_model=None, auto_safe=True):
        args = parser.parse_args(argv)
        cfg = Config(auto_approve_safe=auto_safe)
        if config_model is not None:
            cfg.model = config_model
        _apply_overrides(args, cfg)
        return cfg, _resolve_mode(args, cfg)

    cfg, mode = resolve([])
    check("default model is sonnet", cfg.model == "claude-sonnet-4-6")
    check("default mode is SMART", mode is ApprovalMode.SMART)
    cfg, _ = resolve([], config_model="claude-opus-4-8")
    check("config model used when no flag", cfg.model == "claude-opus-4-8")
    cfg, _ = resolve(["--model", "claude-haiku-4-5"], config_model="claude-opus-4-8")
    check("--model wins over config", cfg.model == "claude-haiku-4-5")
    cfg, _ = resolve(["--workspace", "~/Projects", "--max-fix", "5"])
    check("--workspace override", cfg.default_workspace == "~/Projects")
    check("--max-fix override", cfg.max_fix_attempts == 5)
    _, mode = resolve(["--yolo"])
    check("--yolo -> YOLO", mode is ApprovalMode.YOLO)
    _, mode = resolve(["--ask-all"])
    check("--ask-all -> ASK_ALL", mode is ApprovalMode.ASK_ALL)
    _, mode = resolve([], auto_safe=False)
    check("auto_approve_safe=false -> ASK_ALL", mode is ApprovalMode.ASK_ALL)


def _block(**kw):
    return types.SimpleNamespace(**kw)


class _FakeFinal:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeClient:
    """Always proposes a failing command until tools are disabled."""

    def stream(self, messages, system_blocks, disable_tools, on_text):
        if disable_tools:
            return _FakeFinal("end_turn", [_block(type="text", text="(summary)")])
        return _FakeFinal("tool_use", [
            _block(type=BLOCK_TOOL_USE, id="t", name="run_command",
                   input={"command": "false", "explanation": "fail"})
        ])


def test_bounded_retry():
    section("bounded self-repair loop")
    console = Console(Palette(enabled=False))
    config = Config(max_fix_attempts=3)
    agent = Agent(_FakeClient(), Shell(cwd="/tmp"), console, config,
                  ApprovalMode.YOLO)
    agent.run_turn("do an impossible thing")
    check("stopped at the failure bound", agent.consecutive_failures == 3)
    check("force-stop engaged", agent._force_stop is True)
    last = agent.conversation.messages[-1]
    check("ended with assistant summary", last["role"] == "assistant")


def test_conversation():
    section("Conversation message shaping")
    conv = Conversation()
    conv.add_user_text("hi")
    conv.add_assistant([_block(type="text", text="ok")])
    conv.add_user_blocks([{"type": "tool_result", "tool_use_id": "x", "content": "y"}])
    roles = [m["role"] for m in conv.messages]
    check("roles in order", roles == ["user", "assistant", "user"])


def main():
    for test in (test_risk, test_interactive, test_shell, test_config,
                 test_decisions, test_cli_resolution, test_bounded_retry,
                 test_conversation):
        test()
    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
