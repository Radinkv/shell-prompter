# prompter

A natural-language shell agent powered by Claude. You describe what you want in
plain English; `prompter` figures out the shell commands, shows each one with a
risk rating, asks before anything non-trivial, and runs them — tracking the
working directory as it goes, just like a real shell session.

```
$ prompter "make a folder called scratch, cd into it, then run claude"
$ prompter "download uv if it isn't installed, then print its version"
$ prompter            # no prompt → interactive REPL
```

Unlike a coding agent (which lives inside one repo), prompter's workspace is
your **whole shell**. Launching Claude Code, installing tools, cloning repos,
converting files — it's all just commands it knows how to run. Claude Code
becomes one of the tools prompter can invoke, not the thing you're inside of.

## Install

Requires Python 3.9+ and an Anthropic API key (or a saved `ant auth login`).

```bash
cd shell-prompter
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...      # or run `ant auth login`
```

Now `prompter` is on your PATH (inside the venv). To use it from anywhere
without activating the venv, install with [pipx](https://pipx.pypa.io) instead:

```bash
pipx install -e /path/to/shell-prompter
```

## Config & defaults

On first run, prompter writes `~/.prompter/config.json`:

```json
{
  "default_workspace": "~/Code",
  "model": "claude-sonnet-4-6",
  "max_fix_attempts": 3,
  "auto_approve_safe": true,
  "preferences": [
    "When compiling C++, prefer clang++ with -std=c++17, and fall back to g++ if clang++ isn't available."
  ]
}
```

- **`default_workspace`** — where new projects go when you don't say where. So
  `prompter "make a project called hunchday and run claude"` creates
  `~/Code/hunchday`, not a folder in whatever directory you happened to be in.
- **`model`** — defaults to `claude-sonnet-4-6`. This agent does mostly simple,
  latency-sensitive shell planning, which Sonnet handles well and cheaply. Bump
  to `claude-opus-4-8` for genuinely hard tasks (big multi-step setups, tricky
  debugging) — set it here, or per-run with `--model`.
- **`preferences`** — free-form lines handed straight to the model. Add your own
  (`"Use pnpm, not npm."`, `"Default Python to a .venv."`) and they're respected.
- **`max_fix_attempts`** — see the retry section below.
- **`auto_approve_safe`** — set to `false` to make prompter confirm *everything*.

Run `prompter --config` to print the path. Override per-run with `--workspace`,
`--max-fix`, and `--model`.

## Self-repair, bounded

prompter runs a command, sees the **actual error**, and decides the next step —
so "compile with clang++" failing because clang++ isn't installed leads to a
retry with `g++` on its own, no scripted repair rules needed.

To stop it churning forever on a wedged step, `max_fix_attempts` (default 3)
caps how many commands may fail **in a row**. Hit the cap and prompter tells
Claude to stop, summarize what went wrong, and hand it back to you. A command
*you* decline doesn't count against the limit — only commands that actually ran
and failed.

## How it works

1. Your request + your OS, shell, current directory, default workspace, and
   preferences go to Claude.
2. Claude works toward the goal by calling a `run_command` tool, one command
   at a time, reacting to each result before deciding the next step.
3. Every proposed command is graded into a **risk tier**:

   | Tier        | Examples                                        | Default behavior |
   |-------------|-------------------------------------------------|------------------|
   | 🟢 `SAFE`    | `ls`, `cd`, `mkdir`, `git status`, `cat`        | runs automatically |
   | 🟡 `CONFIRM` | `brew install`, `pip install`, `git clone`, `mv`, `rm`, `curl` | asks first |
   | 🔴 `DANGER`  | `rm -rf`, `sudo`, `curl … \| sh`, force-push, `dd` | asks first, in red |

4. `cd` actually sticks: prompter tracks the working directory across commands,
   so "make a folder, cd in, then run claude" lands you in the right place.

> **Note on directories:** like any tool of this kind, prompter can't change the
> directory of the *outer* shell you launched it from — that's an OS limitation.
> But it runs every subsequent command (and launches Claude) in the right place,
> which covers the workflows above.

## Confirmation prompt

When prompter asks, you can answer:

- `y` / Enter — run this command
- `n` — skip it (Claude is told and adapts)
- `a` — run this and auto-approve the rest of this run
- `q` — quit

## Flags

| Flag            | Effect                                                    |
|-----------------|-----------------------------------------------------------|
| `--workspace P` | Override `default_workspace` for this run.                |
| `--max-fix N`   | Override `max_fix_attempts` for this run.                 |
| `--ask-all`     | Confirm **every** command, including safe ones.           |
| `--yolo`        | Run everything with no confirmation. Dangerous — use sparingly. |
| `--model ID`    | Override the model (config `model`, else `claude-sonnet-4-6`). |
| `--config`      | Print the config file path and exit.                      |

## Interactive programs

For things that take over the terminal — `claude`, `vim`, `ssh`, a Python REPL,
`top` — prompter hands them the real terminal so you can interact, instead of
capturing their output. Claude marks these calls automatically.

## Project layout

```
prompter/
  constants.py   protocol strings + magic numbers, all named
  colors.py      Palette (ANSI, auto-disabled off-TTY)
  config.py      Config dataclass, ApprovalMode, load/save
  risk.py        RiskTier, classify() — the safe/confirm/danger rules
  shell.py       Shell + CommandResult — execution & cwd tracking
  llm.py         ClaudeClient, run_command schema, system prompt
  ui.py          Console + Decision — every print/input lives here
  agent.py       Agent, Conversation — the orchestration loop only
  cli.py         argument parsing, wiring, main()
tests/
  test_offline.py   no-network checks (run: python tests/test_offline.py)
```

The split follows a state boundary: stateful pieces (`Shell`, `Agent`,
`ClaudeClient`, `Console`) are objects that get injected into each other; pure
transforms (`classify`, `truncate`, config loading, context building) are plain
functions. `Agent` orchestrates its collaborators and does no I/O or API calls
itself, which is what makes it testable with a fake client + console.

## Safety notes

- Commands come from a language model. The risk tiers and confirmation gate are
  there so you stay in control; read what you're approving, especially 🔴 items.
- `--yolo` removes that gate entirely. Only use it when you fully trust the task.
- "Download and run this from the internet" (`curl … | sh`) is always flagged as
  `DANGER`, because it executes code you haven't seen.
