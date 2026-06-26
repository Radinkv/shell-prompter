# prompter

prompter turns plain English into real shell commands. You describe what you
want. It figures out the commands, shows each one with a risk rating, asks
before anything risky, and runs them. It tracks the working directory as it
goes, like a real shell session.

It runs on the model you choose: Claude, OpenAI (or any OpenAI-compatible
endpoint such as Groq or OpenRouter), or Gemini.

```
$ prompter "make a folder called scratch, cd into it, then run claude"
$ prompter "download uv if it isn't installed, then print its version"
$ prompter
```

Run `prompter` with no goal to start an interactive REPL.

A coding agent lives inside one repository. prompter works on your whole shell.
Launching a coding agent, installing tools, cloning repos, converting files: it
is all just commands. A coding agent like Claude Code becomes one of the tools
prompter can launch, not the thing you live inside.

## Install

The distribution is `shell-prompter` and it installs one command, `prompter`. A
single install includes all three providers (Anthropic, OpenAI, and Gemini).

Install it globally with [pipx](https://pipx.pypa.io):

```bash
pipx install "git+https://github.com/Radinkv/shell-prompter.git"
```

To work on the code instead, use an editable install in a virtual environment:

```bash
cd shell-prompter
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Then add an API key for your provider:

```bash
prompter keys add anthropic sk-ant-...
```

Or export the matching environment variable (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, or `GEMINI_API_KEY`). See [API keys](#api-keys).

## Config

On first run prompter writes `~/.prompter/config.json`:

```json
{
  "default_workspace": "~/Code",
  "provider": "anthropic",
  "model": "",
  "base_url": null,
  "api_key_env": null,
  "max_fix_attempts": 3,
  "auto_approve_safe": true,
  "preferences": [
    "When compiling C++, prefer clang++ with -std=c++17, and fall back to g++ if clang++ isn't available."
  ]
}
```

| Key | What it does |
|-----|--------------|
| `default_workspace` | Where new projects go when you don't say where. `prompter "make a project called hunchday"` creates `~/Code/hunchday`, not a folder in the current directory. |
| `provider` | `anthropic`, `openai`, or `gemini`. See [Providers](#providers). |
| `model` | Empty means use the provider's default (listed in the Providers table). Set it to pin a model. |
| `base_url` | Points the OpenAI adapter at a compatible endpoint such as Groq or OpenRouter. Other providers ignore it. |
| `api_key_env` | The environment variable that holds the API key. Defaults to `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`. |
| `max_fix_attempts` | How many commands may fail in a row before prompter stops. See [Self-repair](#self-repair). |
| `auto_approve_safe` | Set to `false` to confirm every command, including safe ones. |
| `preferences` | Free-form lines passed straight to the model, such as `"Use pnpm, not npm."` |

Run `prompter config` to print the path, or `prompter status` to see the
current provider, model, workspace, and which providers have a key. Any value
can be overridden for a single run with a flag (see [Flags](#flags)).

## Providers

The model backend sits behind one small interface, so prompter behaves the same
whichever you pick. Set your default with `prompter use <provider> [model]`, or
override it for one run with `--provider`. Provider names are case-insensitive,
and `claude`, `gpt`, and `google` work as aliases.

| Provider | Default model | API key | Notes |
|----------|---------------|---------|-------|
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | Default. Also reads an `ant auth login` profile. |
| `openai` | `gpt-5.4` | `OPENAI_API_KEY` | Set `base_url` for Groq or OpenRouter (both speak the OpenAI API). Drop to `gpt-5.4-mini` to go cheaper. |
| `gemini` | `gemini-3.5-flash` | `GEMINI_API_KEY` | Generous free tier, good for everyday use. |

A note on billing. An Anthropic Pro or Max subscription and an API key are
separate accounts. A program cannot bill against your web subscription. For a
free tier, use Gemini, or Groq and OpenRouter through the OpenAI adapter. Switch
to Anthropic when you want it.

## API keys

prompter needs an API key for the active provider. It looks in two places, in
order:

1. The environment variable for the provider (`ANTHROPIC_API_KEY`,
   `OPENAI_API_KEY`, or `GEMINI_API_KEY`).
2. A key you stored with `prompter keys add`, in `~/.prompter/keys.json`.

The environment variable always wins, so CI and one-off overrides keep working.

No command ever prompts you. If a key is missing when you run, prompter prints
the exact command to fix it and exits, rather than dropping into a hidden prompt:

```text
✗ No API key for gemini
    add it:     prompter keys add gemini <key>
    or export:  GEMINI_API_KEY=<key>
```

Manage stored keys by command:

```bash
prompter keys add anthropic sk-ant-...
prompter keys list
prompter keys remove anthropic
```

Because the key is an argument, it lands in your shell history. To avoid that,
export the environment variable instead. Stored keys live in
`~/.prompter/keys.json` with file mode 0600, readable only by you. It is plain
text, the same approach as `~/.aws/credentials`.

## Risk tiers and confirmation

Every command prompter proposes is graded into one of three tiers. The tier
decides whether it runs on its own or asks first.

| Tier | Examples | Default |
|------|----------|---------|
| 🟢 SAFE | `ls`, `cd`, `mkdir`, `git status`, `cat` | runs automatically |
| 🟡 CONFIRM | `brew install`, `pip install`, `git clone`, `mv`, `rm`, `curl` | asks first |
| 🔴 DANGER | `rm -rf`, `sudo`, `curl ... \| sh`, force-push, `dd` | asks first, shown in red |

When prompter asks, you answer:

- `y` or Enter: run it
- `n`: skip it (the model is told and adapts)
- `a`: run it and auto-approve the rest of this run
- `q`: quit

`curl ... | sh` is always DANGER, because it runs code you have not seen.
`--yolo` removes the gate entirely. Use it only when you trust the task.

## Self-repair

prompter runs a command, reads the actual error, and decides the next step. If
`clang++` is missing it retries with `g++` on its own. There are no scripted
repair rules.

`max_fix_attempts` (default 3) stops it from looping on a stuck step. It counts
commands that fail in a row. At the limit, prompter tells the model to stop and
explain what went wrong. A command you decline does not count. Only commands
that ran and failed do.

## How it works

1. Your request, plus your OS, shell, current directory, default workspace, and
   preferences, go to the model.
2. The model works toward the goal by calling one tool, `run_command`, a single
   command at a time, reacting to each result.
3. prompter grades each command (see [Risk tiers](#risk-tiers-and-confirmation))
   and runs or gates it.
4. The working directory persists across commands, so "make a folder, cd in,
   then run claude" lands in the right place.

prompter cannot change the directory of the shell you launched it from. No
program can. It runs every command, and launches your coding agent, in the right
place, which covers these workflows.

Programs that take over the terminal (`claude`, `vim`, `ssh`, a REPL, `top`) get
the real terminal so you can interact with them. The model marks these
automatically.

## Commands

Management is done with subcommands. Flags only modify a single run. No command
opens a prompt.

| Command | Effect |
|---------|--------|
| `prompter "<goal>"` | Run a one-off goal. |
| `prompter` | Interactive chat (REPL). |
| `prompter keys add <provider> <key>` | Store an API key. |
| `prompter keys list` | Show which providers have a key (masked). |
| `prompter keys remove <provider>` | Delete a stored key. |
| `prompter use <provider> [model]` | Set your default provider and model. |
| `prompter status` | Show the current provider, model, workspace, and keys. |
| `prompter config` | Print the config file path. |
| `prompter help` | Show usage. |

## Flags

Flags modify a single run and sit alongside the goal (`prompter --yolo "..."`).

| Flag | Effect |
|------|--------|
| `--provider NAME` | Use anthropic, openai, or gemini for this run. |
| `--model ID` | Override the model. |
| `--base-url URL` | OpenAI-compatible endpoint (Groq, OpenRouter). |
| `--workspace PATH` | Override the default workspace. |
| `--max-fix N` | Override `max_fix_attempts`. |
| `--ask-all` | Confirm every command, including safe ones. |
| `--yolo` | Run everything with no confirmation. Dangerous. |

## Project layout

```
prompter/
  constants.py   shared primitive constants (empty string, separators)
  colors.py      Palette (ANSI, off when output is not a TTY)
  config.py      Config dataclass, ApprovalMode, load and save
  risk.py        RiskTier and classify(): the safe/confirm/danger rules
  shell.py       Shell and CommandResult: execution and cwd tracking
  prompts.py     system prompt and per-turn environment context
  providers/     pluggable model backends
    base.py        neutral types, ModelProvider ABC, registry
    anthropic_provider.py, openai_provider.py, gemini_provider.py
  ui.py          Console and Decision: all printing and input
  agent.py       Agent and Conversation: the orchestration loop
  cli.py         command dispatch, run setup, actionable errors, main()
tests/
  conftest.py, _helpers.py   fixtures and fakes (FakeProvider, FakeShell)
  test_*.py                  one module per package module
```

The design follows a state boundary. Stateful pieces (`Shell`, `Agent`, the
provider, `Console`) are objects injected into each other. Pure transforms
(`classify`, `truncate`, config loading, context building) are plain functions.
`Agent` orchestrates its collaborators and does no I/O or API calls itself, so it
can be tested with a fake provider and console.

Providers share a template-method base. `ModelProvider.complete()` owns the
fixed algorithm: build a request, stream it into a `TurnCollector`, wrap provider
errors, return a turn. Each adapter supplies only `build_request` and
`run_stream`. Adding a backend is one file in `providers/` plus one `@register`
line, with no change to the agent.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

About 165 unit tests, no network or API key needed. The agent loop is tested
with fakes: a scripted provider, a recording shell, a mock console. Each
adapter's translation to and from its wire format is tested with a fake SDK
client.
