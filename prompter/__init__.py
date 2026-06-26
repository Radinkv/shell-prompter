"""prompter — a natural-language shell agent powered by Claude.

You describe what you want in plain English; prompter figures out the shell
commands, shows them to you with a risk rating, and (after your approval for
anything non-trivial) runs them — keeping track of the working directory as it
goes, just like a real shell session.

    prompter "make a folder called scratch, cd into it, then run claude"
    prompter "download uv if it isn't installed, then check the version"
    prompter            # no prompt → interactive REPL

The agent's workspace is your shell, not a single repo. Claude is just one of
the tools it knows how to launch.
"""

from __future__ import annotations

from .cli import main

__all__ = ["main"]
__version__ = "0.1.0"
