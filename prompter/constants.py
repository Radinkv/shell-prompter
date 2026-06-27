"""Primitive string and number constants shared across the package.

These carry no domain meaning on their own. They exist so that no module spells
out a bare literal such as the empty string or a newline inside a function body.
A literal that does have domain meaning lives as a named constant in the module
that owns it.
"""

from __future__ import annotations

EMPTY = ""
NEWLINE = "\n"
SPACE = " "
PIPE = "|"
COMMA_SPACE = ", "
HOME_MARK = "~"
FILE_WRITE_MODE = "w"
JSON_INDENT = 2
