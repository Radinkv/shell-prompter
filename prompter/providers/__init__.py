"""Model providers.

Importing this package registers every built-in adapter (Anthropic, OpenAI,
Gemini) in the registry. Use create_provider(config) to get one by name.

The adapter modules are imported for their registration side effects and kept
in BUILTIN_ADAPTERS so the names are referenced rather than appearing unused.
"""

from __future__ import annotations

from . import anthropic_provider, gemini_provider, openai_provider
from .base import (
    AssistantMessage,
    AssistantTurn,
    HistoryItem,
    ModelProvider,
    ProviderAuthError,
    ProviderError,
    ProviderNotInstalled,
    ToolInvocation,
    ToolResult,
    ToolResultsMessage,
    UserMessage,
    create_provider,
    known_providers,
    unknown_provider_message,
)

BUILTIN_ADAPTERS = (anthropic_provider, gemini_provider, openai_provider)

__all__ = [
    "AssistantMessage",
    "AssistantTurn",
    "HistoryItem",
    "ModelProvider",
    "ProviderAuthError",
    "ProviderError",
    "ProviderNotInstalled",
    "ToolInvocation",
    "ToolResult",
    "ToolResultsMessage",
    "UserMessage",
    "create_provider",
    "known_providers",
    "unknown_provider_message",
]
