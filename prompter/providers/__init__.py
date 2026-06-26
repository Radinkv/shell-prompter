"""Model providers.

Importing this package registers every built-in adapter (Anthropic, OpenAI,
Gemini) in the registry. Use create_provider(config) to get one by name.
"""

from __future__ import annotations

from . import anthropic_provider, gemini_provider, openai_provider  # noqa: F401
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
