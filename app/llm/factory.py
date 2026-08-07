"""Factory for selecting an LLM provider from centralized configuration."""

from __future__ import annotations

from app.config import Settings
from app.exceptions import ConfigurationError
from app.llm.base import LLMProvider


def create_chat_provider(configuration: Settings) -> LLMProvider:
    """Create the provider identified by ``configuration.llm_provider``."""
    provider_name = configuration.llm_provider.casefold().strip()
    if provider_name == "groq":
        from app.llm.groq_provider import GroqProvider
        return GroqProvider(configuration)
    raise ConfigurationError(
        f"Unsupported LLM provider: {configuration.llm_provider}.",
        details={"provider": configuration.llm_provider},
    )
