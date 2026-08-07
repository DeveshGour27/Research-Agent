"""Provider-agnostic LLM router used by the agent loop.

The :class:`LLMRouter` is the single contact point through which the
agent sends messages and receives structured responses.  It wraps any
:class:`~app.llm.base.LLMProvider` and exposes a clean, stable interface
so the loop does not depend on provider internals and stays trivially
testable.
"""

from __future__ import annotations

from typing import Any

from app.llm.base import LLMProvider, LLMResponse
from app.logger import get_logger

logger = get_logger(__name__)


class LLMRouter:
    """Route agent requests to the configured :class:`LLMProvider`.

    Args:
        provider: An LLM provider that implements
                  :meth:`~app.llm.base.LLMProvider.generate_with_tools`.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send *messages* to the provider and return a structured response.

        Args:
            messages: Full conversation in OpenAI-format dicts.
            tools:    Optional list of OpenAI-compatible tool schemas.  When
                      ``None`` or empty, the provider receives no tool context.

        Returns:
            :class:`~app.llm.base.LLMResponse` — either a text answer or a
            tool-call request.
        """
        resolved_tools = tools or []
        logger.debug(
            "LLMRouter dispatching request",
            extra={"message_count": len(messages), "tool_count": len(resolved_tools)},
        )
        return self._provider.generate_with_tools(messages, resolved_tools)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Backward-compatible alias for :meth:`generate`."""
        return self.generate(messages, tools)
