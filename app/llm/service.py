"""Application service that isolates chat orchestration from provider SDKs."""

from __future__ import annotations

from time import perf_counter
from typing import Sequence

from app.constants import MAX_QUERY_LENGTH
from app.exceptions import AgentError, InputValidationError, LLMAPIError
from app.llm.base import ChatMessage, ChatResponse, LLMProvider
from app.logger import get_logger


logger = get_logger(__name__)


class ChatService:
    """Validate chat requests, delegate generation, and emit safe telemetry."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def chat(
        self, history: Sequence[ChatMessage], user_input: str
    ) -> ChatResponse:
        """Generate a response using history plus one new user message."""
        self._validate_user_input(user_input)
        messages = [*history, ChatMessage(role="user", content=user_input)]
        started_at = perf_counter()
        logger.info(
            "Chat request started",
            extra={"history_messages": len(history), "input_characters": len(user_input)},
        )
        try:
            response = self._provider.generate(messages)
        except AgentError:
            logger.exception(
                "Chat request failed",
                extra={"latency_ms": round((perf_counter() - started_at) * 1_000, 2)},
            )
            raise
        except Exception as error:
            logger.exception(
                "Chat request failed unexpectedly",
                extra={
                    "latency_ms": round((perf_counter() - started_at) * 1_000, 2),
                    "error_type": type(error).__name__,
                },
            )
            raise LLMAPIError(
                "The configured LLM provider failed unexpectedly.",
                details={"error_type": type(error).__name__},
            ) from error

        logger.info(
            "Chat response received",
            extra={
                "model": response.model,
                "response_characters": len(response.content),
                "latency_ms": round((perf_counter() - started_at) * 1_000, 2),
            },
        )
        return response

    @staticmethod
    def _validate_user_input(user_input: str) -> None:
        """Reject empty or oversized requests before calling a provider."""
        if not user_input.strip():
            raise InputValidationError("A chat message cannot be empty.")
        if len(user_input) > MAX_QUERY_LENGTH:
            raise InputValidationError(
                f"A chat message cannot exceed {MAX_QUERY_LENGTH} characters.",
                details={"max_characters": MAX_QUERY_LENGTH},
            )
