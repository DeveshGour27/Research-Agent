"""Provider-neutral contracts for language-model chat."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence


ChatRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One text message in a conversation supplied to a chat provider."""

    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A normalized text response returned from a chat provider."""

    content: str
    model: str


class LLMProvider(ABC):
    """Abstract interface implemented by each supported LLM provider."""

    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        """Generate one assistant response for an ordered conversation."""
