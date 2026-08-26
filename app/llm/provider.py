from abc import ABC, abstractmethod
from typing import Sequence
from app.llm.models import ModelRequest, ModelResponse

class ModelProvider(ABC):
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Return a unique identifier for this provider (e.g., 'groq', 'qwen')."""
        pass

    @abstractmethod
    def generate(self, request: ModelRequest, model_id: str) -> ModelResponse:
        """Generate a response using the given request and specific model_id."""
        pass
