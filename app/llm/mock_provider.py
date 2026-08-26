from __future__ import annotations

from app.llm.models import ModelRequest, ModelResponse, ToolCall
from app.llm.provider import ModelProvider
from app.exceptions import LLMAPIError

class MockProvider(ModelProvider):
    @property
    def provider_id(self) -> str:
        return "mock"
        
    def __init__(self):
        self.responses: list[ModelResponse | Exception] = []
        
    def add_response(self, response: ModelResponse | Exception):
        self.responses.append(response)

    def generate(self, request: ModelRequest, model_id: str) -> ModelResponse:
        if not self.responses:
            return ModelResponse(
                content="Mock default response",
                model=model_id,
                provider="mock",
                input_tokens=10,
                output_tokens=10,
                total_tokens=20
            )
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        # Update model in response to match request
        resp.model = model_id
        resp.provider = "mock"
        return resp
