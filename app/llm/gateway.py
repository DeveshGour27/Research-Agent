from __future__ import annotations

import dataclasses
from typing import Any
from app.llm.models import ModelCapability, ModelProfile, ModelRequest, ModelResponse, TaskType
from app.llm.provider import ModelProvider
from app.exceptions import LLMAPIError, ConfigurationError
from app.logger import get_logger
import time

logger = get_logger(__name__)

class ModelRouter:
    def __init__(self, profiles: list[ModelProfile], policies: dict[TaskType, set[ModelCapability]] | None = None):
        self.profiles = [p for p in profiles if p.enabled]
        # Default policies
        self.policies = policies or {
            TaskType.PLANNING: {ModelCapability.REASONING, ModelCapability.TEXT_GENERATION},
            TaskType.TOOL_CALLING: {ModelCapability.TOOL_CALLING},
            TaskType.RAG_SYNTHESIS: {ModelCapability.LONG_CONTEXT, ModelCapability.TEXT_GENERATION},
            TaskType.REFLECTION: {ModelCapability.REASONING},
            TaskType.GENERAL: {ModelCapability.TEXT_GENERATION},
        }

    def select_profiles(self, request: ModelRequest) -> list[ModelProfile]:
        """Return a list of eligible profiles ordered by priority."""
        req_caps = set(request.required_capabilities)
        if request.task_type in self.policies:
            req_caps.update(self.policies[request.task_type])
        
        # If user explicitly requested structured output format
        if request.response_format is not None:
            req_caps.add(ModelCapability.STRUCTURED_OUTPUT)
        # If user explicitly provided tools
        if request.tools:
            req_caps.add(ModelCapability.TOOL_CALLING)

        eligible = []
        for p in self.profiles:
            if req_caps.issubset(p.capabilities):
                eligible.append(p)
                
        # Sort descending by priority
        eligible.sort(key=lambda p: p.priority, reverse=True)
        return eligible

class ModelGateway:
    def __init__(self, router: ModelRouter, providers: dict[str, ModelProvider], max_retries: int = 3):
        self.router = router
        self.providers = providers
        self.max_retries = max_retries

    def generate(self, request: ModelRequest) -> ModelResponse:
        eligible_profiles = self.router.select_profiles(request)
        if not eligible_profiles:
            raise ConfigurationError("No eligible models found for the requested capabilities.", details={"capabilities": [c.name for c in request.required_capabilities]})

        last_error = None
        failed_providers: set[str] = set()
        for profile in eligible_profiles:
            if profile.provider in failed_providers:
                logger.debug("Skipping profile due to permanent provider failure.", extra={"provider": profile.provider, "model": profile.model_id})
                continue

            provider = self.providers.get(profile.provider)
            if not provider:
                logger.warning(f"Provider {profile.provider} not registered.")
                continue

            for attempt in range(self.max_retries):
                try:
                    logger.debug("Dispatching request", extra={"provider": profile.provider, "model": profile.model_id, "attempt": attempt + 1})
                    start_time = time.perf_counter()
                    
                    req = dataclasses.replace(request)
                    if req.temperature is None:
                        req.temperature = profile.default_temperature
                    if req.max_tokens is None:
                        req.max_tokens = profile.max_output_tokens
                        
                    response = provider.generate(req, profile.model_id)
                    latency = time.perf_counter() - start_time
                    
                    logger.info("Model request completed", extra={
                        "provider": response.provider,
                        "model": response.model,
                        "latency_s": round(latency, 2),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "total_tokens": response.total_tokens,
                        "task_type": request.task_type.name,
                    })
                    return response

                except LLMAPIError as e:
                    last_error = e
                    transient = e.details.get("transient", False)
                    logger.warning("Model request failed", extra={"provider": profile.provider, "model": profile.model_id, "transient": transient, "error": str(e)})
                    if not transient:
                        if getattr(e, "status_code", None) in (401, 402, 403):
                            failed_providers.add(profile.provider)
                        # Permanent error on this provider/model, fallback to next eligible profile
                        break
                    # Transient error, retry same profile
                    time.sleep(1.0 * (attempt + 1))
                except Exception as e:
                    last_error = e
                    logger.exception("Unexpected error in model provider", extra={"provider": profile.provider, "model": profile.model_id})
                    break # Fallback to next profile

        raise LLMAPIError("All eligible model providers failed.", details={"last_error": str(last_error)})
