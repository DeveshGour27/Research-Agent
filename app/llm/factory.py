from __future__ import annotations
import json
from app.config import Settings
from app.exceptions import ConfigurationError
from app.llm.models import ModelCapability, ModelProfile, TaskType
from app.llm.provider import ModelProvider
from app.llm.gateway import ModelGateway, ModelRouter

def create_model_gateway(configuration: Settings) -> ModelGateway:
    """Create the ModelGateway configured with providers and profiles."""
    providers: dict[str, ModelProvider] = {}
    
    # Initialize Groq provider
    if configuration.groq_api_key:
        from app.llm.groq_provider import GroqProvider
        providers["groq"] = GroqProvider(configuration)

    # Initialize OpenAI provider
    if getattr(configuration, "openai_api_key", None):
        from app.llm.openai_provider import OpenAIProvider
        providers["openai"] = OpenAIProvider(configuration)
        
    profiles: list[ModelProfile] = []
    if getattr(configuration, "llm_gateway_profiles_json", None):
        try:
            raw_profiles = json.loads(configuration.llm_gateway_profiles_json)
            for rp in raw_profiles:
                caps = {ModelCapability[c] for c in rp.get("capabilities", [])}
                profiles.append(ModelProfile(
                    provider=rp["provider"],
                    model_id=rp["model_id"],
                    capabilities=caps,
                    priority=rp.get("priority", 100),
                    context_limit=rp.get("context_limit", 8192),
                ))
        except Exception as e:
            raise ConfigurationError("Failed to parse llm_gateway_profiles_json", details={"error": str(e)}) from e
    else:
        # Fallback to legacy configuration
        caps = {ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.STRUCTURED_OUTPUT, ModelCapability.LONG_CONTEXT}
        profiles.append(ModelProfile(
            provider=configuration.llm_provider,
            model_id=configuration.llm_model,
            capabilities=caps,
            priority=100
        ))
        profiles.append(ModelProfile(
            provider=configuration.llm_provider,
            model_id=configuration.llm_fallback_model,
            capabilities=caps,
            priority=50
        ))

    router = ModelRouter(profiles=profiles)
    return ModelGateway(router=router, providers=providers, max_retries=configuration.max_retries)
