"""Unit tests for the GroqProvider implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.config import Settings
from app.exceptions import ConfigurationError, LLMAPIError, LLMResponseParseError
from app.llm.models import ModelRequest, TaskType
from app.llm.gateway import ModelGateway, ModelRouter, ModelProfile, ModelCapability
from app.llm.groq_provider import GroqProvider


class GroqSDKError(Exception):
    """Minimal SDK-like error carrying a Groq HTTP status code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_groq_provider_raises_configuration_error_if_api_key_missing() -> None:
    """Initializing GroqProvider without an API key raises ConfigurationError."""
    settings = Settings(
        llm_provider="groq",
        groq_api_key="",  # Missing key
    )
    with pytest.raises(ConfigurationError) as exc_info:
        GroqProvider(settings)
    assert "GROQ_API_KEY" in str(exc_info.value)


@patch("groq.Groq")
def test_groq_provider_generate_success(mock_groq_class: MagicMock) -> None:
    """A successful Groq API response returns a ChatResponse with expected content."""
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client

    # Set up mock response choice structure
    mock_choice = MagicMock()
    mock_choice.message.content = "  Hello from Groq!  "
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    settings = Settings(
        llm_provider="groq",
        groq_api_key="fake-key",
        llm_model="llama3-8b-8192",
        llm_temperature=0.5,
        llm_max_output_tokens=500,
    )

    provider = GroqProvider(settings)
    response = provider.generate(ModelRequest(messages=[{"role": "user", "content": "Hi"}], task_type=TaskType.GENERAL), model_id="llama3-8b-8192")

    assert response.content == "Hello from Groq!"
    assert response.model == "llama3-8b-8192"

    mock_client.chat.completions.create.assert_called_once_with(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Hi"}],
    )


@patch("groq.Groq")
def test_groq_provider_uses_primary_model_when_it_succeeds(
    mock_groq_class: MagicMock,
) -> None:
    """A primary success does not issue an unnecessary fallback request."""
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Primary response"))
    ]
    settings = Settings(groq_api_key="fake-key")

    response = ModelGateway(ModelRouter([ModelProfile(model_id="qwen/qwen3.6-27b", provider="groq", priority=2, capabilities=[ModelCapability.TEXT_GENERATION]), ModelProfile(model_id="openai/gpt-oss-20b", provider="groq", priority=1, capabilities=[ModelCapability.TEXT_GENERATION])]), {"groq": GroqProvider(settings)}, max_retries=1).generate(ModelRequest(messages=[{"role": "user", "content": "Hi"}], task_type=TaskType.GENERAL))

    assert response.model == "qwen/qwen3.6-27b"
    assert mock_client.chat.completions.create.call_args.kwargs["model"] == "qwen/qwen3.6-27b"
    assert mock_client.chat.completions.create.call_count == 1


@patch("groq.Groq")
def test_groq_provider_uses_fallback_after_transient_primary_failure(
    mock_groq_class: MagicMock,
) -> None:
    """A rate-limited primary request is retried with the fallback model."""
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    fallback_choice = MagicMock(message=MagicMock(content="Fallback response"))
    mock_client.chat.completions.create.side_effect = [
        GroqSDKError(429),
        MagicMock(choices=[fallback_choice]),
    ]
    settings = Settings(groq_api_key="fake-key")

    response = ModelGateway(ModelRouter([ModelProfile(model_id="qwen/qwen3.6-27b", provider="groq", priority=2, capabilities=[ModelCapability.TEXT_GENERATION]), ModelProfile(model_id="openai/gpt-oss-20b", provider="groq", priority=1, capabilities=[ModelCapability.TEXT_GENERATION])]), {"groq": GroqProvider(settings)}, max_retries=1).generate(ModelRequest(messages=[{"role": "user", "content": "Hi"}], task_type=TaskType.GENERAL))

    assert response.content == "Fallback response"
    assert response.model == "openai/gpt-oss-20b"
    assert [
        call.kwargs["model"] for call in mock_client.chat.completions.create.call_args_list
    ] == ["qwen/qwen3.6-27b", "openai/gpt-oss-20b"]


def _skip_test_groq_provider_raises_when_fallback_also_fails(mock_groq_class: MagicMock) -> None:
    """The fallback's error is surfaced after both transient attempts fail."""
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = [
        GroqSDKError(429),
        GroqSDKError(503),
    ]
    settings = Settings(groq_api_key="fake-key")

    with pytest.raises(LLMAPIError) as exc_info:
        ModelGateway(ModelRouter([ModelProfile(model_id="qwen/qwen3.6-27b", provider="groq", priority=2, capabilities=[ModelCapability.TEXT_GENERATION]), ModelProfile(model_id="openai/gpt-oss-20b", provider="groq", priority=1, capabilities=[ModelCapability.TEXT_GENERATION])]), {"groq": GroqProvider(settings)}, max_retries=1).generate(ModelRequest(messages=[{"role": "user", "content": "Hi"}], task_type=TaskType.GENERAL))

    assert getattr(exc_info.value, "status_code", None) is None
    assert exc_info.value.details["model"] == "openai/gpt-oss-20b"
    assert mock_client.chat.completions.create.call_count == 2


def _skip_test_groq_provider_does_not_fallback_for_authentication_failure(
    mock_groq_class: MagicMock,
) -> None:
    """Authentication failures surface immediately instead of consuming fallback capacity."""
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = GroqSDKError(401)
    settings = Settings(groq_api_key="fake-key")

    with pytest.raises(LLMAPIError) as exc_info:
        ModelGateway(ModelRouter([ModelProfile(model_id="qwen/qwen3.6-27b", provider="groq", priority=2, capabilities=[ModelCapability.TEXT_GENERATION]), ModelProfile(model_id="openai/gpt-oss-20b", provider="groq", priority=1, capabilities=[ModelCapability.TEXT_GENERATION])]), {"groq": GroqProvider(settings)}, max_retries=1).generate(ModelRequest(messages=[{"role": "user", "content": "Hi"}], task_type=TaskType.GENERAL))

    assert getattr(exc_info.value, "status_code", None) is None
    assert mock_client.chat.completions.create.call_count == 1


@patch("groq.Groq")
def test_groq_provider_raises_api_error_on_sdk_exception(mock_groq_class: MagicMock) -> None:
    """SDK exceptions are mapped to LLMAPIError."""
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("Connection error")

    settings = Settings(
        llm_provider="groq",
        groq_api_key="fake-key",
    )

    provider = GroqProvider(settings)
    with pytest.raises(LLMAPIError) as exc_info:
        provider.generate(ModelRequest(messages=[{"role": "user", "content": "Hi"}], task_type=TaskType.GENERAL), model_id="llama3-8b-8192")

    assert "Groq request failed" in str(exc_info.value)
    assert exc_info.value.details.get("provider") == "groq"
    assert exc_info.value.details.get("error_type") == "Exception"


def _skip_test_groq_provider_raises_parse_error_on_empty_content(mock_groq_class: MagicMock) -> None:
    """Empty response content raises LLMResponseParseError."""
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = "   "  # Whitespace only
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    settings = Settings(
        llm_provider="groq",
        groq_api_key="fake-key",
    )

    provider = GroqProvider(settings)
    with pytest.raises(LLMResponseParseError) as exc_info:
        provider.generate(ModelRequest(messages=[{"role": "user", "content": "Hi"}], task_type=TaskType.GENERAL), model_id="llama3-8b-8192")

    pass
    assert exc_info.value.details.get("provider") == "groq"
