"""Tests for the WebSearchTool."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.exceptions import ToolExecutionError
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure settings are predictably configured for testing."""
    monkeypatch.setattr(settings, "web_search_provider", "searxng")
    monkeypatch.setattr(settings, "searxng_base_url", "http://localhost:8080")
    monkeypatch.setattr(settings, "web_search_max_results", 2)
    monkeypatch.setattr(settings, "web_search_timeout_seconds", 5.0)


@pytest.fixture
def tool() -> WebSearchTool:
    return WebSearchTool()


def test_web_search_success(tool: WebSearchTool, mock_settings: None) -> None:
    """Test successful normalization of provider response."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "results": [
            {"title": "Title 1", "url": "http://url1.com", "content": "Snippet 1"},
            {"title": "Title 2", "url": "http://url2.com", "content": "Snippet 2"},
            {"title": "Title 3", "url": "http://url3.com", "content": "Snippet 3"},
        ]
    }).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        result_str = tool.execute(query="test query")

    # Assert urlopen called with correct parameters
    request_obj = mock_urlopen.call_args[0][0]
    
    import urllib.parse
    params = urllib.parse.urlencode({"q": "test query", "format": "json"})
    assert request_obj.full_url == f"http://localhost:8080/search?{params}"
    assert request_obj.method == "GET"
    assert request_obj.headers.get("User-agent") == "ResearchAgent/1.0"

    # Assert normalized result
    results = json.loads(result_str)
    assert len(results) == 2  # Max results applied
    assert results[0] == {"title": "Title 1", "url": "http://url1.com", "snippet": "Snippet 1"}
    assert results[1] == {"title": "Title 2", "url": "http://url2.com", "snippet": "Snippet 2"}


def test_web_search_empty_query(tool: WebSearchTool, mock_settings: None) -> None:
    """Test validation of empty queries."""
    with pytest.raises(ToolExecutionError, match="must not be empty"):
        tool.execute(query="")


def test_web_search_whitespace_query(tool: WebSearchTool, mock_settings: None) -> None:
    """Test validation of whitespace queries."""
    with pytest.raises(ToolExecutionError, match="must not be empty"):
        tool.execute(query="   \n \t  ")


def test_web_search_unsupported_provider(tool: WebSearchTool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error on unsupported provider."""
    monkeypatch.setattr(settings, "web_search_provider", "unsupported")
    with pytest.raises(ToolExecutionError, match="Unsupported web search provider"):
        tool.execute(query="test")

def test_web_search_provider_resolution() -> None:
    """Test that WebSearchTool resolves to searxng without falling back to Tavily."""
    tool = WebSearchTool()
    
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"results": []}).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        tool.execute(query="test query")
        
    request_obj = mock_urlopen.call_args[0][0]
    assert "localhost" in request_obj.full_url
    assert "searxng" not in getattr(request_obj, "full_url", "")
    assert "tavily" not in request_obj.full_url

def test_web_search_http_error(tool: WebSearchTool, mock_settings: None) -> None:
    """Test handling of HTTP error from the provider."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:8080/search",
            code=403,
            msg="Forbidden",
            hdrs=None, # type: ignore
            fp=None,
        )
        with pytest.raises(ToolExecutionError, match="HTTP 403: Forbidden"):
            tool.execute(query="test")


def test_web_search_url_error(tool: WebSearchTool, mock_settings: None) -> None:
    """Test handling of network-level connection failures."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(ToolExecutionError, match="Failed to connect"):
            tool.execute(query="test")


def test_web_search_timeout_error(tool: WebSearchTool, mock_settings: None) -> None:
    """Test handling of provider timeout."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = TimeoutError()
        with pytest.raises(ToolExecutionError, match="timed out"):
            tool.execute(query="test")


def test_web_search_malformed_json_response(tool: WebSearchTool, mock_settings: None) -> None:
    """Test handling of malformed JSON from provider."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"<html>Not JSON</html>"

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        with pytest.raises(ToolExecutionError, match="malformed JSON"):
            tool.execute(query="test")


def test_web_search_invalid_results_format(tool: WebSearchTool, mock_settings: None) -> None:
    """Test handling of JSON response with unexpected schema."""
    mock_response = MagicMock()
    # Provide a dict for results instead of a list
    mock_response.read.return_value = json.dumps({"results": {"bad": "format"}}).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        with pytest.raises(ToolExecutionError, match="invalid results format"):
            tool.execute(query="test")


def test_web_search_zero_results(tool: WebSearchTool, mock_settings: None) -> None:
    """Test provider returning zero results."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"results": []}).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        result = tool.execute(query="test")
        assert result == "No results found."


def test_registry_integration() -> None:
    """Test that WebSearchTool integrates seamlessly with ToolRegistry."""
    registry = ToolRegistry()
    tool = WebSearchTool()
    registry.register(tool)
    
    retrieved = registry.get("web_search")
    assert retrieved is tool
    
    schemas = registry.schemas()
    assert any(s["function"]["name"] == "web_search" for s in schemas)


def test_tool_metadata(tool: WebSearchTool) -> None:
    """Test that tool exposes correct schema metadata."""
    schema = tool.to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["required"]
