"""Web search tool using SearXNG HTTP API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from app.config import settings
from app.exceptions import ToolExecutionError
from app.tools.base import BaseTool


class WebSearchTool(BaseTool):
    """Perform a web search using the configured provider (e.g., SearXNG)."""

    name = "web_search"
    description = (
        "Search the web for up-to-date information on a given topic. "
        "Returns a list of relevant search results including titles, URLs, and snippets."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            }
        },
        "required": ["query"],
    }

    def execute(self, **kwargs: object) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ToolExecutionError(
                "The 'query' argument must not be empty.",
                tool_name=self.name,
            )

        provider = settings.web_search_provider.lower()
        if provider != "searxng":
            raise ToolExecutionError(
                f"Unsupported web search provider: {provider}",
                tool_name=self.name,
            )

        max_results = settings.web_search_max_results
        timeout = settings.web_search_timeout_seconds
        base_url = settings.searxng_base_url.rstrip("/")

        import urllib.parse
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json"
        })
        url = f"{base_url}/search?{params}"
        
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchAgent/1.0"},
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
                result_data = json.loads(response_body)
        except urllib.error.HTTPError as error:
            raise ToolExecutionError(
                f"Web search provider returned HTTP {error.code}: {error.reason}",
                tool_name=self.name,
            )
        except urllib.error.URLError as error:
            raise ToolExecutionError(
                f"Failed to connect to web search provider: {error.reason}",
                tool_name=self.name,
            )
        except TimeoutError:
            raise ToolExecutionError(
                "Web search request timed out.",
                tool_name=self.name,
            )
        except json.JSONDecodeError:
            raise ToolExecutionError(
                "Web search provider returned malformed JSON.",
                tool_name=self.name,
            )
        except Exception as error:
            raise ToolExecutionError(
                f"Unexpected error during web search: {type(error).__name__}",
                tool_name=self.name,
            )

        # Normalize the results
        results = result_data.get("results", [])
        if not isinstance(results, list):
            raise ToolExecutionError(
                "Web search provider returned an invalid results format.",
                tool_name=self.name,
            )

        normalized_results = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            normalized_results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })

        if not normalized_results:
            return "No results found."

        return json.dumps(normalized_results, indent=2, ensure_ascii=False)
