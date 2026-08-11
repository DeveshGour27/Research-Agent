"""Web search tool using Tavily API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from app.config import settings
from app.exceptions import ToolExecutionError
from app.tools.base import BaseTool


class WebSearchTool(BaseTool):
    """Perform a web search using the configured provider (e.g., Tavily)."""

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
        if provider != "tavily":
            raise ToolExecutionError(
                f"Unsupported web search provider: {provider}",
                tool_name=self.name,
            )

        api_key = settings.web_search_api_key
        if not api_key:
            raise ToolExecutionError(
                "Web search API key is not configured.",
                tool_name=self.name,
            )

        max_results = settings.web_search_max_results
        timeout = settings.web_search_timeout_seconds

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": max_results,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
                result_data = json.loads(response_body)
        except urllib.error.HTTPError as error:
            # Handle HTTP errors safely (masking secrets)
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
        for item in results[:max_results]:  # Enforce limit just in case
            if not isinstance(item, dict):
                continue
            normalized_results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),  # Tavily uses 'content' for snippet
            })

        if not normalized_results:
            return "No results found."

        return json.dumps(normalized_results, indent=2, ensure_ascii=False)
