"""Web search tool using DuckDuckGo.

Returns results with title, url, and snippet.
"""

from __future__ import annotations

import json
from typing import Any

from app.exceptions import ToolExecutionError
from app.config import settings
from app.tools.base import BaseTool

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

class WebSearchTool(BaseTool):
    """Perform a web search using DuckDuckGo."""

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
        if DDGS is None:
            raise ToolExecutionError(
                "ddgs package is not installed.",
                tool_name=self.name,
            )
        
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ToolExecutionError(
                "The 'query' argument must not be empty.",
                tool_name=self.name,
            )
        
        provider = str(getattr(settings, "web_search_provider", "duckduckgo")).casefold()
        if provider != "duckduckgo":
            raise ToolExecutionError(
                f"Unsupported web search provider: {provider}",
                tool_name=self.name,
            )

        provider_limit = int(getattr(settings, "web_search_max_results", 10))

        try:
            results: list[dict[str, str]] = []
            with DDGS() as ddgs:
                ddg_results = ddgs.text(query, max_results=provider_limit)
                if ddg_results:
                    for item in ddg_results:
                        entry = {
                            "title": item.get("title", ""),
                            "url": item.get("href", ""),
                            "snippet": item.get("body", ""),
                        }
                        results.append(entry)
        except Exception as error:
            raise ToolExecutionError(
                f"Unexpected error during web search: {type(error).__name__} - {str(error)}",
                tool_name=self.name,
            )

        if not results:
            return "No results found."

        return json.dumps(results, indent=2, ensure_ascii=False)
