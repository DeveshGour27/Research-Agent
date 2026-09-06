"""Web search tool using local SearXNG instance.

Returns results with title, url, snippet, and publication date preserved so
downstream agents can filter by year or recency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from app.exceptions import ToolExecutionError
from app.tools.base import BaseTool

# How many results to fetch from SearXNG (enough to have good coverage after filtering)
_FETCH_LIMIT = 20
# How many results to return to the agent
_RETURN_LIMIT = 10


class WebSearchTool(BaseTool):
    """Perform a web search using local SearXNG instance (http://localhost:8080)."""

    name = "web_search"
    description = (
        "Search the web for up-to-date information on a given topic. "
        "Returns a list of relevant search results including titles, URLs, snippets, and publication dates."
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

    _TIMEOUT = 10

    def execute(self, **kwargs: object) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ToolExecutionError(
                "The 'query' argument must not be empty.",
                tool_name=self.name,
            )

        url = "http://localhost:8080/search"
        params = urllib.parse.urlencode({"q": query, "format": "json"})
        req = urllib.request.Request(
            f"{url}?{params}",
            headers={"User-Agent": "ResearchAgent/1.0"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise ToolExecutionError(
                f"Failed to connect to SearXNG: {getattr(error, 'reason', error)}",
                tool_name=self.name,
            )
        except TimeoutError:
            raise ToolExecutionError(
                "Web search request timed out.",
                tool_name=self.name,
            )
        except json.JSONDecodeError:
            raise ToolExecutionError(
                "SearXNG returned malformed JSON.",
                tool_name=self.name,
            )
        except Exception as error:
            raise ToolExecutionError(
                f"Unexpected error during web search: {type(error).__name__}",
                tool_name=self.name,
            )

        results: list[dict[str, str]] = []
        for item in data.get("results", []):
            if len(results) >= _RETURN_LIMIT:
                break

            title = item.get("title", "").strip()
            url_str = item.get("url", "").strip()
            content = item.get("content", "").strip()

            # Capture publication date — SearXNG uses different field names across engines
            pub_date: str = (
                item.get("publishedDate")
                or item.get("published_date")
                or item.get("date")
                or ""
            )

            if title and url_str:
                entry: dict[str, str] = {
                    "title": title,
                    "url": url_str,
                    "snippet": content,
                }
                if pub_date:
                    entry["date"] = str(pub_date).strip()
                results.append(entry)

        if not results:
            return "No results found."

        return json.dumps(results, indent=2, ensure_ascii=False)
