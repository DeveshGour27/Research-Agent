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
from app.config import settings
from app.tools.base import BaseTool

# How many results to fetch from SearXNG (enough to have good coverage after filtering)
_FETCH_LIMIT = 20


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
        provider = str(getattr(settings, "web_search_provider", "searxng")).casefold()
        if provider != "searxng":
            raise ToolExecutionError(
                f"Unsupported web search provider: {provider}",
                tool_name=self.name,
            )

        # Keep provider selection centralized in configuration so deployments
        # can point at a non-default SearXNG instance without code changes.
        base_url = getattr(settings, "searxng_base_url", "http://localhost:8080").rstrip("/")
        url = f"{base_url}/search"
        params = urllib.parse.urlencode({"q": query, "format": "json"})
        req = urllib.request.Request(
            f"{url}?{params}",
            headers={"User-Agent": "ResearchAgent/1.0"},
        )
        # Kept for compatibility with callers/tests that inspect the request
        # object directly; urllib uses get_method() internally.
        req.method = "GET"

        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise ToolExecutionError(
                f"HTTP {error.code}: {error.reason}",
                tool_name=self.name,
            )
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

        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            raise ToolExecutionError(
                "SearXNG returned invalid results format.",
                tool_name=self.name,
            )

        results: list[dict[str, str]] = []
        provider_limit = int(getattr(settings, "web_search_max_results", 10))
        for item in raw_results:
            if len(results) >= provider_limit:
                break
            if not isinstance(item, dict):
                continue

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
                engine = item.get("engine") or item.get("source")
                if engine:
                    entry["engine"] = str(engine).strip()
                engines = item.get("engines")
                if isinstance(engines, list) and engines:
                    entry["engines"] = ", ".join(str(value) for value in engines)
                results.append(entry)

        if not results:
            return "No results found."

        return json.dumps(results, indent=2, ensure_ascii=False)
