"""Web page content fetcher tool for deep primary-source verification."""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from bs4 import BeautifulSoup

from app.exceptions import ToolExecutionError
from app.tools.base import BaseTool


class WebFetchTool(BaseTool):
    """Fetch and extract readable text from a URL to verify primary source claims."""

    name = "web_fetch"
    description = (
        "Fetch the readable text content of a specific web page URL. "
        "Use this tool to read the full text of articles, press releases, or academic papers to verify claims."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The HTTP or HTTPS URL of the web page to read.",
            }
        },
        "required": ["url"],
    }

    _TIMEOUT = 10
    _MAX_CHARS = 4000

    def execute(self, **kwargs: object) -> str:
        url = str(kwargs.get("url", "")).strip()
        if not url:
            raise ToolExecutionError("The 'url' argument must not be empty.", tool_name=self.name)

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ToolExecutionError(f"Invalid URL scheme: '{parsed.scheme}'. Must be http or https.", tool_name=self.name)

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return f"Non-HTML content type: {content_type}"
                html_bytes = resp.read(250_000)  # Read at most 250KB
                html_text = html_bytes.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            return f"HTTP error {error.code}: {error.reason}"
        except urllib.error.URLError as error:
            return f"Connection error: {getattr(error, 'reason', error)}"
        except TimeoutError:
            return "Connection timed out after 10 seconds."
        except Exception as error:
            return f"Failed to fetch URL: {error}"

        # Parse and extract text using BeautifulSoup
        try:
            soup = BeautifulSoup(html_text, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
                tag.decompose()

            # Extract title
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Extract paragraphs
            paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "article"])]
            text_blocks = [p for p in paragraphs if len(p) > 40]

            if not text_blocks:
                body_text = soup.get_text(separator=" ", strip=True)
                body_text = re.sub(r"\s+", " ", body_text)
                combined = body_text[:self._MAX_CHARS]
            else:
                combined = "\n\n".join(text_blocks)
                combined = re.sub(r"\s+", " ", combined)[:self._MAX_CHARS]

            result = f"Title: {title}\n\nContent:\n{combined.strip()}"
            return result
        except Exception as err:
            return f"Error extracting page text: {err}"
