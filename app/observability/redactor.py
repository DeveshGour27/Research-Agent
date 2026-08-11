"""Redactor for safely masking sensitive information before logging or emitting events."""

from __future__ import annotations

import re
from typing import Any


class Redactor:
    """Recursively redacts sensitive information from payloads.
    
    Ensures that observability failure does not break the application.
    """

    def __init__(self, replacement: str = "[REDACTED]") -> None:
        self._replacement = replacement

    def redact(self, payload: Any) -> Any:
        """
        Safely redact sensitive values from the payload.
        Never raises an exception, ensuring control-flow isolation.
        """
        try:
            return self._redact_recursive(payload, depth=0)
        except Exception:
            # Observability must NEVER become a control-flow dependency.
            return f"<{self._replacement}: Processing Failed>"

    def _redact_recursive(self, obj: Any, depth: int) -> Any:
        if depth > 100:  # Prevent infinite recursion or stack overflows
            return f"<{self._replacement}: Max Depth Exceeded>"

        if isinstance(obj, str):
            return self._redact_string(obj)
        elif isinstance(obj, dict):
            return {
                self._redact_string(str(k)): self._redact_recursive(v, depth + 1)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._redact_recursive(v, depth + 1) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._redact_recursive(v, depth + 1) for v in obj)
        elif isinstance(obj, set):
            return {self._redact_recursive(v, depth + 1) for v in obj}
        else:
            # Leave primitives like int, float, bool, None as is
            return obj

    def _redact_string(self, text: str) -> str:
        """Applies regex patterns to redact sensitive substrings."""
        if not isinstance(text, str):
            return text
            
        redacted_text = text
        
        # 1. OpenAI Keys
        redacted_text = re.sub(
            r'sk-(?:proj-)?[A-Za-z0-9_-]{20,}', 
            self._replacement, 
            redacted_text
        )
        
        # 2. Groq Keys
        redacted_text = re.sub(
            r'gsk_[A-Za-z0-9_-]{20,}', 
            self._replacement, 
            redacted_text
        )
        
        # 3. Bearer tokens (must not match spaces)
        redacted_text = re.sub(
            r'(?i)(bearer\s+)[A-Za-z0-9_\-\.]{15,}', 
            rf'\g<1>{self._replacement}', 
            redacted_text
        )
        
        # 4. Generic secrets (api_key=..., secret: ...)
        # Expects the secret to be alphanumeric with dashes/dots/underscores, no spaces.
        redacted_text = re.sub(
            r'(?i)((?:api[_-]?key|secret|token)[\'"]?\s*[:=]\s*[\'"]?)([A-Za-z0-9_\-\.]{15,})', 
            rf'\g<1>{self._replacement}', 
            redacted_text
        )
        
        return redacted_text
