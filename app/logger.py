"""
Logging module for the Production AI Research Agent.

Responsibilities:
    - Configure a structured (JSON) log formatter
    - Expose a factory function `get_logger` used by every other module
    - Honour the log level set in `app.config.settings`

Design Decisions:
    JSON output is chosen over plain-text because structured logs are
    trivially searchable in log-aggregation platforms (Datadog, CloudWatch,
    Loki, etc.) and easy to parse programmatically during debugging.

    Each logger is a child of the root "agent" logger so that the log
    level can be changed once — in Settings — and propagate everywhere.

Usage:
    from app.logger import get_logger

    logger = get_logger(__name__)
    logger.info("tool executed", extra={"tool": "web_search", "duration_ms": 120})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.config import settings


# ------------------------------------------------------------------ #
# Formatter
# ------------------------------------------------------------------ #

class _JSONFormatter(logging.Formatter):
    """
    Render log records as single-line JSON objects.

    Every record includes at minimum:
        timestamp, level, logger, message

    Optional fields written when present on the record:
        exception  — formatted traceback string
        *          — any key/value pairs passed via ``extra={}``
    """

    # Keys that are built into every LogRecord.
    # We must never re-emit these as custom extra fields — doing so causes
    # logging.Logger.makeRecord to raise KeyError at the call site.
    _RESERVED: frozenset[str] = frozenset(
        {
            # Standard LogRecord attributes
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message",
            "module", "msecs", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "taskName",
            "thread", "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        # Ensure record.message is populated
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Append traceback when an exception is attached
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Append stack info when present
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # Append any caller-supplied extra fields
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _configure_root_logger() -> None:
    """
    Set up the application-wide root logger ("agent") once.

    Called automatically when this module is first imported.
    Subsequent calls are no-ops because of the ``_configured`` guard.
    """
    root = logging.getLogger("agent")

    # Idempotent — only configure once per process.
    if root.handlers:
        return

    level = getattr(logging, settings.log_level.value, logging.INFO)
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_JSONFormatter())

    root.addHandler(handler)
    # Prevent double-logging if the Python root logger also has handlers.
    root.propagate = False


# Run once at import time.
_configure_root_logger()


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger of the application root logger.

    Args:
        name: Typically ``__name__`` of the calling module.
              The resulting logger name will be ``agent.<name>``.

    Returns:
        A ``logging.Logger`` instance ready for use.

    Example::

        from app.logger import get_logger

        logger = get_logger(__name__)
        # Use non-reserved key names in extra= (avoid e.g. 'module', 'msg')
        logger.info("starting retrieval", extra={"query": query, "source": "retriever"})
    """
    child_name = f"agent.{name}" if not name.startswith("agent") else name
    return logging.getLogger(child_name)
