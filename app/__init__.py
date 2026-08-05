"""
app package for the Production AI Research Agent.

Exposes the foundational infrastructure so that sibling modules and
external entry points can import from a single, stable surface:

    from app import settings, get_logger
    from app.exceptions import ToolExecutionError
    from app.constants import REPORTS_DIR
"""

from app.config import Settings, settings
from app.logger import get_logger

__all__ = [
    "Settings",
    "settings",
    "get_logger",
]
