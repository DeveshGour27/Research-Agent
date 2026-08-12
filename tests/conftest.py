"""Global test configuration and fixtures."""

import pytest
from unittest.mock import patch
from app.config import settings

@pytest.fixture(autouse=True)
def disable_reflection_for_tests():
    """Disable Phase 9 reflection by default in tests to prevent LLM calls."""
    with patch.object(settings, "reflection_enabled", False):
        yield
