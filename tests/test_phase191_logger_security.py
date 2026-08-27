"""Tests for Phase 19.1: Logger Security Fix."""

import json
import logging
from io import StringIO

from app.logger import get_logger, _JSONFormatter


def test_logger_redacts_secrets_in_extra() -> None:
    """Verify that secrets passed in extra={...} are redacted in JSON output."""
    
    # Create a custom logger with the JSON formatter attached to a StringIO buffer
    logger = logging.getLogger("test_logger_redaction")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    
    # Log a message with sensitive extra fields
    sensitive_extra = {
        "user_input": "normal text",
        "api_key": "sk-proj-1234567890abcdefghij1234567890",
        "nested": {
            "password": "supersecretpassword",
            "token": "gsk_1234567890abcdefghij1234567890",
            "bearer_header": "Bearer 1234567890abcdefghij"
        }
    }
    
    logger.info("Executing tool", extra={"arguments": sensitive_extra})
    
    output = buffer.getvalue().strip()
    assert output, "Log output should not be empty"
    
    # Parse the JSON log
    log_data = json.loads(output)
    
    # Assertions
    assert log_data["message"] == "Executing tool"
    
    # Check that the arguments dictionary was redacted
    args = log_data.get("arguments", {})
    assert args.get("user_input") == "normal text"
    assert args.get("api_key") == "[REDACTED]"
    
    # Check nested redaction
    nested = args.get("nested", {})
    assert nested.get("password") == "[REDACTED]"
    assert nested.get("token") == "[REDACTED]"
    
    # Check that generic secrets are redacted from strings via regex
    assert "[REDACTED]" in nested.get("bearer_header", "")
    assert "1234567890" not in nested.get("bearer_header", "")


def test_logger_redacts_secrets_in_message() -> None:
    """Verify that secrets inside the main log message are redacted."""
    logger = logging.getLogger("test_logger_redaction_message")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    
    # Log a message containing a secret
    logger.info("Failed to authenticate with token: gsk_1234567890abcdefghij1234567890")
    
    output = buffer.getvalue().strip()
    log_data = json.loads(output)
    
    assert "[REDACTED]" in log_data["message"]
    assert "gsk_" not in log_data["message"]
    assert "1234567890abcdefghij" not in log_data["message"]


def test_logger_redacts_lists() -> None:
    """Verify that lists inside extra are also redacted."""
    logger = logging.getLogger("test_logger_redaction_list")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    
    logger.info("Keys", extra={"keys": ["sk-proj-1234567890abcdefghij1234567890", "safe_value"]})
    
    output = buffer.getvalue().strip()
    log_data = json.loads(output)
    
    keys = log_data.get("keys", [])
    assert keys[0] == "[REDACTED]"
    assert keys[1] == "safe_value"
