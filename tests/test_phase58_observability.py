"""Tests for Phase 5.8 Observability (Step 1)."""

import json
import logging
from typing import Any

from app.observability.redactor import Redactor
from app.observability.events import (
    BaseEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    PlanGeneratedEvent,
    PlanStepCompletedEvent,
    HandoffInitiatedEvent,
    HandoffResolvedEvent,
    RetryAttemptedEvent,
    TimeoutCancellationEvent,
    ToolCalledEvent,
    ToolCompletedEvent,
    MemoryRetrievedEvent,
)


def test_redactor_openai_key():
    redactor = Redactor()
    text = "Here is my key: sk-proj-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    redacted = redactor.redact(text)
    assert redacted == "Here is my key: [REDACTED]"
    
    text2 = "sk-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    assert redactor.redact(text2) == "[REDACTED]"


def test_redactor_groq_key():
    redactor = Redactor()
    text = "Groq key gsk_1234567890abcdefghij1234567890abcdefghij is here"
    redacted = redactor.redact(text)
    assert redacted == "Groq key [REDACTED] is here"


def test_redactor_bearer_token():
    redactor = Redactor()
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    redacted = redactor.redact(text)
    assert redacted == "Authorization: Bearer [REDACTED]"
    
    # Check it doesn't over-redact normal words
    assert redactor.redact("The bearer of bad news") == "The bearer of bad news"


def test_redactor_generic_secrets():
    redactor = Redactor()
    # Should redact
    assert redactor.redact("api_key='1234567890abcdef'") == "api_key='[REDACTED]'"
    assert redactor.redact('{"secret": "1234567890abcdef"}') == '{"secret": "[REDACTED]"}'
    assert redactor.redact('token=1234567890abcdef') == 'token=[REDACTED]'
    assert redactor.redact('API_KEY: 1234567890abcdef') == 'API_KEY: [REDACTED]'
    
    # Should not over-redact normal text
    assert redactor.redact('The secret to success is hard work') == 'The secret to success is hard work'


def test_redactor_recursive():
    redactor = Redactor()
    payload = {
        "user_id": 123,
        "api_key": "gsk_1234567890abcdefghij1234567890abcdefghij",
        "nested": {
            "key": "sk-proj-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            "list": [
                "token=1234567890abcdef",
                "normal string"
            ]
        }
    }
    
    redacted = redactor.redact(payload)
    assert redacted["user_id"] == 123
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["key"] == "[REDACTED]"
    assert redacted["nested"]["list"][0] == "token=[REDACTED]"
    assert redacted["nested"]["list"][1] == "normal string"


def test_redactor_malformed_payload():
    redactor = Redactor()
    
    class Unstrigifiable:
        def __str__(self):
            raise ValueError("I cannot be stringified")
    
    payload = {"key": Unstrigifiable()}
    
    # The redactor should not crash, it might return a stringification error or safely handle it
    # Currently _redact_recursive calls _redact_string(str(k)), but doesn't call str() on v unless it's a string.
    # So it will hit the 'else' branch and return Unstrigifiable directly.
    redacted = redactor.redact(payload)
    assert isinstance(redacted["key"], Unstrigifiable)
    
    # Let's force an exception by making keys unstringifiable
    payload2 = {Unstrigifiable(): "value"}
    redacted2 = redactor.redact(payload2)
    assert redacted2 == "<[REDACTED]: Processing Failed>"


def test_event_construction_and_fields():
    event = AgentExecutionStartedEvent(
        trace_id="trace1",
        run_id="run1",
        span_id="span1",
        parent_span_id="pspan1",
        agent_name="TestAgent",
        input_text="Hello world"
    )
    
    assert event.event_type == "AgentExecutionStarted"
    assert event.trace_id == "trace1"
    assert event.run_id == "run1"
    assert event.span_id == "span1"
    assert event.parent_span_id == "pspan1"
    assert event.event_data["agent_name"] == "TestAgent"
    assert event.event_data["input_text"] == "Hello world"
    assert isinstance(event.timestamp, str)


def test_event_emission_and_redaction(caplog):
    caplog.set_level(logging.INFO)
    
    event = ToolCalledEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id="ps1",
        tool_name="web_search",
        arguments={"query": "my secret API_KEY: 1234567890abcdef is here"}
    )
    event.emit()
    
    assert len(caplog.records) == 1
    record = caplog.records[0]
    
    # Verify logger correctly got the extra event dictionary
    assert hasattr(record, "event")
    event_dict = record.event
    
    assert event_dict["event_type"] == "ToolCalled"
    assert event_dict["trace_id"] == "t1"
    
    # Verify redaction occurred on event_data
    assert "API_KEY: [REDACTED]" in event_dict["event_data"]["arguments"]["query"]
    assert "1234567890abcdef" not in event_dict["event_data"]["arguments"]["query"]


def test_event_failure_isolation(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    
    event = BaseEvent(
        event_type="TestEvent",
        trace_id="t1",
        run_id="r1",
        span_id="s1",
        parent_span_id=None,
        event_data={"key": "value"}
    )
    
    # Monkeypatch the redactor to throw an exception to simulate failure
    def mock_redact(*args, **kwargs):
        raise RuntimeError("Fake Redactor Crash")
        
    import app.observability.events as events_module
    monkeypatch.setattr(events_module._redactor, "redact", mock_redact)
    
    # Emit should NOT raise an exception
    event.emit()
    
    # The event should not be logged because it failed, but the application shouldn't crash
    # OR the exception is just swallowed. Let's verify it swallowed.
    assert len(caplog.records) == 0


def test_all_event_types_construction():
    events = [
        AgentExecutionStartedEvent("t", "r", "s", "p", "agent", "input"),
        AgentExecutionCompletedEvent("t", "r", "s", "p", "agent", True, "out"),
        PlanGeneratedEvent("t", "r", "s", "p", "pid", 3, False),
        PlanStepCompletedEvent("t", "r", "s", "p", "pid", "sid", "completed"),
        HandoffInitiatedEvent("t", "r", "s", "p", "type", "msg"),
        HandoffResolvedEvent("t", "r", "s", "p", "type", True, 2),
        RetryAttemptedEvent("t", "r", "s", "p", "error", 1),
        TimeoutCancellationEvent("t", "r", "s", "p", "timeout"),
        ToolCalledEvent("t", "r", "s", "p", "tool", {}),
        ToolCompletedEvent("t", "r", "s", "p", "tool", True, "done"),
        MemoryRetrievedEvent("t", "r", "s", "p", "query", 5, 2),
    ]
    
    for event in events:
        assert event.trace_id == "t"
        assert event.run_id == "r"
        assert isinstance(event.event_type, str)
        assert isinstance(event.event_data, dict)
        assert isinstance(event.timestamp, str)
