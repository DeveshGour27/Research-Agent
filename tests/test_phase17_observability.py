import pytest
from app.observability.events import (
    LLMCallEvent,
    ToolCallEvent,
    RetrievalEvent,
    RerankEvent,
    ReflectionEvent,
    AgentStepEvent
)
from app.evaluation.benchmark_log_capture import EvaluationLogCaptureHandler
import logging
from unittest.mock import patch, MagicMock
from app.observability.redactor import Redactor

def test_event_creation_and_fields():
    event = LLMCallEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id="ps1",
        provider="openai", model="gpt-4", latency_s=1.2,
        input_tokens=10, output_tokens=20, total_tokens=30, task_type="GENERAL"
    )
    assert event.event_type == "LLMCallEvent"
    assert event.trace_id == "t1"
    assert event.event_data["provider"] == "openai"
    assert event.event_data["success"] is True

def test_event_emission(caplog):
    caplog.set_level(logging.INFO, logger="observability")
    event = AgentStepEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id="ps1",
        agent_name="TestAgent", step_name="step1", success=False, latency_s=0.5, error_info="failed"
    )
    event.emit()
    
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.msg == "AgentStepEvent"
    assert hasattr(record, "event")
    assert record.event["event_type"] == "AgentStepEvent"
    assert record.event["event_data"]["agent_name"] == "TestAgent"
    assert record.event["event_data"]["success"] is False

def test_failed_operations():
    event = ToolCallEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id="ps1",
        tool_name="test_tool", arguments={}, success=False, error_info="Tool crash"
    )
    assert event.event_data["success"] is False
    assert event.event_data["error_info"] == "Tool crash"

def test_telemetry_consumption_by_evaluation():
    handler = EvaluationLogCaptureHandler()
    
    # We must mock LogRecord because we don't have a real logger sending it, 
    # we just create dummy objects that resemble the structure
    class DummyRecord:
        def __init__(self, event_dict):
            self.event = event_dict
            
        def getMessage(self):
            return ""

    handler.records.append(DummyRecord({"event_type": "LLMCallEvent", "event_data": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}))
    handler.records.append(DummyRecord({"event_type": "LLMCallEvent", "event_data": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}}))
    handler.records.append(DummyRecord({"event_type": "ToolCallEvent", "event_data": {}}))
    handler.records.append(DummyRecord({"event_type": "RetrievalEvent", "event_data": {}}))
    handler.records.append(DummyRecord({"event_type": "MemoryRetrievedEvent", "event_data": {}}))
    
    usage = handler.get_token_usage()
    assert usage["input_tokens"] == 30
    assert usage["output_tokens"] == 15
    assert usage["total_tokens"] == 45
    
    assert handler.get_llm_calls() == 2
    assert handler.get_tool_calls() == 1
    assert handler.get_retrieval_calls() == 2

def test_sensitive_data_protection():
    # Verify that redactor works on our event fields
    event = ToolCallEvent(
        trace_id="t1", run_id="r1", span_id="s1", parent_span_id="ps1",
        tool_name="test_tool", arguments={"api_key": "secret123", "password": "abc"}, success=True
    )
    
    # Redactor is applied in emit()
    with patch("app.logger.logging.Logger.info") as mock_info:
        event.emit()
        called_kwargs = mock_info.call_args[1]
        emitted_event = called_kwargs["extra"]["event"]
        assert emitted_event["event_data"]["arguments"]["api_key"] == "[REDACTED]"
        assert emitted_event["event_data"]["arguments"]["password"] == "[REDACTED]"

