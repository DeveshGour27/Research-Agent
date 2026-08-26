import os
os.environ['GROQ_API_KEY'] = 'test'
"""Tests for Phase 9 Reflection Module and Bounded Retry."""

import json
import pytest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
from app.config import settings
from app.reflection.contracts import ReflectionDecision, ReflectionResult
from app.reflection.evaluator import ReflectionEvaluator
from app.reflection.policy import EvidenceSufficiencyPolicy
from rag.hybrid_search import Retrieved

def test_reflection_contracts_valid():
    """Verify ReflectionResult validates correctly on valid input."""
    result = ReflectionResult(
        decision="ACCEPT",
        confidence=0.9,
        reason="Good evidence",
        retry_retrieval=False
    )
    assert result.decision == ReflectionDecision.ACCEPT
    assert result.confidence == 0.9

def test_reflection_contracts_invalid_bounds():
    """Verify ReflectionResult rejects confidence out of bounds."""
    with pytest.raises(ValidationError):
        ReflectionResult(
            decision="ACCEPT",
            confidence=1.5,
            reason="Too confident",
        )
    with pytest.raises(ValidationError):
        ReflectionResult(
            decision="ACCEPT",
            confidence=-0.1,
            reason="Not confident",
        )

def test_reflection_contracts_invalid_decision():
    """Verify ReflectionResult rejects unknown decisions."""
    with pytest.raises(ValidationError):
        ReflectionResult(
            decision="MAYBE",
            confidence=0.5,
            reason="Not sure",
        )

def test_evaluator_safe_failure_malformed_json(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test')
    """Verify evaluator falls back to safe failure on malformed LLM JSON."""
    from app.config import settings
    settings.groq_api_key = 'test'
    evaluator = ReflectionEvaluator()
    evaluator.provider.generate = MagicMock(return_value=MagicMock(content="Not JSON at all"))
    
    docs = [Retrieved(chunk_id="1", score=1.0, source_score=1.0, vector_score=1.0, metadata={"text": "Hello"})]
    result = evaluator.evaluate("query", docs)
    
    assert result.decision == ReflectionDecision.INSUFFICIENT_EVIDENCE
    assert result.confidence == 0.0
    assert result.retry_retrieval is True

def test_evaluator_safe_failure_markdown_json(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test')
    """Verify evaluator cleans markdown JSON formatting."""
    from app.config import settings
    settings.groq_api_key = 'test'
    evaluator = ReflectionEvaluator()
    json_response = '```json\n{"decision": "ACCEPT", "confidence": 0.8, "reason": "Yes"}\n```'
    evaluator.provider.generate = MagicMock(return_value=MagicMock(content=json_response))
    
    docs = [Retrieved(chunk_id="1", score=1.0, source_score=1.0, vector_score=1.0, metadata={"text": "Hello"})]
    result = evaluator.evaluate("query", docs)
    
    assert result.decision == ReflectionDecision.ACCEPT
    assert result.confidence == 0.8

def test_policy_empty_evidence_override():
    """Verify policy overrides ACCEPT if evidence is empty."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ReflectionResult(
        decision=ReflectionDecision.ACCEPT,
        confidence=0.9,
        reason="Model hallucinated acceptance of empty evidence."
    )
    policy = EvidenceSufficiencyPolicy(evaluator=evaluator)
    
    result = policy.evaluate("query", [], current_attempt=0)
    assert result.decision == ReflectionDecision.INSUFFICIENT_EVIDENCE

def test_policy_enforces_retry_budget():
    """Verify policy forces INSUFFICIENT_EVIDENCE when retry budget exhausted."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ReflectionResult(
        decision=ReflectionDecision.RETRY_RETRIEVAL,
        confidence=0.5,
        reason="Needs more info"
    )
    policy = EvidenceSufficiencyPolicy(evaluator=evaluator)
    
    # attempt = 0, limit = 1 -> should allow retry
    with patch("app.reflection.policy.settings.max_retrieval_retries", 1):
        res1 = policy.evaluate("query", [Retrieved(chunk_id="1", score=1.0, source_score=1.0, vector_score=1.0, metadata={})], 0)
        assert res1.decision == ReflectionDecision.RETRY_RETRIEVAL
        
        # attempt = 1, limit = 1 -> should block retry
        res2 = policy.evaluate("query", [Retrieved(chunk_id="1", score=1.0, source_score=1.0, vector_score=1.0, metadata={})], 1)
        assert res2.decision == ReflectionDecision.INSUFFICIENT_EVIDENCE

def test_policy_max_budget_exhausted_fast_fail():
    """Verify policy instantly fails if current_attempt > max retries, without calling evaluator."""
    evaluator = MagicMock()
    policy = EvidenceSufficiencyPolicy(evaluator=evaluator)
    
    with patch("app.reflection.policy.settings.max_retrieval_retries", 1):
        res = policy.evaluate("query", [], 2)
        assert res.decision == ReflectionDecision.INSUFFICIENT_EVIDENCE
        evaluator.evaluate.assert_not_called()

@patch("app.reflection.policy.ReflectionEvaluator")
@patch("app.agent.specialized.rag_agent.Retriever")
def test_rag_agent_bounded_retry_loop(mock_retriever_class, mock_evaluator_class):
    """Test RAGAgent loops up to max retries and then exits safely."""
    from app.agent.specialized.rag_agent import RAGAgent
    from app.agent.contracts import AgentRequest
    from app.agent.execution_context import AgentExecutionContext
    
    # Mock Retriever
    mock_retriever = mock_retriever_class.return_value
    mock_retriever.retrieve.return_value = [Retrieved(chunk_id="1", score=1.0, source_score=1.0, vector_score=1.0, metadata={"text": "dummy"})]
    
    # Mock Evaluator within Policy
    mock_evaluator = mock_evaluator_class.return_value
    # Always request retry
    mock_evaluator.evaluate.return_value = ReflectionResult(
        decision=ReflectionDecision.RETRY_RETRIEVAL,
        confidence=0.5,
        reason="Need retry"
    )
    
    agent = RAGAgent(retriever=mock_retriever)
    req = AgentRequest(input_text="test query", context=AgentExecutionContext(task="test", user_id="u1"))
    
    with patch.object(settings, "max_retrieval_retries", 2), \
         patch.object(settings, "reflection_enabled", True):
        res = agent.execute(req)
        
        assert res.success is True
        assert res.output == "No relevant documents found."
        assert mock_retriever.retrieve.call_count == 3  # attempt 0, 1, 2
