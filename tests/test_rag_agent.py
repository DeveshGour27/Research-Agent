"""Tests for the RAGAgent."""

import json
from unittest.mock import MagicMock

import pytest

from app.agent.contracts import AgentRequest
from app.agent.execution_context import AgentExecutionContext
from app.agent.specialized.rag_agent import RAGAgent
from app.exceptions import AgentExecutionError
from rag.hybrid_search import Retrieved


def test_rag_agent_identity_and_capabilities() -> None:
    agent = RAGAgent(retriever=MagicMock())
    assert agent.identity.name == "rag_agent"
    assert "rag_search" in agent.capabilities.task_types
    assert "web_search" not in agent.capabilities.task_types
    assert agent.capabilities.retrieval is True

def test_rag_agent_tool_isolation() -> None:
    agent = RAGAgent(retriever=MagicMock())
    # Should not have any tool registry or web search tool
    assert not hasattr(agent, "_tool_registry")

def test_rag_agent_successful_retrieval() -> None:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Retrieved(chunk_id="1", score=0.9, source_score=0.9, vector_score=0.9, metadata={"text": "Doc 1"}),
        Retrieved(chunk_id="2", score=0.8, source_score=0.8, vector_score=0.8, metadata={"text": "Doc 2"}),
    ]

    agent = RAGAgent(retriever=mock_retriever)
    context = AgentExecutionContext(task="Find something")
    request = AgentRequest(input_text="What is X?", context=context)
    
    result = agent.execute(request)

    mock_retriever.retrieve.assert_called_once_with(query="What is X?")
    assert result.success is True
    assert result.request is request
    assert result.context is context
    
    output_data = json.loads(result.output)
    assert len(output_data) == 2
    assert output_data[0]["chunk_id"] == "1"
    assert output_data[0]["text"] == "Doc 1"

def test_rag_agent_empty_retrieval() -> None:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    agent = RAGAgent(retriever=mock_retriever)
    request = AgentRequest(input_text="Unknown topic")
    result = agent.execute(request)

    assert result.success is True
    assert result.output == "No relevant documents found."

def test_rag_agent_retrieval_failure() -> None:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = Exception("Database connection lost")

    agent = RAGAgent(retriever=mock_retriever)
    request = AgentRequest(input_text="Query")
    
    result = agent.execute(request)
    
    assert result.success is False
    assert result.output is None
    assert isinstance(result.error, AgentExecutionError)
    assert "RAG retrieval failed" in result.error.message

def test_rag_agent_empty_input() -> None:
    agent = RAGAgent(retriever=MagicMock())
    request = AgentRequest(input_text="   ")
    with pytest.raises(AgentExecutionError, match="must not be empty"):
        agent.execute(request)
