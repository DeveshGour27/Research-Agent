import pytest
from unittest.mock import patch, MagicMock

import main
from app.tools.calculator import CalculatorTool
from app.tools.web_search import WebSearchTool
from app.agent import Agent
from app.agent.supervisor import Supervisor
from app.agent.specialized.web_agent import WebResearchAgent
from app.agent.specialized.rag_agent import RAGAgent

@patch('builtins.input', side_effect=['', '', 'exit'])
@patch('main.Agent')
@patch('main.ToolRegistry')
@patch('main.JsonFileMemoryStore')
def test_run_agent_wiring(mock_memory_store, mock_tool_registry, mock_agent, mock_input):
    mock_registry_instance = MagicMock()
    mock_tool_registry.return_value = mock_registry_instance
    
    # We patch run_agent loop to just exit immediately via mocked input
    main.run_agent()
    
    # Verify tools registered
    registered_tools = [call[0][0].name for call in mock_registry_instance.register.call_args_list]
    assert 'calculate' in registered_tools
    assert 'web_search' in registered_tools
    
    # Verify agent instantiated with hitl_service
    mock_agent.assert_called_once()
    kwargs = mock_agent.call_args.kwargs
    assert 'hitl_service' in kwargs
    assert kwargs['hitl_service'] is not None

@patch('builtins.input', side_effect=['exit'])
@patch('main.Supervisor')
@patch('main.AgentRegistry')
@patch('main.Agent')
@patch('main.ToolRegistry')
def test_run_supervisor_wiring(mock_tool_registry, mock_agent, mock_agent_registry, mock_supervisor, mock_input):
    mock_registry_instance = MagicMock()
    mock_agent_registry.return_value = mock_registry_instance
    
    mock_tool_registry_instance = MagicMock()
    mock_tool_registry.return_value = mock_tool_registry_instance
    
    # mock Agent so we can check its identity
    mock_general_agent = MagicMock()
    mock_general_agent.identity.name = "production-research-agent"
    mock_agent.return_value = mock_general_agent

    main.run_supervisor()
    
    # Verify tools registered for general agent
    registered_tools = [call[0][0].name for call in mock_tool_registry_instance.register.call_args_list]
    assert 'calculate' in registered_tools
    assert 'web_search' in registered_tools
    
    # Verify agents registered in Supervisor's AgentRegistry
    registered_agents = [call[0][0].identity.name for call in mock_registry_instance.register.call_args_list]
    
    assert 'production-research-agent' in registered_agents
    assert 'web_research_agent' in registered_agents
    assert 'rag_agent' in registered_agents
    
    # Verify supervisor instantiated with hitl_service
    mock_supervisor.assert_called_once()
    kwargs = mock_supervisor.call_args.kwargs
    assert 'hitl_service' in kwargs
    assert kwargs['hitl_service'] is not None
