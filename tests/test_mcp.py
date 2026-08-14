import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.mcp.contracts import MCPServerConfig, MCPToolMetadata, MCPInvocationRequest
from app.mcp.policy import MCPPolicy, PolicyDecision
from app.mcp.client import MCPClient
from app.mcp.tool import MCPTool
from app.mcp.registry import MCPRegistry
from app.mcp.transport import MockMCPTransport
from app.exceptions import ToolExecutionError
from app.mcp.errors import MCPPolicyDeniedError

@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.initialize = AsyncMock()
    
    # Mock list_tools
    tool_mock = MagicMock()
    tool_mock.name = "test_tool"
    tool_mock.description = "A test tool"
    tool_mock.inputSchema = {"type": "object", "properties": {}}
    session.list_tools.return_value = MagicMock(tools=[tool_mock])
    
    # Mock call_tool
    result_mock = MagicMock()
    result_mock.isError = False
    result_mock.is_error = False
    content_mock = MagicMock()
    content_mock.type = "text"
    content_mock.text = "success"
    result_mock.content = [content_mock]
    session.call_tool.return_value = result_mock
    
    return session

def test_mcp_policy():
    policy = MCPPolicy(allowed_servers=["safe_server"])
    assert policy.evaluate_server("safe_server") == PolicyDecision.ALLOW
    assert policy.evaluate_server("unsafe_server") == PolicyDecision.DENY
    
    assert policy.evaluate_tool("safe_server", "any_tool") == PolicyDecision.ALLOW
    assert policy.evaluate_tool("unsafe_server", "any_tool") == PolicyDecision.DENY

@patch("app.mcp.client.ClientSession")
def test_mcp_client(mock_client_session, mock_session):
    mock_client_session.return_value.__aenter__.return_value = mock_session
    
    config = MCPServerConfig(command="dummy", args=[], enabled=True)
    client = MCPClient("test_server", config, MockMCPTransport)
    
    # Connect
    client.connect()
    mock_session.initialize.assert_awaited_once()
    
    # Discover tools
    tools = client.discover_tools()
    assert len(tools) == 1
    assert tools[0].name == "test_tool"
    
    # Invoke tool
    req = MCPInvocationRequest(server_name="test_server", tool_name="test_tool", arguments={})
    res = client.invoke_tool(req)
    assert res.content == "success"
    assert res.is_error is False
    
    client.disconnect()

def test_mcp_tool():
    client_mock = MagicMock()
    client_mock.name = "test_server"
    
    policy = MCPPolicy()
    metadata = MCPToolMetadata(name="my_tool", description="desc", input_schema={})
    
    tool = MCPTool(client_mock, policy, metadata)
    assert tool.name == "mcp.test_server.my_tool"
    
    # Successful execution
    res_mock = MagicMock()
    res_mock.is_error = False
    res_mock.content = "worked"
    client_mock.invoke_tool.return_value = res_mock
    
    assert tool.execute(arg1="val") == "worked"
    client_mock.invoke_tool.assert_called_once()
    
    # Failed execution (MCP error)
    res_mock.is_error = True
    res_mock.content = "failed"
    
    with pytest.raises(ToolExecutionError, match="MCP Tool returned error: failed"):
        tool.execute()

def test_mcp_registry():
    policy = MCPPolicy(allowed_servers=["allowed_server"])
    registry = MCPRegistry(policy=policy, transport_cls=MockMCPTransport)
    
    config1 = MCPServerConfig(command="dummy", enabled=True)
    config2 = MCPServerConfig(command="dummy", enabled=True)
    
    # Should be registered
    registry.register_server("allowed_server", config1)
    # Should be ignored (policy)
    registry.register_server("denied_server", config2)
    # Should be ignored (disabled)
    config_disabled = MCPServerConfig(command="dummy", enabled=False)
    registry.register_server("allowed_server2", config_disabled)
    
    assert "allowed_server" in registry._clients
    assert "denied_server" not in registry._clients
    assert "allowed_server2" not in registry._clients
    
    registry.shutdown()
