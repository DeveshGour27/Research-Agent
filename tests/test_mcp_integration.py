import sys
import pytest
from app.mcp.contracts import MCPServerConfig, MCPInvocationRequest
from app.mcp.policy import MCPPolicy
from app.mcp.client import MCPClient
from app.mcp.transport import StdioTransport
from app.mcp.registry import MCPRegistry

def test_mcp_real_stdio_integration():
    """Test connecting to a real Python MCP server subprocess."""
    config = MCPServerConfig(
        command=sys.executable,
        args=["tests/dummy_mcp_server.py"],
        enabled=True
    )
    
    policy = MCPPolicy()
    registry = MCPRegistry(policy=policy, transport_cls=StdioTransport)
    
    registry.register_server("dummy", config)
    registry.discover_all_tools()
    
    tools = registry.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "mcp.dummy.dummy_tool"
    
    # Execute the tool
    res = tools[0].execute(text="hello world")
    assert "Echo: hello world" in res
    
    registry.shutdown()
