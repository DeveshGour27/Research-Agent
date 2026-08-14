"""
MCP implementation of the BaseTool.
"""

from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool
from app.mcp.client import MCPClient
from app.mcp.policy import MCPPolicy, PolicyDecision
from app.mcp.contracts import MCPInvocationRequest, MCPToolMetadata
from app.mcp.errors import MCPPolicyDeniedError

class MCPTool(BaseTool):
    """
    Wraps an MCP tool as a standard Agent BaseTool.
    
    The AgentLoop interacts with this tool normally, entirely unaware of MCP.
    """

    def __init__(
        self, 
        client: MCPClient, 
        policy: MCPPolicy, 
        metadata: MCPToolMetadata
    ) -> None:
        self._client = client
        self._policy = policy
        self._metadata = metadata
        self._server_tool_name = metadata.name

        # Namespace the tool name to prevent collisions
        # e.g., mcp.math.add
        self.name = f"mcp.{self._client.name}.{metadata.name}"
        self.description = metadata.description
        self.input_schema = metadata.input_schema

    def execute(self, **kwargs: object) -> str:
        """Invoke the MCP tool through the client after policy validation."""
        
        # Policy enforcement MUST happen before execution
        if self._policy.evaluate_tool(self._client.name, self._server_tool_name) == PolicyDecision.DENY:
            raise MCPPolicyDeniedError(
                f"Policy denied execution of tool '{self.name}' on server '{self._client.name}'."
            )
            
        request = MCPInvocationRequest(
            server_name=self._client.name,
            tool_name=self._server_tool_name,
            arguments=dict(kwargs),
        )
        
        response = self._client.invoke_tool(request)
        
        if response.is_error:
            # We return it as a string. ToolRegistry wraps exceptions in ToolResult(is_error=True).
            # If the tool failed gracefully (like giving an error message the LLM can see),
            # we can just return it, but since MCP explicitly marks it as an error,
            # we should raise an exception so the AgentLoop handles it using its failure semantics.
            from app.exceptions import ToolExecutionError
            raise ToolExecutionError(
                f"MCP Tool returned error: {response.content}",
                tool_name=self.name
            )
            
        return response.content
