"""
MCP-specific custom exceptions that map into the project's failure taxonomy.
"""

from __future__ import annotations

from app.exceptions import (
    AgentError,
    RetryableError,
    FatalError,
    AgentTimeoutError,
)

class MCPError(AgentError):
    """Base class for all MCP integration errors."""

class MCPConnectionError(RetryableError):
    """Raised when an MCP client fails to connect to the server."""

class MCPInitializationError(FatalError):
    """Raised when the MCP server handshake/initialization fails."""

class MCPDiscoveryError(FatalError):
    """Raised when tool discovery fails or returns invalid metadata."""

class MCPInvalidToolError(FatalError):
    """Raised when an invalid tool is requested."""

class MCPInvalidArgumentsError(FatalError):
    """Raised when a tool is called with invalid arguments."""

class MCPInvalidResponseError(RetryableError):
    """Raised when the MCP server returns a malformed response."""

class MCPServerError(FatalError):
    """Raised when the MCP server explicitly reports an internal error."""

class MCPTransportError(RetryableError):
    """Raised for underlying transport failures (e.g., pipe broken)."""

class MCPPolicyDeniedError(FatalError):
    """Raised when an MCP invocation is denied by policy."""

class MCPTimeoutError(AgentTimeoutError):
    """Raised when an MCP operation times out."""
