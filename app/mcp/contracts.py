"""
Strict typed contracts for the MCP boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, Field

class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    enabled: bool = Field(default=True)

@dataclass(frozen=True)
class MCPToolMetadata:
    """Normalized metadata for a discovered MCP tool."""
    name: str
    description: str
    input_schema: dict[str, Any]

@dataclass(frozen=True)
class MCPInvocationRequest:
    """Request to invoke an MCP tool."""
    server_name: str
    tool_name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class MCPInvocationResponse:
    """Normalized response from an MCP tool invocation."""
    content: str
    is_error: bool = False
