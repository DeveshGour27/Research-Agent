"""
Transport abstractions for the MCP client.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.client.stdio import stdio_client, StdioServerParameters

from app.mcp.contracts import MCPServerConfig
from app.mcp.errors import MCPTransportError

class MCPTransport(ABC):
    """Abstract base for MCP transports."""
    
    @asynccontextmanager
    @abstractmethod
    async def connect(self) -> AsyncIterator[tuple[Any, Any]]:
        """Yields (read_stream, write_stream) compatible with mcp ClientSession."""
        yield None, None


class StdioTransport(MCPTransport):
    """Transport that launches and communicates with a local MCP server via stdio."""
    
    def __init__(self, config: MCPServerConfig):
        self._config = config
        
    @asynccontextmanager
    async def connect(self) -> AsyncIterator[tuple[Any, Any]]:
        server_params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            env=self._config.env,
        )
        try:
            async with stdio_client(server_params) as (read, write):
                yield read, write
        except Exception as e:
            raise MCPTransportError(f"Failed to connect to stdio server '{self._config.command}': {e}") from e

class MockMCPTransport(MCPTransport):
    """Mock transport for testing, returns dummy streams."""
    
    def __init__(self, config: MCPServerConfig):
        self._config = config
        
    @asynccontextmanager
    async def connect(self) -> AsyncIterator[tuple[Any, Any]]:
        # In tests, we might mock this at the ClientSession level instead of streams.
        # But we provide this class as a placeholder.
        yield object(), object()
