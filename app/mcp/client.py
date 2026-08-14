"""
Synchronous client for communicating with an MCP server via async SDK.
"""

from __future__ import annotations

import asyncio
import threading
import concurrent.futures
from typing import Any, Type

from mcp.client.session import ClientSession

from app.mcp.contracts import (
    MCPServerConfig,
    MCPToolMetadata,
    MCPInvocationRequest,
    MCPInvocationResponse,
)
from app.mcp.transport import MCPTransport
from app.mcp.errors import (
    MCPConnectionError,
    MCPInitializationError,
    MCPDiscoveryError,
    MCPInvalidArgumentsError,
    MCPServerError,
    MCPTransportError,
    MCPTimeoutError,
    MCPInvalidResponseError,
)

class MCPClient:
    """
    Manages a connection to a single MCP server.
    
    This client provides a synchronous interface to the async MCP SDK by running
    a dedicated background event loop thread per client and keeping context managers alive.
    """

    def __init__(self, name: str, config: MCPServerConfig, transport_cls: Type[MCPTransport]) -> None:
        self.name = name
        self.config = config
        self._transport_cls = transport_cls
        
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name=f"MCPClient-{self.name}", daemon=True)
        self._session: ClientSession | None = None
        self._connected = False
        
        self._ready_event = threading.Event()
        self._connect_error: Exception | None = None
        self._disconnect_event: asyncio.Event | None = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main_task())
        except Exception:
            pass

    async def _main_task(self) -> None:
        try:
            transport = self._transport_cls(self.config)
            async with transport.connect() as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session
                    self._disconnect_event = asyncio.Event()
                    self._ready_event.set()
                    await self._disconnect_event.wait()
        except Exception as e:
            self._connect_error = e
            self._ready_event.set()

    def _run_sync(self, coro: Any, timeout: float | None = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            raise MCPTimeoutError(f"Operation timed out after {timeout} seconds.") from e
        except Exception as e:
            raise e

    def connect(self) -> None:
        """Initialize the transport and perform the MCP handshake."""
        if self._connected:
            return
            
        self._thread.start()
        
        if not self._ready_event.wait(timeout=self.config.timeout_seconds):
            raise MCPTimeoutError(f"Connection to server '{self.name}' timed out.")
            
        if self._connect_error:
            raise MCPInitializationError(f"Failed to initialize server '{self.name}': {self._connect_error}") from self._connect_error
            
        self._connected = True

    def discover_tools(self) -> list[MCPToolMetadata]:
        """Request tools/list from the server."""
        if not self._connected:
            self.connect()
            
        try:
            return self._run_sync(self._async_discover_tools(), timeout=self.config.timeout_seconds)
        except Exception as e:
            raise MCPDiscoveryError(f"Failed to discover tools for server '{self.name}': {e}") from e

    async def _async_discover_tools(self) -> list[MCPToolMetadata]:
        assert self._session is not None
        result = await self._session.list_tools()
        tools = []
        for t in result.tools:
            # Ensure name and description exist
            if not t.name:
                continue
            
            schema = getattr(t, "inputSchema", getattr(t, "input_schema", {})) or {}
            
            tools.append(MCPToolMetadata(
                name=t.name,
                description=t.description or f"Tool {t.name}",
                input_schema=schema
            ))
        return tools

    def invoke_tool(self, request: MCPInvocationRequest) -> MCPInvocationResponse:
        """Call a tool on the MCP server."""
        if not self._connected:
            self.connect()
            
        try:
            return self._run_sync(self._async_invoke_tool(request), timeout=self.config.timeout_seconds)
        except MCPTimeoutError:
            raise
        except Exception as e:
            error_str = str(e)
            if "invalid arguments" in error_str.lower():
                raise MCPInvalidArgumentsError(f"Invalid arguments for '{request.tool_name}': {e}") from e
            raise MCPServerError(f"Server error during tool invocation: {e}") from e

    async def _async_invoke_tool(self, request: MCPInvocationRequest) -> MCPInvocationResponse:
        assert self._session is not None
        result = await self._session.call_tool(request.tool_name, arguments=request.arguments)
        
        is_error = getattr(result, "is_error", getattr(result, "isError", False))
        if is_error:
            content = "\\n".join([c.text for c in result.content if getattr(c, 'type', '') == 'text'])
            return MCPInvocationResponse(content=content, is_error=True)
            
        content = "\\n".join([c.text for c in result.content if getattr(c, 'type', '') == 'text'])
        return MCPInvocationResponse(content=content, is_error=False)

    def disconnect(self) -> None:
        """Close the session and transport."""
        if not self._connected:
            return
            
        try:
            self._run_sync(self._async_disconnect(), timeout=5.0)
        except Exception:
            pass
        finally:
            self._connected = False
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)

    async def _async_disconnect(self) -> None:
        if self._disconnect_event:
            self._disconnect_event.set()
