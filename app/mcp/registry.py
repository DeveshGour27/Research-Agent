"""
Registry for MCP servers and tools.
"""

from __future__ import annotations

from typing import Type

from app.mcp.contracts import MCPServerConfig
from app.mcp.client import MCPClient
from app.mcp.policy import MCPPolicy, PolicyDecision
from app.mcp.transport import MCPTransport, StdioTransport
from app.mcp.tool import MCPTool
from app.logger import get_logger

logger = get_logger(__name__)

class MCPRegistry:
    """
    Coordinates multiple MCPClient instances, discovers tools, and builds MCPTools.
    """

    def __init__(self, policy: MCPPolicy, transport_cls: Type[MCPTransport] = StdioTransport) -> None:
        self._policy = policy
        self._transport_cls = transport_cls
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, MCPTool] = {}

    def register_server(self, name: str, config: MCPServerConfig) -> None:
        """Register and initialize an MCP server."""
        if not config.enabled:
            logger.info("MCP server is disabled", extra={"server": name})
            return
            
        if self._policy.evaluate_server(name) == PolicyDecision.DENY:
            logger.warning("MCP server denied by policy", extra={"server": name})
            return
            
        if name in self._clients:
            logger.warning("MCP server already registered", extra={"server": name})
            return
            
        client = MCPClient(name=name, config=config, transport_cls=self._transport_cls)
        self._clients[name] = client
        logger.info("MCP server registered", extra={"server": name})
        
    def discover_all_tools(self) -> None:
        """Connect to all registered servers and discover tools."""
        for name, client in self._clients.items():
            try:
                metadata_list = client.discover_tools()
                for meta in metadata_list:
                    # Apply policy to tool
                    if self._policy.evaluate_tool(name, meta.name) == PolicyDecision.DENY:
                        logger.debug(
                            "MCP tool denied by policy", 
                            extra={"server": name, "tool": meta.name}
                        )
                        continue
                        
                    mcp_tool = MCPTool(client, self._policy, meta)
                    
                    if mcp_tool.name in self._tools:
                        logger.warning(
                            "MCP tool collision, skipping", 
                            extra={"server": name, "tool": mcp_tool.name}
                        )
                        continue
                        
                    self._tools[mcp_tool.name] = mcp_tool
                    logger.debug(
                        "MCP tool discovered and registered", 
                        extra={"server": name, "tool": mcp_tool.name}
                    )
            except Exception as e:
                logger.error(
                    "Failed to discover tools for MCP server", 
                    extra={"server": name, "error": str(e)}
                )

    def get_tools(self) -> list[MCPTool]:
        """Return all discovered MCP tools."""
        return list(self._tools.values())
        
    def shutdown(self) -> None:
        """Cleanly disconnect all clients."""
        for name, client in self._clients.items():
            try:
                client.disconnect()
            except Exception as e:
                logger.warning(
                    "Error disconnecting MCP client", 
                    extra={"server": name, "error": str(e)}
                )
        self._clients.clear()
        self._tools.clear()

_mcp_registry_instance: MCPRegistry | None = None

def get_mcp_registry() -> MCPRegistry:
    """Return the global MCPRegistry singleton, initializing it from config if needed."""
    global _mcp_registry_instance
    if _mcp_registry_instance is None:
        from app.config import settings
        import json
        
        policy = MCPPolicy(allowed_servers=settings.mcp_allowed_servers)
        _mcp_registry_instance = MCPRegistry(policy=policy)
        
        # Parse MCP server configuration
        try:
            mcp_servers_config = json.loads(settings.mcp_servers)
            for name, config_dict in mcp_servers_config.items():
                server_config = MCPServerConfig(**config_dict)
                _mcp_registry_instance.register_server(name, server_config)
                
            _mcp_registry_instance.discover_all_tools()
        except Exception as e:
            logger.error("Failed to initialize global MCPRegistry", extra={"error": str(e)})
            
    return _mcp_registry_instance

