"""
Security policy for MCP integration.
"""

from __future__ import annotations

from enum import Enum

class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HUMAN = "require_human"

class MCPPolicy:
    """
    Evaluates policy decisions for MCP operations.
    
    Defaults to strict DENY unless the server is explicitly present in allowed_servers,
    or allow_all is explicitly enabled (e.g. for isolated test suites).
    """

    def __init__(
        self,
        allowed_servers: list[str] | set[str] | None = None,
        allow_all: bool = False,
    ) -> None:
        self._allowed_servers = set(allowed_servers) if allowed_servers is not None else set()
        self._allow_all = allow_all
    
    def evaluate_server(self, server_name: str) -> PolicyDecision:
        """Evaluate if the server is allowed to connect."""
        if self._allow_all or server_name in self._allowed_servers:
            return PolicyDecision.ALLOW
        return PolicyDecision.DENY

    def evaluate_tool(self, server_name: str, tool_name: str) -> PolicyDecision:
        """Evaluate if the tool is allowed to be invoked."""
        server_policy = self.evaluate_server(server_name)
        if server_policy == PolicyDecision.DENY:
            return PolicyDecision.DENY
        elif server_policy == PolicyDecision.REQUIRE_HUMAN:
            return PolicyDecision.REQUIRE_HUMAN
        return PolicyDecision.ALLOW
