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
    
    Phase 10 supports deterministic ALLOW/DENY decisions. Phase 11 may introduce
    REQUIRE_HUMAN. Policy evaluation MUST happen before touching the MCP Client.
    """

    def __init__(self, allowed_servers: list[str] | None = None) -> None:
        # Default to allowing everything for Phase 10 if not specified, 
        # or implement a strict allowlist.
        self._allowed_servers = allowed_servers
    
    def evaluate_server(self, server_name: str) -> PolicyDecision:
        """Evaluate if the server is allowed to connect."""
        if self._allowed_servers is not None and server_name not in self._allowed_servers:
            return PolicyDecision.DENY
        return PolicyDecision.ALLOW

    def evaluate_tool(self, server_name: str, tool_name: str) -> PolicyDecision:
        """Evaluate if the tool is allowed to be invoked."""
        # For Phase 10, if the server is allowed, its tools are allowed.
        # Future implementations can expand on tool-specific policies.
        server_policy = self.evaluate_server(server_name)
        if server_policy == PolicyDecision.DENY:
            return PolicyDecision.DENY
        elif server_policy == PolicyDecision.REQUIRE_HUMAN:
            return PolicyDecision.REQUIRE_HUMAN
        return PolicyDecision.ALLOW
