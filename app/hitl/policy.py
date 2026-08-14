"""HITL Policy for evaluating tool and plan executions."""

from __future__ import annotations

from typing import Any

from app.agent.contracts import AgentExecutionContext
from app.hitl.models import HITLPolicyDecision
from app.agent.plan import Plan


class HITLPolicy:
    """
    Evaluates whether a tool or plan requires human intervention.
    
    This can be configured with specific tool names or MCP servers that
    require approval.
    """

    def __init__(
        self,
        require_human_tools: set[str] | None = None,
        require_human_plans: bool = False,
        denied_tools: set[str] | None = None,
    ) -> None:
        self._require_human_tools = require_human_tools or set()
        self._require_human_plans = require_human_plans
        self._denied_tools = denied_tools or set()

    def evaluate_tool(
        self, tool: Any, arguments: dict[str, Any], context: AgentExecutionContext | None = None
    ) -> HITLPolicyDecision:
        """
        Evaluate if a tool execution requires human approval or is denied.
        """
        tool_name = tool.name
        
        # Consult MCP Policy if it's an MCP tool
        if hasattr(tool, "_policy") and hasattr(tool, "_client") and hasattr(tool, "_server_tool_name"):
            # It's an MCPTool
            try:
                from app.mcp.policy import PolicyDecision
                mcp_decision = tool._policy.evaluate_tool(tool._client.name, tool._server_tool_name)
                if mcp_decision == PolicyDecision.DENY:
                    return HITLPolicyDecision.DENY
                elif mcp_decision == PolicyDecision.REQUIRE_HUMAN:
                    return HITLPolicyDecision.REQUIRE_HUMAN
            except ImportError:
                pass
                
        if tool_name in self._denied_tools:
            return HITLPolicyDecision.DENY

        if tool_name in self._require_human_tools:
            return HITLPolicyDecision.REQUIRE_HUMAN

        return HITLPolicyDecision.ALLOW

    def evaluate_plan(self, plan: Plan, context: AgentExecutionContext | None = None) -> HITLPolicyDecision:
        """
        Evaluate if a generated plan requires human approval.
        """
        if self._require_human_plans:
            return HITLPolicyDecision.REQUIRE_HUMAN
            
        return HITLPolicyDecision.ALLOW

