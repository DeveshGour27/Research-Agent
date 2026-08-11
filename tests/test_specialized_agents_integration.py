from unittest.mock import MagicMock
from app.agent.contracts import AgentRequest
from app.agent.registry import AgentRegistry
from app.agent.routing import CapabilityRouter
from app.agent.specialized.rag_agent import RAGAgent
from app.agent.specialized.web_agent import WebResearchAgent


def test_registry_integration() -> None:
    registry = AgentRegistry()
    web_agent = WebResearchAgent()
    rag_agent = RAGAgent(retriever=MagicMock())
    
    registry.register(web_agent)
    registry.register(rag_agent)
    
    assert registry.has("web_research_agent")
    assert registry.has("rag_agent")
    
    assert registry.get("web_research_agent") is web_agent
    assert registry.get("rag_agent") is rag_agent

def test_capability_router_integration() -> None:
    router = CapabilityRouter()
    agents = [WebResearchAgent(), RAGAgent(retriever=MagicMock())]
    
    # Test routing to WebResearchAgent
    web_request = AgentRequest(
        input_text="Search for X",
        metadata={"required_task_type": "web_search"}
    )
    selected = router.select_agent(web_request, agents)
    assert selected is not None
    assert selected.identity.name == "web_research_agent"
    
    # Test routing to RAGAgent
    rag_request = AgentRequest(
        input_text="Find internal docs",
        metadata={"required_task_type": "rag_search"}
    )
    selected = router.select_agent(rag_request, agents)
    assert selected is not None
    assert selected.identity.name == "rag_agent"
    
    # Test unknown capability
    unknown_request = AgentRequest(
        input_text="Do magic",
        metadata={"required_task_type": "unknown_task"}
    )
    selected = router.select_agent(unknown_request, agents)
    assert selected is None
