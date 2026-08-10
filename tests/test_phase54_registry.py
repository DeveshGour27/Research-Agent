import pytest

from app.agent.contracts import AgentCapabilities, AgentIdentity, AgentRequest, AgentResult, BaseAgent
from app.agent.registry import AgentRegistry
from app.agent.state import AgentState
from app.exceptions import AgentNotFoundError


class MockAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        self._identity = AgentIdentity(name=name)
        self._capabilities = AgentCapabilities()

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def execute(self, request: AgentRequest) -> AgentResult:
        return AgentResult(
            request=request,
            state=AgentState(),
            output="mock output",
            success=True,
        )


def test_registry_registration() -> None:
    registry = AgentRegistry()
    agent = MockAgent("agent1")
    
    registry.register(agent)
    
    assert registry.has("agent1")
    assert registry.get("agent1") is agent
    assert registry.get_all() == [agent]
    assert registry.agent_ids() == ["agent1"]


def test_registry_duplicate_registration_raises() -> None:
    registry = AgentRegistry()
    agent1 = MockAgent("agent1")
    agent2 = MockAgent("agent1")
    
    registry.register(agent1)
    
    with pytest.raises(ValueError, match="already registered"):
        registry.register(agent2)


def test_registry_get_missing_raises() -> None:
    registry = AgentRegistry()
    
    with pytest.raises(AgentNotFoundError, match="Agent 'missing' not found in registry"):
        registry.get("missing")


def test_registry_empty_name_raises() -> None:
    registry = AgentRegistry()
    agent = MockAgent("")
    
    with pytest.raises(ValueError, match="identity name must not be empty"):
        registry.register(agent)
