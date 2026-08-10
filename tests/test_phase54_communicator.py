import pytest

from app.agent.communicator import InProcessCommunicator
from app.agent.contracts import AgentCapabilities, AgentExecutionError, AgentIdentity, AgentRequest, AgentResult, BaseAgent
from app.agent.registry import AgentRegistry
from app.agent.state import AgentState
from app.exceptions import AgentNotFoundError, ContractValidationError


class MockAgent(BaseAgent):
    def __init__(self, name: str, should_fail: bool = False) -> None:
        self._identity = AgentIdentity(name=name)
        self._capabilities = AgentCapabilities()
        self.should_fail = should_fail
        self.calls = 0

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def execute(self, request: AgentRequest) -> AgentResult:
        self.calls += 1
        if self.should_fail:
            raise AgentExecutionError("Failed as requested")
        return AgentResult(
            request=request,
            state=AgentState(),
            output="mock output",
            success=True,
        )


def test_in_process_communicator_success() -> None:
    registry = AgentRegistry()
    agent = MockAgent("agent1")
    registry.register(agent)
    
    communicator = InProcessCommunicator(registry)
    
    request = AgentRequest(input_text="test", metadata={"selected_agent": "agent1"})
    result = communicator.send(request)
    
    assert result.success is True
    assert result.output == "mock output"
    assert agent.calls == 1


def test_in_process_communicator_missing_selected_agent_raises() -> None:
    registry = AgentRegistry()
    communicator = InProcessCommunicator(registry)
    
    request = AgentRequest(input_text="test", metadata={})
    
    with pytest.raises(ContractValidationError, match="must contain 'selected_agent'"):
        communicator.send(request)


def test_in_process_communicator_missing_agent_raises() -> None:
    registry = AgentRegistry()
    communicator = InProcessCommunicator(registry)
    
    request = AgentRequest(input_text="test", metadata={"selected_agent": "missing"})
    
    with pytest.raises(AgentNotFoundError, match="Agent 'missing' not found"):
        communicator.send(request)


def test_in_process_communicator_agent_error_propagates() -> None:
    registry = AgentRegistry()
    agent = MockAgent("agent1", should_fail=True)
    registry.register(agent)
    
    communicator = InProcessCommunicator(registry)
    
    request = AgentRequest(input_text="test", metadata={"selected_agent": "agent1"})
    
    with pytest.raises(AgentExecutionError, match="Failed as requested"):
        communicator.send(request)
    
    assert agent.calls == 1
