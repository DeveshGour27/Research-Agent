import pytest

from app.agent.communicator import InProcessCommunicator
from app.agent.contracts import AgentCapabilities, AgentExecutionError, AgentIdentity, AgentRequest, AgentResult, BaseAgent
from app.agent.registry import AgentRegistry
from app.agent.state import AgentState
from app.agent.supervisor import Supervisor
from app.agent.execution_context import AgentExecutionContext


class MockAgent(BaseAgent):
    def __init__(self, name: str, should_fail: bool = False, output: str = "mock output") -> None:
        self._identity = AgentIdentity(name=name)
        self._capabilities = AgentCapabilities()
        self.should_fail = should_fail
        self.output = output
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
            raise AgentExecutionError(f"Failed as requested: {self.identity.name}")
            
        if request.context is not None:
            request.context.publish_agent_output(
                agent_id=self.identity.name,
                output=self.output,
                success=True,
            )
            
        return AgentResult(
            request=request,
            state=AgentState(),
            output=self.output,
            success=True,
        )


def test_supervisor_uses_registry_and_communicator() -> None:
    registry = AgentRegistry()
    agent1 = MockAgent("agent1", should_fail=True)
    agent2 = MockAgent("agent2", output="agent2 success")
    registry.register(agent1)
    registry.register(agent2)
    
    communicator = InProcessCommunicator(registry)
    
    supervisor = Supervisor(registry=registry, communicator=communicator)
    
    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="test", context=context)
    
    result = supervisor.execute(request)
    
    assert result.success is True
    assert result.output == "agent2 success"
    assert agent1.calls == 1
    assert agent2.calls == 1
    
    # Check correlation ID tracking
    output1 = context.get_agent_output("agent1")
    assert output1 is not None and output1.success is False
    
    output2 = context.get_agent_output("agent2")
    assert output2 is not None and output2.success is True


def test_supervisor_raises_value_error_if_no_agents_or_registry() -> None:
    with pytest.raises(ValueError, match="Either agents or registry must be provided"):
        Supervisor()


def test_supervisor_legacy_instantiation_without_di_works() -> None:
    agent1 = MockAgent("agent1", should_fail=True)
    agent2 = MockAgent("agent2", output="agent2 success")
    
    supervisor = Supervisor(agents=[agent1, agent2])
    
    request = AgentRequest(input_text="test")
    result = supervisor.execute(request)
    
    assert result.success is True
    assert result.output == "agent2 success"
    assert agent1.calls == 1
    assert agent2.calls == 1
