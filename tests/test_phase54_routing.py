from app.agent.contracts import AgentCapabilities, AgentIdentity, AgentRequest, BaseAgent, AgentResult
from app.agent.routing import CapabilityRouter


class MockAgent(BaseAgent):
    def __init__(self, name: str, task_types: frozenset[str] = frozenset()) -> None:
        self._identity = AgentIdentity(name=name)
        self._capabilities = AgentCapabilities(task_types=task_types)

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def execute(self, request: AgentRequest) -> AgentResult:
        raise NotImplementedError()


def test_capability_router_strict_task_type_filtering() -> None:
    router = CapabilityRouter()
    
    agent_research = MockAgent("researcher", task_types=frozenset(["research"]))
    agent_analysis = MockAgent("analyst", task_types=frozenset(["analysis"]))
    agent_general = MockAgent("general", task_types=frozenset(["research", "analysis", "coding"]))
    
    agents = [agent_research, agent_analysis, agent_general]
    
    # Request specifying analysis
    request_analysis = AgentRequest(input_text="do analysis", metadata={"required_task_type": "analysis"})
    selected = router.select_agent(request_analysis, agents)
    
    # Should not be researcher. Should be analyst or general.
    assert selected is not None
    assert selected.identity.name in ("analyst", "general")

    # Request specifying coding
    request_coding = AgentRequest(input_text="do coding", metadata={"required_task_type": "coding"})
    selected_coding = router.select_agent(request_coding, agents)
    
    assert selected_coding is not None
    assert selected_coding.identity.name == "general"

    # Request specifying non-existent task type
    request_unknown = AgentRequest(input_text="do magic", metadata={"required_task_type": "magic"})
    selected_unknown = router.select_agent(request_unknown, agents)
    
    assert selected_unknown is None
