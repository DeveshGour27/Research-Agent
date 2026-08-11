"""Tests for Phase 5.6 advanced multi-agent collaboration."""

import pytest

from app.agent.collaboration import (
    Artifact,
    CollaborationPolicy,
    CollaborationSession,
    Handoff,
)
from app.agent.communicator import InProcessCommunicator
from app.agent.contracts import (
    AgentCapabilities,
    AgentExecutionError,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
    CollaborationRequest,
)
from app.agent.execution_context import AgentExecutionContext
from app.agent.registry import AgentRegistry
from app.agent.retry import RetryPolicy
from app.agent.routing import CapabilityRouter
from app.agent.state import AgentState


class MockCollaboratingAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        task_types: frozenset[str],
        output: str = "done",
        handoff_to: str | None = None,
        handoff_message: str | None = None,
    ) -> None:
        self._identity = AgentIdentity(name=name)
        self._capabilities = AgentCapabilities(task_types=task_types)
        self.output = output
        self.handoff_to = handoff_to
        self.handoff_message = handoff_message
        self.calls = 0

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def execute(self, request: AgentRequest) -> AgentResult:
        self.calls += 1
        
        collaboration_request = None
        if self.handoff_to and self.calls == 1:
            # First call requests handoff, second call finishes
            collaboration_request = CollaborationRequest(
                requested_task_type=self.handoff_to,
                message=self.handoff_message or "Please help",
            )
            
        return AgentResult(
            request=request,
            state=AgentState(),
            output=self.output,
            success=True,
            collaboration_request=collaboration_request,
        )


def _setup_session(policy: CollaborationPolicy | None = None) -> tuple[CollaborationSession, AgentRegistry]:
    registry = AgentRegistry()
    researcher = MockCollaboratingAgent(
        "researcher",
        task_types=frozenset(["research"]),
        handoff_to="analysis",
    )
    analyst = MockCollaboratingAgent(
        "analyst",
        task_types=frozenset(["analysis"]),
    )
    registry.register(researcher)
    registry.register(analyst)

    router = CapabilityRouter()
    communicator = InProcessCommunicator(registry)
    context = AgentExecutionContext(task="test")
    
    request = AgentRequest(
        input_text="Do research",
        metadata={"required_task_type": "research"},
        context=context,
    )
    
    session = CollaborationSession(
        step_id="s1",
        initial_request=request,
        router=router,
        registry=registry,
        communicator=communicator,
        retry_policy=RetryPolicy(),
        policy=policy,
    )
    
    return session, registry


def test_collaboration_session_successful_handoff() -> None:
    session, registry = _setup_session()
    result = session.execute()
    
    assert result.success is True
    
    researcher = registry.get("researcher")
    analyst = registry.get("analyst")
    assert researcher
    assert analyst
    assert researcher.calls == 1  # Handed off
    assert analyst.calls == 1     # Took over and finished
    assert "researcher" in session.participating_agents
    assert "analyst" in session.participating_agents


def test_collaboration_session_max_messages_exceeded() -> None:
    registry = AgentRegistry()
    ping = MockCollaboratingAgent("ping", frozenset(["ping"]), handoff_to="pong")
    pong = MockCollaboratingAgent("pong", frozenset(["pong"]), handoff_to="ping")
    # Make them continually handoff to each other
    ping.handoff_to = "pong"
    pong.handoff_to = "ping"
    # Overwrite execute to always handoff
    def always_handoff(agent: MockCollaboratingAgent, req: AgentRequest) -> AgentResult:
        agent.calls += 1
        return AgentResult(
            request=req,
            state=AgentState(),
            output="handoff",
            success=True,
            collaboration_request=CollaborationRequest(agent.handoff_to, "help")
        )
    ping.execute = lambda req: always_handoff(ping, req)
    pong.execute = lambda req: always_handoff(pong, req)
    
    registry.register(ping)
    registry.register(pong)

    router = CapabilityRouter()
    communicator = InProcessCommunicator(registry)
    context = AgentExecutionContext(task="test")
    request = AgentRequest(
        input_text="start",
        metadata={"required_task_type": "ping"},
        context=context,
    )
    
    # Restrict messages
    policy = CollaborationPolicy(max_messages_per_session=3)
    session = CollaborationSession(
        "s1", request, router, registry, communicator, RetryPolicy(), policy
    )
    
    with pytest.raises(AgentExecutionError, match="Max messages"):
        session.execute()


def test_collaboration_session_max_agents_exceeded() -> None:
    registry = AgentRegistry()
    a1 = MockCollaboratingAgent("a1", frozenset(["t1"]))
    a2 = MockCollaboratingAgent("a2", frozenset(["t2"]))
    a3 = MockCollaboratingAgent("a3", frozenset(["t3"]))
    
    # a1 -> a2 -> a3 -> a1
    def chain_handoff(agent: MockCollaboratingAgent, target: str, req: AgentRequest) -> AgentResult:
        return AgentResult(
            request=req,
            state=AgentState(),
            output="handoff",
            success=True,
            collaboration_request=CollaborationRequest(target, "help")
        )
    a1.execute = lambda req: chain_handoff(a1, "t2", req)
    a2.execute = lambda req: chain_handoff(a2, "t3", req)
    a3.execute = lambda req: chain_handoff(a3, "t1", req)
    
    registry.register(a1)
    registry.register(a2)
    registry.register(a3)

    policy = CollaborationPolicy(max_participating_agents=2)
    session = CollaborationSession(
        "s1",
        AgentRequest("start", metadata={"required_task_type": "t1"}),
        CapabilityRouter(),
        registry,
        InProcessCommunicator(registry),
        RetryPolicy(),
        policy,
    )
    
    with pytest.raises(AgentExecutionError, match="Max participating agents"):
        session.execute()


def test_artifact_immutability_enforced() -> None:
    context = AgentExecutionContext(task="test")
    art1 = Artifact("res_v1", "markdown", "researcher", 1, "corr1", "content")
    
    context.publish_artifact(art1)
    assert context.get_artifact("res_v1") == art1
    
    art2 = Artifact("res_v1", "markdown", "analyst", 2, "corr2", "different content")
    with pytest.raises(ValueError, match="Artifact ID 'res_v1' already exists. Artifacts are immutable."):
        context.publish_artifact(art2)
        
    with pytest.raises(ValueError, match="Artifact key 'res_v1' already exists. Artifacts are immutable."):
        context.set_artifact("res_v1", "some regular object")
