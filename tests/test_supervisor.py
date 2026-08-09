from __future__ import annotations

from app.agent import AgentExecutionContext, AgentRequest, Supervisor
from app.agent.contracts import AgentCapabilities, AgentExecutionError, AgentIdentity, AgentResult, BaseAgent
from app.agent.state import AgentState


class FakeAgent(BaseAgent):
    def __init__(self, name: str, output: str | None, *, success: bool = True) -> None:
        self._identity = AgentIdentity(name=name, description=f"Fake agent {name}")
        self._capabilities = AgentCapabilities(tool_use=False, memory=False, multi_turn=False, retrieval=False)
        self._output = output
        self._success = success
        self.calls = 0

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def execute(self, request: AgentRequest) -> AgentResult:
        self.calls += 1

        if not self._success:
            raise AgentExecutionError(
                "Child agent failed.",
                request=request,
                details={"agent": self.identity.name},
            )

        state = AgentState()
        state.finished = True
        state.final_answer = self._output

        if request.context is not None:
            request.context.publish_agent_output(
                agent_id=self.identity.name,
                output=self._output,
                success=True,
                metadata={"agent": self.identity.name},
            )

        return AgentResult(
            request=request,
            state=state,
            output=self._output,
            success=True,
            context=request.context,
            metadata={"agent": self.identity.name},
        )


def test_supervisor_tries_next_agent_when_child_fails() -> None:
    first = FakeAgent("researcher", None, success=False)
    second = FakeAgent("analyst", "Analysis complete.", success=True)
    supervisor = Supervisor([first, second])

    context = AgentExecutionContext(task="Research and analyze")
    result = supervisor.execute(AgentRequest(input_text="Research and analyze.", context=context))

    assert result.success is True
    assert result.output == "Analysis complete."
    assert first.calls == 1
    assert second.calls == 1
    assert context.get_agent_output(first.identity.name).success is False
    assert context.get_agent_output(second.identity.name).output == "Analysis complete."
    assert context.get_agent_output(supervisor.identity.name).output == "Analysis complete."


def test_supervisor_raises_when_all_children_fail() -> None:
    first = FakeAgent("researcher", None, success=False)
    second = FakeAgent("analyst", None, success=False)
    supervisor = Supervisor([first, second])

    try:
        supervisor.execute(AgentRequest(input_text="Try again later."))
    except AgentExecutionError as error:
        assert error.request is not None
        assert error.request.input_text == "Try again later."
    else:
        raise AssertionError("Expected AgentExecutionError")
