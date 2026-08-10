import pytest

from app.agent.communicator import AgentCommunicator
from app.agent.contracts import AgentRequest, AgentResult
from app.agent.retry import RetryBoundary, RetryPolicy
from app.exceptions import CommunicationError, ContractValidationError


class MockCommunicator(AgentCommunicator):
    def __init__(self, fails_with: Exception | None = None, fail_times: int = 0) -> None:
        self.fails_with = fails_with
        self.fail_times = fail_times
        self.calls = 0

    def send(self, request: AgentRequest) -> AgentResult:
        self.calls += 1
        if self.fail_times > 0 and self.fails_with:
            self.fail_times -= 1
            raise self.fails_with
        
        return AgentResult(
            request=request,
            state=None, # type: ignore
            output="success",
            success=True,
        )


def test_retry_boundary_success() -> None:
    communicator = MockCommunicator()
    boundary = RetryBoundary(RetryPolicy(), communicator)
    
    result = boundary.send(AgentRequest(input_text="test"))
    
    assert result.success is True
    assert communicator.calls == 1


def test_retry_boundary_retries_and_succeeds() -> None:
    communicator = MockCommunicator(fails_with=CommunicationError("temp error"), fail_times=2)
    boundary = RetryBoundary(RetryPolicy(max_attempts=3), communicator)
    
    result = boundary.send(AgentRequest(input_text="test"))
    
    assert result.success is True
    assert communicator.calls == 3


def test_retry_boundary_exhausts_retries() -> None:
    communicator = MockCommunicator(fails_with=CommunicationError("temp error"), fail_times=5)
    boundary = RetryBoundary(RetryPolicy(max_attempts=3), communicator)
    
    with pytest.raises(CommunicationError, match="temp error"):
        boundary.send(AgentRequest(input_text="test"))
    
    assert communicator.calls == 3


def test_retry_boundary_does_not_retry_unretryable_errors() -> None:
    communicator = MockCommunicator(fails_with=ContractValidationError("bad input"), fail_times=1)
    boundary = RetryBoundary(RetryPolicy(max_attempts=3), communicator)
    
    with pytest.raises(ContractValidationError, match="bad input"):
        boundary.send(AgentRequest(input_text="test"))
    
    assert communicator.calls == 1
