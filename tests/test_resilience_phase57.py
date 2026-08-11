import pytest
from app.agent.contracts import AgentRequest
from app.agent.execution_context import AgentExecutionContext, ExecutionStatus
from app.agent.retry import RetryPolicy, RetryBoundary
from app.exceptions import (
    CommunicationError,
    RetryableError,
    FatalError,
    AgentCancellationError,
    AgentTimeoutError,
    RecoverableError
)
from app.agent.plan import Plan, PlanStep, PlanStatus, PlanStepStatus
from app.agent.executor import PlanExecutor
from app.agent.supervisor import Supervisor
from app.agent.collaboration import CollaborationSession
from app.agent.routing import CapabilityRouter
from app.agent.registry import AgentRegistry

class DummyCommunicator:
    def __init__(self, fails=0, error_type=CommunicationError):
        self.fails = fails
        self.error_type = error_type
        self.attempts = 0
    def send(self, request):
        self.attempts += 1
        if self.attempts <= self.fails:
            raise self.error_type("Failed")
        from app.agent.contracts import AgentResult
        from app.agent.state import AgentState
        return AgentResult(request=request, state=AgentState(), output="Success", success=True)

def test_retry_boundary_success_after_retries():
    comm = DummyCommunicator(fails=2)
    policy = RetryPolicy(max_attempts=3)
    boundary = RetryBoundary(policy, comm)
    req = AgentRequest(input_text="test")
    res = boundary.send(req)
    assert res.success
    assert comm.attempts == 3

def test_retry_boundary_budget_exhausted():
    comm = DummyCommunicator(fails=4)
    policy = RetryPolicy(max_attempts=3)
    boundary = RetryBoundary(policy, comm)
    req = AgentRequest(input_text="test")
    with pytest.raises(CommunicationError):
        boundary.send(req)
    assert comm.attempts == 3

def test_retry_boundary_max_repeated_failures():
    comm = DummyCommunicator(fails=6)
    policy = RetryPolicy(max_attempts=10, max_repeated_failures=5)
    boundary = RetryBoundary(policy, comm)
    req = AgentRequest(input_text="test")
    with pytest.raises(FatalError, match="Max repeated failures budget exceeded"):
        boundary.send(req)
    assert comm.attempts == 5

def test_retry_boundary_fatal_error():
    comm = DummyCommunicator(fails=1, error_type=FatalError)
    policy = RetryPolicy(max_attempts=3)
    boundary = RetryBoundary(policy, comm)
    req = AgentRequest(input_text="test")
    with pytest.raises(FatalError):
        boundary.send(req)
    assert comm.attempts == 1

def test_execution_context_cancellation():
    ctx = AgentExecutionContext(task="test")
    ctx.mark_cancelled()
    assert ctx.is_cancelled
    assert ctx.status == ExecutionStatus.CANCELLED

def test_execution_context_timeout():
    ctx = AgentExecutionContext(task="test")
    ctx.mark_timed_out()
    assert ctx.is_timed_out
    assert ctx.status == ExecutionStatus.TIMED_OUT

def test_executor_cancellation_propagation():
    ctx = AgentExecutionContext(task="test")
    ctx.mark_cancelled()
    plan = Plan()
    plan.steps["1"] = PlanStep(step_id="1", description="s1")
    
    comm = DummyCommunicator()
    executor = PlanExecutor(CapabilityRouter(), AgentRegistry(), comm)
    from app.agent.execution_policy import ExecutionPolicy
    
    with pytest.raises(AgentCancellationError):
        executor.execute(plan, ctx, ExecutionPolicy())

def test_executor_timeout_propagation():
    ctx = AgentExecutionContext(task="test")
    ctx.mark_timed_out()
    plan = Plan()
    plan.steps["1"] = PlanStep(step_id="1", description="s1")
    
    comm = DummyCommunicator()
    executor = PlanExecutor(CapabilityRouter(), AgentRegistry(), comm)
    from app.agent.execution_policy import ExecutionPolicy
    
    with pytest.raises(AgentTimeoutError):
        executor.execute(plan, ctx, ExecutionPolicy())

def test_partial_success_logic():
    ctx = AgentExecutionContext(task="test")
    plan = Plan()
    # Step 1: required, will succeed
    plan.steps["1"] = PlanStep(step_id="1", description="s1", is_required=True, status=PlanStepStatus.COMPLETED)
    # Step 2: optional, will fail
    plan.steps["2"] = PlanStep(step_id="2", description="s2", is_required=False, status=PlanStepStatus.FAILED)
    
    comm = DummyCommunicator()
    executor = PlanExecutor(CapabilityRouter(), AgentRegistry(), comm)
    
    # We can just manually call the logic at the end of execute if we mock the execution
    # or just let execute run with no ready steps
    from app.agent.execution_policy import ExecutionPolicy
    result_plan = executor.execute(plan, ctx, ExecutionPolicy())
    assert result_plan.status == PlanStatus.PARTIAL_SUCCESS

def test_partial_success_logic_required_fails():
    ctx = AgentExecutionContext(task="test")
    plan = Plan()
    # Step 1: required, will fail
    plan.steps["1"] = PlanStep(step_id="1", description="s1", is_required=True, status=PlanStepStatus.FAILED)
    # Step 2: optional, will succeed
    plan.steps["2"] = PlanStep(step_id="2", description="s2", is_required=False, status=PlanStepStatus.COMPLETED)
    
    comm = DummyCommunicator()
    executor = PlanExecutor(CapabilityRouter(), AgentRegistry(), comm)
    from app.agent.execution_policy import ExecutionPolicy
    result_plan = executor.execute(plan, ctx, ExecutionPolicy())
    assert result_plan.status == PlanStatus.FAILED

