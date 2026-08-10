"""Tests for Phase 5.5 executor, replanning, and supervisor integration."""

import pytest

from app.agent.communicator import InProcessCommunicator
from app.agent.contracts import (
    AgentCapabilities,
    AgentExecutionError,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
)
from app.agent.execution_context import AgentExecutionContext
from app.agent.execution_policy import ExecutionPolicy
from app.agent.executor import PlanExecutor
from app.agent.plan import Plan, PlanStatus, PlanStep, PlanStepStatus, StepResult
from app.agent.planner import Planner
from app.agent.registry import AgentRegistry
from app.agent.replanning import ReplanningPolicy
from app.agent.retry import RetryPolicy
from app.agent.routing import CapabilityRouter
from app.agent.state import AgentState
from app.agent.supervisor import Supervisor


# ------------------------------------------------------------------ #
# Test fixtures
# ------------------------------------------------------------------ #

class MockAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        output: str = "mock output",
        task_types: frozenset[str] = frozenset(),
    ) -> None:
        self._identity = AgentIdentity(name=name)
        self._capabilities = AgentCapabilities(task_types=task_types)
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
            raise AgentExecutionError(f"Failed: {self.identity.name}")
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


class MockPlanner(Planner):
    """Returns a pre-built plan."""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.call_count = 0
        self.received_previous_plan = None
        self.received_failure_context = None

    def generate_plan(
        self,
        goal: str,
        context: AgentExecutionContext,
        previous_plan: Plan | None = None,
        failure_context: list[StepResult] | None = None,
    ) -> Plan:
        self.call_count += 1
        self.received_previous_plan = previous_plan
        self.received_failure_context = failure_context
        # Return a fresh copy for each call to avoid state pollution
        return Plan(
            goal=self._plan.goal,
            steps={
                sid: PlanStep(
                    step_id=s.step_id,
                    description=s.description,
                    task_type=s.task_type,
                    required_capabilities=s.required_capabilities,
                    dependencies=s.dependencies,
                )
                for sid, s in self._plan.steps.items()
            },
            metadata=dict(self._plan.metadata),
        )


class FailingThenSucceedingPlanner(Planner):
    """First plan has a failing step, second plan succeeds."""

    def __init__(self) -> None:
        self.call_count = 0
        self.received_previous_plan = None
        self.received_failure_context = None

    def generate_plan(
        self,
        goal: str,
        context: AgentExecutionContext,
        previous_plan: Plan | None = None,
        failure_context: list[StepResult] | None = None,
    ) -> Plan:
        self.call_count += 1
        self.received_previous_plan = previous_plan
        self.received_failure_context = failure_context
        if self.call_count == 1:
            # Return a plan with a step that requires a missing task type
            return Plan(
                goal=goal,
                steps={
                    "s1": PlanStep(
                        step_id="s1",
                        description="Do impossible work",
                        task_type="nonexistent_type",
                    ),
                },
            )
        else:
            # Return a plan with a step that can be routed
            return Plan(
                goal=goal,
                steps={
                    "s1": PlanStep(
                        step_id="s1",
                        description="Do research",
                        task_type="research",
                    ),
                },
            )


# ------------------------------------------------------------------ #
# PlanExecutor tests
# ------------------------------------------------------------------ #

def _build_executor() -> tuple[PlanExecutor, AgentRegistry, MockAgent, MockAgent]:
    registry = AgentRegistry()
    researcher = MockAgent("researcher", output="research result", task_types=frozenset(["research"]))
    analyst = MockAgent("analyst", output="analysis result", task_types=frozenset(["analysis"]))
    registry.register(researcher)
    registry.register(analyst)

    communicator = InProcessCommunicator(registry)
    executor = PlanExecutor(
        router=CapabilityRouter(),
        registry=registry,
        communicator=communicator,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    return executor, registry, researcher, analyst


def test_executor_single_step_plan() -> None:
    executor, _, researcher, _ = _build_executor()
    plan = Plan(
        goal="Research topic",
        steps={
            "s1": PlanStep(step_id="s1", description="Do research", task_type="research"),
        },
    )
    context = AgentExecutionContext(task="test")

    result_plan = executor.execute(plan, context, ExecutionPolicy())

    assert result_plan.status == PlanStatus.COMPLETED
    assert result_plan.steps["s1"].status == PlanStepStatus.COMPLETED
    assert result_plan.steps["s1"].result is not None
    assert result_plan.steps["s1"].result.success is True
    assert researcher.calls == 1


def test_executor_sequential_dependencies() -> None:
    executor, _, researcher, analyst = _build_executor()
    plan = Plan(
        goal="Research and analyze",
        steps={
            "s1": PlanStep(step_id="s1", description="Do research", task_type="research"),
            "s2": PlanStep(step_id="s2", description="Do analysis", task_type="analysis", dependencies=("s1",)),
        },
    )
    context = AgentExecutionContext(task="test")

    result_plan = executor.execute(plan, context, ExecutionPolicy())

    assert result_plan.status == PlanStatus.COMPLETED
    assert result_plan.steps["s1"].status == PlanStepStatus.COMPLETED
    assert result_plan.steps["s2"].status == PlanStepStatus.COMPLETED
    assert researcher.calls == 1
    assert analyst.calls == 1


def test_executor_branching_dependencies() -> None:
    """
    s1 (research) ──┐
                    ├──→ s3 (analysis)
    s2 (research) ──┘
    """
    executor, _, researcher, analyst = _build_executor()
    plan = Plan(
        goal="Multi-branch",
        steps={
            "s1": PlanStep(step_id="s1", description="Research A", task_type="research"),
            "s2": PlanStep(step_id="s2", description="Research B", task_type="research"),
            "s3": PlanStep(step_id="s3", description="Analyze all", task_type="analysis", dependencies=("s1", "s2")),
        },
    )
    context = AgentExecutionContext(task="test")

    result_plan = executor.execute(plan, context, ExecutionPolicy())

    assert result_plan.status == PlanStatus.COMPLETED
    assert all(s.status == PlanStepStatus.COMPLETED for s in result_plan.steps.values())
    assert researcher.calls == 2
    assert analyst.calls == 1


def test_executor_dependent_step_waits_correctly() -> None:
    executor, _, researcher, analyst = _build_executor()
    plan = Plan(
        goal="Sequential test",
        steps={
            "s1": PlanStep(step_id="s1", description="Do research", task_type="research"),
            "s2": PlanStep(step_id="s2", description="Do analysis", task_type="analysis", dependencies=("s1",)),
        },
    )
    context = AgentExecutionContext(task="test")

    # Initially s2 should be PENDING
    assert plan.steps["s2"].status == PlanStepStatus.PENDING

    result_plan = executor.execute(plan, context, ExecutionPolicy())

    # After execution, s2 should have been unlocked and completed
    assert result_plan.steps["s1"].status == PlanStepStatus.COMPLETED
    assert result_plan.steps["s2"].status == PlanStepStatus.COMPLETED


def test_executor_routing_failure_marks_step_failed() -> None:
    executor, _, _, _ = _build_executor()
    plan = Plan(
        goal="Impossible route",
        steps={
            "s1": PlanStep(step_id="s1", description="Unknown task", task_type="nonexistent"),
        },
    )
    context = AgentExecutionContext(task="test")

    result_plan = executor.execute(plan, context, ExecutionPolicy())

    assert result_plan.status == PlanStatus.FAILED
    assert result_plan.steps["s1"].status == PlanStepStatus.FAILED
    assert result_plan.steps["s1"].result is not None
    assert "No suitable agent" in (result_plan.steps["s1"].result.error or "")


def test_executor_respects_max_failures() -> None:
    executor, _, _, _ = _build_executor()
    plan = Plan(
        goal="Max failures",
        steps={
            "s1": PlanStep(step_id="s1", description="Unknown 1", task_type="nonexistent"),
            "s2": PlanStep(step_id="s2", description="Unknown 2", task_type="also_nonexistent"),
        },
    )
    context = AgentExecutionContext(task="test")
    policy = ExecutionPolicy(max_failures=1)

    result_plan = executor.execute(plan, context, policy)

    assert result_plan.status == PlanStatus.FAILED


def test_executor_respects_max_steps() -> None:
    executor, _, _, _ = _build_executor()
    plan = Plan(
        goal="Max steps",
        steps={
            "s1": PlanStep(step_id="s1", description="Do research", task_type="research"),
            "s2": PlanStep(step_id="s2", description="Do more research", task_type="research"),
            "s3": PlanStep(step_id="s3", description="Do analysis", task_type="analysis"),
        },
    )
    context = AgentExecutionContext(task="test")
    policy = ExecutionPolicy(max_steps=1)

    result_plan = executor.execute(plan, context, policy)

    assert result_plan.status == PlanStatus.FAILED


# ------------------------------------------------------------------ #
# ReplanningPolicy tests
# ------------------------------------------------------------------ #

def test_replanning_allowed_on_failure() -> None:
    policy = ReplanningPolicy(ExecutionPolicy(allow_replanning=True, max_replans=2))
    failed_plan = Plan(goal="test", status=PlanStatus.FAILED)
    assert policy.should_replan(failed_plan) is True


def test_replanning_disabled_by_policy() -> None:
    policy = ReplanningPolicy(ExecutionPolicy(allow_replanning=False))
    failed_plan = Plan(goal="test", status=PlanStatus.FAILED)
    assert policy.should_replan(failed_plan) is False


def test_replanning_max_reached() -> None:
    policy = ReplanningPolicy(ExecutionPolicy(allow_replanning=True, max_replans=1))
    failed_plan = Plan(goal="test", status=PlanStatus.FAILED)

    assert policy.should_replan(failed_plan) is True
    policy.record_replan()

    assert policy.should_replan(failed_plan) is False


def test_replanning_not_needed_on_success() -> None:
    policy = ReplanningPolicy(ExecutionPolicy(allow_replanning=True, max_replans=5))
    completed_plan = Plan(goal="test", status=PlanStatus.COMPLETED)
    assert policy.should_replan(completed_plan) is False


# ------------------------------------------------------------------ #
# Supervisor with planning stack tests
# ------------------------------------------------------------------ #

def test_supervisor_planning_path_success() -> None:
    registry = AgentRegistry()
    agent = MockAgent("researcher", output="research done", task_types=frozenset(["research"]))
    registry.register(agent)

    communicator = InProcessCommunicator(registry)
    plan_executor = PlanExecutor(
        router=CapabilityRouter(),
        registry=registry,
        communicator=communicator,
    )

    plan = Plan(
        goal="Do research",
        steps={
            "s1": PlanStep(step_id="s1", description="Research topic", task_type="research"),
        },
    )
    planner = MockPlanner(plan)

    supervisor = Supervisor(
        registry=registry,
        planner=planner,
        plan_executor=plan_executor,
    )

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="research AI safety", context=context)
    result = supervisor.execute(request)

    assert result.success is True
    assert "research done" in result.output
    assert planner.call_count == 1


def test_supervisor_planning_replanning_on_failure() -> None:
    registry = AgentRegistry()
    agent = MockAgent("researcher", output="research done", task_types=frozenset(["research"]))
    registry.register(agent)

    communicator = InProcessCommunicator(registry)
    plan_executor = PlanExecutor(
        router=CapabilityRouter(),
        registry=registry,
        communicator=communicator,
    )

    planner = FailingThenSucceedingPlanner()

    supervisor = Supervisor(
        registry=registry,
        planner=planner,
        plan_executor=plan_executor,
        execution_policy=ExecutionPolicy(allow_replanning=True, max_replans=2),
    )

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="research AI safety", context=context)
    result = supervisor.execute(request)

    assert result.success is True
    assert planner.call_count == 2  # First plan fails, second succeeds


def test_supervisor_planning_terminal_failure() -> None:
    registry = AgentRegistry()
    agent = MockAgent("researcher", output="research done", task_types=frozenset(["research"]))
    registry.register(agent)

    communicator = InProcessCommunicator(registry)
    plan_executor = PlanExecutor(
        router=CapabilityRouter(),
        registry=registry,
        communicator=communicator,
    )

    # Plan that always fails (no matching agent)
    failing_plan = Plan(
        goal="Impossible",
        steps={
            "s1": PlanStep(step_id="s1", description="Impossible", task_type="nonexistent"),
        },
    )
    planner = MockPlanner(failing_plan)

    supervisor = Supervisor(
        registry=registry,
        planner=planner,
        plan_executor=plan_executor,
        execution_policy=ExecutionPolicy(allow_replanning=False),
    )

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="impossible task", context=context)

    with pytest.raises(AgentExecutionError, match="replanning exhausted"):
        supervisor.execute(request)


def test_supervisor_legacy_path_still_works() -> None:
    """Phase 5.3/5.4 legacy path must still work without planning stack."""
    agent = MockAgent("agent1", output="legacy output")

    supervisor = Supervisor(agents=[agent])

    request = AgentRequest(input_text="do something")
    result = supervisor.execute(request)

    assert result.success is True
    assert result.output == "legacy output"
    assert agent.calls == 1


def test_planner_receives_previous_plan_on_replan() -> None:
    registry = AgentRegistry()
    agent = MockAgent("researcher", output="research done", task_types=frozenset(["research"]))
    registry.register(agent)

    communicator = InProcessCommunicator(registry)
    plan_executor = PlanExecutor(
        router=CapabilityRouter(),
        registry=registry,
        communicator=communicator,
    )

    planner = FailingThenSucceedingPlanner()

    supervisor = Supervisor(
        registry=registry,
        planner=planner,
        plan_executor=plan_executor,
        execution_policy=ExecutionPolicy(allow_replanning=True, max_replans=2),
    )

    context = AgentExecutionContext(task="test")
    request = AgentRequest(input_text="research AI safety", context=context)
    supervisor.execute(request)

    # On the second call, planner should have received the previous failed plan
    assert planner.received_previous_plan is not None
    assert planner.received_previous_plan.status == PlanStatus.FAILED
    assert planner.received_failure_context is not None
    assert len(planner.received_failure_context) > 0
