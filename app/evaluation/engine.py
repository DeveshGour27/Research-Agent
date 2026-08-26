"""Evaluation Engine for orchestrating GoldenCase evaluation."""

import time
import traceback
from typing import List, Type, Dict, Any, Optional
import uuid

from app.evaluation.contracts import (
    GoldenCase, EvaluationResult, EvaluationSummary, DeterministicGateResult, JudgeResult
)
from app.evaluation.judge import LLMJudge
from app.evaluation.replay import ReplayEngine, ReplayStatus
from app.agent.contracts import BaseAgent, AgentRequest, AgentResult
from app.agent.execution_context import AgentExecutionContext
from app.observability.events import BaseEvent
from app.logger import get_logger
from unittest.mock import patch

logger = get_logger(__name__)


class EvaluationEngine:
    """Orchestrates golden dataset evaluation, delegating to replay or live sandboxes."""

    def __init__(self, agent_class: Type[BaseAgent], agent_kwargs: Dict[str, Any], judge: Optional[LLMJudge] = None):
        self._agent_class = agent_class
        self._agent_kwargs = agent_kwargs
        self._judge = judge

    def run_suite(self, dataset: List[GoldenCase]) -> EvaluationSummary:
        """Run a full evaluation suite against a list of GoldenCases."""
        results = []
        for case in dataset:
            results.append(self.evaluate_case(case))
            
        passed = sum(1 for r in results if r.deterministic.passed)
        failed = len(results) - passed
        
        avg_scores = {"correctness": 0.0, "relevance": 0.0, "instruction_adherence": 0.0}
        judge_count = 0
        for r in results:
            if r.probabilistic and not r.probabilistic.error:
                avg_scores["correctness"] += r.probabilistic.correctness
                avg_scores["relevance"] += r.probabilistic.relevance
                avg_scores["instruction_adherence"] += r.probabilistic.instruction_adherence
                judge_count += 1
                
        if judge_count > 0:
            avg_scores = {k: v / judge_count for k, v in avg_scores.items()}

        return EvaluationSummary(
            total_cases=len(dataset),
            passed_deterministic=passed,
            failed_deterministic=failed,
            average_scores=avg_scores,
            results=results
        )

    def evaluate_case(self, case: GoldenCase) -> EvaluationResult:
        start_time = time.time()
        
        # 1. Execution Phase
        if case.replay_record:
            # Deterministic Replay
            agent = self._agent_class(**self._agent_kwargs)
            replay_engine = ReplayEngine(agent, case.replay_record)
            
            # Formulate the request from the golden case task input
            req = AgentRequest(input_text=case.task_input)
            
            # Execute replay
            replay_res = replay_engine.execute(req)
            
            agent_output = replay_res.details.get("output")
            agent_success = replay_res.details.get("success", False)
            agent_metadata = replay_res.details.get("metadata", {})
            replay_mismatch = replay_res.status == ReplayStatus.REPLAY_MISMATCH
            fatal_error = replay_res.status == ReplayStatus.FAILED
            if fatal_error:
                print(f"REPLAY ENGINE FATAL ERROR: {replay_res.details.get("traceback", "")}")
            
            # Create a mock agent result for the judge if we succeeded
            result = AgentResult(
                request=req, state=None, output=agent_output, success=agent_success, metadata=agent_metadata # type: ignore
            ) if not fatal_error else None
            
        else:
            # Live Sandboxed Evaluation
            # Evaluation must not alter normal agent execution behavior and no live mutation
            captured_events = []
            
            def _capture_event(self, *args, **kwargs):
                captured_events.append(self)
                
            try:
                agent = self._agent_class(**self._agent_kwargs)
                
                req = AgentRequest(
                    input_text=case.task_input,
                    context=AgentExecutionContext(task=case.task_input, execution_id=str(uuid.uuid4()))
                )
                
                original_emit = BaseEvent.emit
                BaseEvent.emit = _capture_event
                try:
                    result = agent.execute(req)
                finally:
                    BaseEvent.emit = original_emit
                    
                agent_output = result.output
                agent_success = result.success
                replay_mismatch = False
                fatal_error = False
            except Exception as e:
                logger.error(f"Live evaluation failed for {case.case_id}", exc_info=True)
                result = None
                agent_output = None
                agent_success = False
                replay_mismatch = False
                fatal_error = True

        # 2. Deterministic Gate Evaluation
        failures = []
        if fatal_error and case.constraints.expected_success:
            failures.append("Execution resulted in a fatal error.")
        if replay_mismatch:
            failures.append("Replay trace diverged from GoldenCase ReplayRecord.")
            
        if not fatal_error and not replay_mismatch:
            if case.constraints.expected_success != agent_success:
                failures.append(f"Expected success={case.constraints.expected_success}, got {agent_success}")
                
            if not case.replay_record:
                # Live evaluation event verification
                agent_names = [e.event_data.get("agent_name") for e in captured_events if e.event_type == "AgentExecutionStarted"]
                tools_called = set(e.event_data.get("tool_name") for e in captured_events if e.event_type == "ToolCalled")
                handoffs = [e for e in captured_events if e.event_type == "HandoffInitiated"]
                
                if case.constraints.expected_agent:
                    if not agent_names or agent_names[0] != case.constraints.expected_agent:
                        failures.append(f"Expected agent {case.constraints.expected_agent}, but executed {agent_names}")
                
                if case.constraints.required_tools:
                    missing_tools = case.constraints.required_tools - tools_called
                    if missing_tools:
                        failures.append(f"Missing required tools: {missing_tools}")
                        
                if case.constraints.must_not_handoff and handoffs:
                    failures.append("Handoff initiated but must_not_handoff was True.")

            # Plan assertions
            plan = None
            if result and result.metadata:
                plan = result.metadata.get("executed_plan")

            if case.constraints.expected_plan_steps is not None:
                if plan:
                    if len(plan.steps) != case.constraints.expected_plan_steps:
                        failures.append(f"Expected {case.constraints.expected_plan_steps} plan steps, got {len(plan.steps)}")
                elif not case.replay_record:
                    # Fallback to observability event
                    plan_events = [e for e in captured_events if e.event_type == "PlanGenerated"]
                    if not plan_events:
                        failures.append("Expected plan steps but no PlanGeneratedEvent found and no plan in metadata.")
                    else:
                        last_plan = plan_events[-1]
                        if last_plan.event_data.get("step_count") != case.constraints.expected_plan_steps:
                            failures.append(f"Expected {case.constraints.expected_plan_steps} plan steps in event, got {last_plan.event_data.get('step_count')}")
                else:
                    failures.append("Expected plan steps but no plan found in replay result metadata.")

            if case.constraints.expected_task_types:
                if plan:
                    actual_types = [s.task_type for s in plan.steps.values()]
                    missing = set(case.constraints.expected_task_types) - set(actual_types)
                    if missing:
                        failures.append(f"Expected task types {missing} not found in plan.")
                else:
                    failures.append("Expected task types but no plan found in result metadata.")

            if case.constraints.expected_capabilities:
                if plan:
                    actual_caps = set()
                    for s in plan.steps.values():
                        actual_caps.update(s.required_capabilities)
                    missing = set(case.constraints.expected_capabilities) - actual_caps
                    if missing:
                        failures.append(f"Expected capabilities {missing} not found in plan.")
                else:
                    failures.append("Expected capabilities but no plan found in result metadata.")

        det_result = DeterministicGateResult(
            passed=len(failures) == 0,
            failures=failures,
            replay_mismatch=replay_mismatch
        )

        # 3. Probabilistic Evaluation (LLM-Judge)
        prob_result = None
        if self._judge and result:
            # Judge is informational, we always run it if result exists and judge is present
            prob_result = self._judge.evaluate(case, result)
            
        duration = (time.time() - start_time) * 1000
        return EvaluationResult(
            case_id=case.case_id,
            deterministic=det_result,
            probabilistic=prob_result,
            duration_ms=duration
        )
