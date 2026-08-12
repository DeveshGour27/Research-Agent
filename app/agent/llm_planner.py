"""LLMPlanner implementation for Phase 6.3."""

from __future__ import annotations

import json
from typing import Any

from app.agent.execution_context import AgentExecutionContext
from app.agent.plan import Plan, PlanStep, PlanStepStatus, StepResult
from app.agent.planner import Planner
from app.agent.validator import PlanValidator
from app.config import settings
from app.exceptions import PlanCreationError, PlanValidationError
from app.llm.base import ChatMessage, LLMProvider, ToolCall
from app.logger import get_logger
from app.observability.events import PlanGeneratedEvent

logger = get_logger(__name__)

ALLOWED_TASK_TYPES: frozenset[str] = frozenset({"web_search", "rag_search", "calculation", "reasoning"})
ALLOWED_CAPABILITIES: frozenset[str] = frozenset({"web_search", "rag_search", "tool_use", "retrieval"})

PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the structured plan to execute the goal.",
        "arguments": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "List of steps in the plan.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {
                                "type": "string",
                                "description": "Unique identifier for the step."
                            },
                            "description": {
                                "type": "string",
                                "description": "Clear description of the work to be performed."
                            },
                            "task_type": {
                                "type": "string",
                                "description": "The specific task type.",
                                "enum": list(ALLOWED_TASK_TYPES)
                            },
                            "required_capabilities": {
                                "type": "array",
                                "items": {"type": "string", "enum": list(ALLOWED_CAPABILITIES)},
                                "description": "Capabilities required to execute this step."
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of step_ids that must complete before this step."
                            }
                        },
                        "required": ["step_id", "description", "task_type", "required_capabilities", "dependencies"]
                    }
                }
            },
            "required": ["steps"]
        }
    }
}


class LLMPlanner(Planner):
    """LLM-backed plan generator that converts a goal into a Plan DAG.
    
    Generates a structured plan via LLM tool-calling, deserializes it, enforces
    server-side constraints, and validates it before returning.
    """

    def __init__(self, provider: LLMProvider, validator: PlanValidator | None = None) -> None:
        self._provider = provider
        self._validator = validator or PlanValidator()
        self._max_plan_steps = settings.max_plan_steps

    def generate_plan(
        self,
        goal: str,
        context: AgentExecutionContext,
        previous_plan: Plan | None = None,
        failure_context: list[StepResult] | None = None,
    ) -> Plan:
        # 1. Prompt construction
        system_prompt = self._construct_system_prompt()
        user_prompt = self._construct_user_prompt(goal, previous_plan, failure_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        tools = [PLAN_SCHEMA]

        # 2. Call LLM provider
        try:
            llm_response = self._provider.generate_with_tools(messages, tools)
        except Exception as e:
            raise PlanCreationError(f"LLM provider failed during plan generation: {e}") from e

        # 3. Extract tool call
        tool_call = llm_response.tool_call
        if not tool_call or tool_call.name != "submit_plan":
            raise PlanCreationError("LLM failed to return a plan via the 'submit_plan' tool.")

        # 4. JSON parsing and Deserialization
        parsed_args = tool_call.arguments
        if "__parse_error__" in parsed_args:
            raise PlanCreationError(
                "LLM returned malformed JSON arguments for 'submit_plan'.",
                details={"raw_arguments": parsed_args.get("__raw_arguments__")}
            )

        if not isinstance(parsed_args, dict) or "steps" not in parsed_args:
            raise PlanCreationError("LLM returned incomplete plan arguments (missing 'steps').")

        raw_steps = parsed_args.get("steps")
        if not isinstance(raw_steps, list):
            raise PlanCreationError("'steps' must be an array.")

        # 5. Server-side constraints (Task types, capabilities, step count)
        if len(raw_steps) > self._max_plan_steps:
            raise PlanCreationError(
                f"Generated plan exceeds maximum allowed steps ({self._max_plan_steps}).",
                details={"step_count": len(raw_steps)}
            )

        plan = Plan(goal=goal)
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                raise PlanCreationError("Invalid step structure (must be an object).")

            step_id = raw_step.get("step_id")
            description = raw_step.get("description")
            task_type = raw_step.get("task_type")
            req_caps = raw_step.get("required_capabilities", [])
            deps = raw_step.get("dependencies", [])

            if not step_id or not isinstance(step_id, str):
                raise PlanCreationError("Step is missing a valid 'step_id'.")
            if not description or not isinstance(description, str):
                raise PlanCreationError(f"Step '{step_id}' is missing a valid 'description'.")

            # Validate server constraints
            if task_type not in ALLOWED_TASK_TYPES:
                raise PlanCreationError(f"Step '{step_id}' has an unknown task type: '{task_type}'.")
            
            validated_caps: set[str] = set()
            for cap in req_caps:
                if cap not in ALLOWED_CAPABILITIES:
                    raise PlanCreationError(f"Step '{step_id}' specifies an unknown capability: '{cap}'.")
                validated_caps.add(cap)

            if step_id in plan.steps:
                raise PlanCreationError(f"Duplicate step ID '{step_id}' detected.")

            plan.steps[step_id] = PlanStep(
                step_id=step_id,
                description=description,
                task_type=task_type,
                required_capabilities=frozenset(validated_caps),
                dependencies=tuple(str(d) for d in deps),
                status=PlanStepStatus.PENDING,
            )

        # 6. Existing PlanValidator
        try:
            self._validator.validate(plan)
        except PlanValidationError as e:
            raise PlanCreationError(f"Generated plan failed structural validation: {e.message}", details=e.details) from e

        # 7. PlanGeneratedEvent
        is_replan = previous_plan is not None
        PlanGeneratedEvent(
            trace_id=context.trace_id,
            run_id=context.run_id,
            span_id=context.span_id,
            parent_span_id=context.parent_span_id,
            plan_id=plan.plan_id,
            step_count=len(plan.steps),
            is_replan=is_replan
        ).emit()

        # 8. Return Plan
        return plan

    def _construct_system_prompt(self) -> str:
        return (
            "You are an expert planning agent. Your task is to generate a structured execution plan to achieve a goal.\n"
            f"You MUST return the plan by calling the 'submit_plan' tool. Maximum steps allowed: {self._max_plan_steps}.\n"
            "Each step must reference a valid task_type and valid capabilities.\n"
            f"Allowed task types: {sorted(list(ALLOWED_TASK_TYPES))}.\n"
            f"Allowed capabilities: {sorted(list(ALLOWED_CAPABILITIES))}.\n"
            "Steps can depend on prior steps. A step with dependencies can only execute when all its dependencies have completed.\n"
            "Do NOT include cyclical dependencies."
        )

    def _construct_user_prompt(self, goal: str, previous_plan: Plan | None, failure_context: list[StepResult] | None) -> str:
        prompt = f"Goal:\n{goal}\n\n"
        
        if previous_plan and failure_context:
            prompt += "--- REPLANNING CONTEXT ---\n"
            prompt += "A previous attempt to execute this goal failed. Create a revised plan.\n"
            prompt += "Previous step failures:\n"
            for result in failure_context:
                prompt += f"- Step {result.step_id}: {result.error}\n"
        
        return prompt
