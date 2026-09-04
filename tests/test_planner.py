import pytest
import asyncio
from app.config import settings
from app.llm.factory import create_model_gateway
from app.agent.llm_planner import LLMPlanner
from app.agent.execution_context import AgentExecutionContext

def test_planner_simple_request():
    async def run_test():
        gateway = create_model_gateway(settings)
        planner = LLMPlanner(provider=gateway)
        context = AgentExecutionContext(task='What is the capital of France?')
        
        plan = planner.generate_plan(
            goal='What is the capital of France?',
            context=context
        )
        
        assert plan is not None
        assert len(plan.steps) > 0
        step = list(plan.steps.values())[0]
        assert step.step_id is not None
        assert step.description is not None
        assert step.task_type is not None
    asyncio.run(run_test())

def test_planner_research_request():
    async def run_test():
        gateway = create_model_gateway(settings)
        planner = LLMPlanner(provider=gateway)
        context = AgentExecutionContext(task='Find recent research papers on NASA C-MAPSS remaining useful life prediction.')
        
        plan = planner.generate_plan(
            goal='Find recent research papers on NASA C-MAPSS remaining useful life prediction.',
            context=context
        )
        
        assert plan is not None
        assert len(plan.steps) > 0
        step = list(plan.steps.values())[0]
        assert step.step_id is not None
        assert step.description is not None
        assert step.task_type is not None
    asyncio.run(run_test())
