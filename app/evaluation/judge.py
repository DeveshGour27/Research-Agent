"""Probabilistic LLM Judge evaluator."""

import json
from typing import Any, Dict, List, Optional

from app.llm.gateway import ModelGateway
from app.llm.models import ModelRequest, TaskType
from app.evaluation.contracts import GoldenCase, JudgeResult
from app.agent.contracts import AgentResult
from app.logger import get_logger

logger = get_logger(__name__)


class LLMJudge:
    """Evaluates agent responses probabilistically using an LLM provider."""

    def __init__(self, provider: ModelGateway):
        self._provider = provider
        
        self._tool_schema = {
            "type": "function",
            "function": {
                "name": "submit_evaluation",
                "description": "Submit structured evaluation scores.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "correctness": {
                            "type": "number", 
                            "description": "Score from 1.0 to 5.0 indicating how factually correct the output is."
                        },
                        "relevance": {
                            "type": "number", 
                            "description": "Score from 1.0 to 5.0 indicating if the output addresses the specific prompt."
                        },
                        "instruction_adherence": {
                            "type": "number", 
                            "description": "Score from 1.0 to 5.0 indicating how well negative and positive constraints were followed."
                        },
                        "reasoning": {
                            "type": "string", 
                            "description": "Concise reasoning for the given scores."
                        }
                    },
                    "required": ["correctness", "relevance", "instruction_adherence", "reasoning"]
                }
            }
        }

    def evaluate(self, case: GoldenCase, result: AgentResult) -> JudgeResult:
        """Evaluate an execution result against the golden case."""
        
        prompt = self._build_prompt(case, result)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._provider.generate_with_tools(
                messages=messages,
                tools=[self._tool_schema]
            )
            
            # 1. Parse Tool Call
            if response.tool_call and response.tool_call.name == "submit_evaluation":
                args = response.tool_call.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                    
                return self._parse_structured_args(args, response.model)
                
            # 2. Fallback to parsing text content if the model ignored the tool
            if response.content:
                try:
                    # Attempt to extract JSON from text output if tool calling failed
                    start_idx = response.content.find('{')
                    end_idx = response.content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = response.content[start_idx:end_idx+1]
                        args = json.loads(json_str)
                        return self._parse_structured_args(args, response.model)
                except Exception:
                    pass
                    
            # Failed to parse anywhere
            return JudgeResult(
                correctness=0.0, relevance=0.0, instruction_adherence=0.0,
                reasoning="Failed to extract structured evaluation scores.",
                judge_model=response.model,
                error="MALFORMED_OUTPUT"
            )
            
        except Exception as e:
            logger.warning("LLM Judge evaluation failed", exc_info=True)
            return JudgeResult(
                correctness=0.0, relevance=0.0, instruction_adherence=0.0,
                reasoning="API or connection failure during evaluation.",
                judge_model="unknown",
                error=str(e)
            )

    def _build_prompt(self, case: GoldenCase, result: AgentResult) -> str:
        prompt = (
            "You are an expert quality assurance evaluator.\n"
            "Evaluate the Agent Output based on the Task Input and Reference Answer.\n\n"
            f"--- TASK INPUT ---\n{case.task_input}\n\n"
        )
        
        if case.reference_answer:
            prompt += f"--- REFERENCE ANSWER ---\n{case.reference_answer}\n\n"
            
        prompt += f"--- AGENT OUTPUT ---\n{result.output or '<Empty Output>'}\n\n"
        
        prompt += (
            "Provide your evaluation using the submit_evaluation tool. "
            "Keep reasoning concise. Do not expose chain-of-thought."
        )
        return prompt

    def _parse_structured_args(self, args: Dict[str, Any], model: str) -> JudgeResult:
        try:
            return JudgeResult(
                correctness=float(args.get("correctness", 0.0)),
                relevance=float(args.get("relevance", 0.0)),
                instruction_adherence=float(args.get("instruction_adherence", 0.0)),
                reasoning=str(args.get("reasoning", "")),
                judge_model=model,
                error=None
            )
        except (ValueError, TypeError) as e:
            return JudgeResult(
                correctness=0.0, relevance=0.0, instruction_adherence=0.0,
                reasoning=f"Failed to parse score types: {e}",
                judge_model=model,
                error="TYPE_ERROR"
            )


