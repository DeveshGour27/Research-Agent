from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json
from app.llm.gateway import ModelGateway
from app.llm.models import ModelRequest, TaskType, ModelCapability

class JudgeScore(BaseModel):
    score: int = Field(..., ge=1, le=5, description="Score from 1 to 5")
    reason: str = Field(..., description="Short explanation for the score")

class LLMJudgeEvaluator:
    def __init__(self, provider: ModelGateway, model_id: str):
        self.provider = provider
        self.model_id = model_id
        
    def evaluate(self, question: str, answer: str, expected: Optional[str]) -> Dict[str, Any]:
        prompt = (
            f"Question: {question}\n"
            f"Agent Answer: {answer}\n"
        )
        if expected:
            prompt += f"Reference Answer: {expected}\n"
            
        prompt += "\nEvaluate the Agent Answer based on Correctness (1-5). Return JSON with 'score' and 'reason'."
        
        req = ModelRequest(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.REFLECTION,
            required_capabilities={ModelCapability.STRUCTURED_OUTPUT, ModelCapability.REASONING},
            response_format={"type": "json_schema", "json_schema": {"name": "judge_score", "schema": JudgeScore.model_json_schema()}}
        )
        
        try:
            resp = self.provider.generate(req)
            data = json.loads(resp.content or "{}")
            return {
                "correctness_score": data.get("score"),
                "correctness_reason": data.get("reason"),
                "judge_model": resp.model,
                "judge_provider": resp.provider
            }
        except Exception as e:
            return {
                "correctness_score": None,
                "correctness_reason": f"Judge error: {str(e)}",
                "judge_model": self.model_id,
                "judge_provider": "unknown"
            }
