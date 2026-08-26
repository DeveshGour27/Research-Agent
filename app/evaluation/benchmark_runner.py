from typing import Dict, Any, List, Optional
import time
import uuid
import logging
from dataclasses import dataclass, field
from app.evaluation.benchmark_dataset import BenchmarkCase, BenchmarkDataset
from app.evaluation.benchmark_metrics import DeterministicMetrics
from app.evaluation.benchmark_evaluators import LLMJudgeEvaluator
from app.evaluation.benchmark_log_capture import EvaluationLogCaptureHandler
from app.agent.agent import Agent
from app.agent.contracts import AgentRequest
from app.logger import get_logger

logger = get_logger(__name__)

@dataclass
class BenchmarkRunResult:
    run_id: str
    timestamp: float
    agent_config: Dict[str, Any]
    case_results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "agent_config": self.agent_config,
            "case_results": self.case_results,
            "summary": self.summary
        }

class BenchmarkRunner:
    def __init__(self, agent: Agent, judge: Optional[LLMJudgeEvaluator] = None):
        self.agent = agent
        self.judge = judge
        
    def run(self, dataset: BenchmarkDataset, agent_config_meta: Dict[str, Any]) -> BenchmarkRunResult:
        run_id = str(uuid.uuid4())
        results = []
        
        # Setup log capture
        capture = EvaluationLogCaptureHandler()
        root_logger = logging.getLogger()
        root_logger.addHandler(capture)
        # also capture gateway and observability logs which might be custom instances
        logging.getLogger("agent.app.llm.gateway").addHandler(capture)
        logging.getLogger("observability").addHandler(capture)
        
        try:
            for case in dataset:
                logger.info(f"Running benchmark case {case.id}")
                capture.reset()
                start_time = time.perf_counter()
                req = AgentRequest(input_text=case.question)
                
                try:
                    res = self.agent.execute(req)
                    success = res.success
                    answer = res.output
                    error = None
                    failure_category = None
                    if not success:
                        failure_category = "agent_failure"
                        res_str = str(res.output).lower()
                        if "plan" in res_str:
                            failure_category = "planning failure"
                        elif "tool" in res_str:
                            failure_category = "tool failure"
                except Exception as e:
                    success = False
                    answer = ""
                    error = str(e)
                    failure_category = "timeout" if "timeout" in error.lower() else "model failure"
                    
                latency = time.perf_counter() - start_time
                
                tokens = capture.get_token_usage()
                
                case_result = {
                    "case_id": case.id,
                    "success": success,
                    "answer": answer,
                    "latency": latency,
                    "error": error,
                    "failure_category": failure_category,
                    "metrics": {
                        "exact_match": DeterministicMetrics.exact_match(answer, case.expected_answer),
                        "concept_coverage": DeterministicMetrics.concept_coverage(answer, case.required_concepts)
                    },
                    "operational": {
                        "total_tokens": tokens["total_tokens"],
                        "input_tokens": tokens["input_tokens"],
                        "output_tokens": tokens["output_tokens"],
                        "llm_calls": capture.get_llm_calls(),
                        "tool_calls": capture.get_tool_calls(),
                        "retrieval_calls": capture.get_retrieval_calls(),
                        "cost": "unknown"
                    }
                }
                
                if self.judge and success:
                    judge_metrics = self.judge.evaluate(case.question, answer, case.expected_answer)
                    case_result["metrics"].update(judge_metrics)
                    
                results.append(case_result)
        finally:
            root_logger.removeHandler(capture)
            logging.getLogger("agent.app.llm.gateway").removeHandler(capture)
            logging.getLogger("observability").removeHandler(capture)
            
        return self._summarize(run_id, agent_config_meta, results)
        
    def _summarize(self, run_id: str, agent_config: Dict[str, Any], results: List[Dict[str, Any]]) -> BenchmarkRunResult:
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        
        latencies = [r["latency"] for r in results]
        mean_latency = sum(latencies) / total if total > 0 else 0
        
        coverages = [r["metrics"]["concept_coverage"] for r in results if r["metrics"]["concept_coverage"] is not None]
        mean_coverage = sum(coverages) / len(coverages) if coverages else None
        
        scores = [r["metrics"].get("correctness_score") for r in results if r.get("metrics", {}).get("correctness_score") is not None]
        mean_score = sum(scores) / len(scores) if scores else None
        
        summary = {
            "total_cases": total,
            "success_rate": successful / total if total > 0 else 0,
            "mean_latency": mean_latency,
            "mean_concept_coverage": mean_coverage,
            "mean_judge_correctness": mean_score
        }
        
        return BenchmarkRunResult(
            run_id=run_id,
            timestamp=time.time(),
            agent_config=agent_config,
            case_results=results,
            summary=summary
        )
