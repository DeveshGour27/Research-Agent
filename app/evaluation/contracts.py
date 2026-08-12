"""Evaluation contracts for Phase 5.8 Step 6."""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

from app.evaluation.replay import ReplayRecord, _reject_callables


@dataclass(frozen=True)
class GoldenConstraints:
    expected_agent: Optional[str] = None
    required_tools: FrozenSet[str] = field(default_factory=frozenset)
    must_not_handoff: bool = False
    expected_success: bool = True

    expected_task_types: List[str] = field(default_factory=list)
    expected_capabilities: List[str] = field(default_factory=list)
    expected_plan_steps: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenConstraints":
        if not isinstance(data, dict):
            raise ValueError("GoldenConstraints data must be a dictionary")
        _reject_callables(data)
        
        req_tools = data.get("required_tools", [])
        if not isinstance(req_tools, list):
             raise ValueError("required_tools must be a list")
             
        task_types = data.get("expected_task_types", [])
        if not isinstance(task_types, list):
             raise ValueError("expected_task_types must be a list")
             
        caps = data.get("expected_capabilities", [])
        if not isinstance(caps, list):
             raise ValueError("expected_capabilities must be a list")
             
        plan_steps = data.get("expected_plan_steps")
        if plan_steps is not None and not isinstance(plan_steps, int):
             raise ValueError("expected_plan_steps must be an integer")
             
        return cls(
            expected_agent=str(data["expected_agent"]) if data.get("expected_agent") is not None else None,
            required_tools=frozenset(str(t) for t in req_tools),
            must_not_handoff=bool(data.get("must_not_handoff", False)),
            expected_success=bool(data.get("expected_success", True)),
            expected_task_types=[str(t) for t in task_types],
            expected_capabilities=[str(c) for c in caps],
            expected_plan_steps=plan_steps,
        )


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    task_input: str
    constraints: GoldenConstraints
    reference_answer: Optional[str] = None
    replay_record: Optional[ReplayRecord] = None
    tags: FrozenSet[str] = field(default_factory=frozenset)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenCase":
        if not isinstance(data, dict):
            raise ValueError("GoldenCase data must be a dictionary")
        _reject_callables(data)
        
        if "case_id" not in data or "task_input" not in data or "constraints" not in data:
            raise ValueError("case_id, task_input, and constraints are required")
            
        constraints = GoldenConstraints.from_dict(data["constraints"])
        
        record = None
        raw_record = data.get("replay_record")
        if raw_record is not None:
            if not isinstance(raw_record, dict):
                raise ValueError("replay_record must be a dictionary")
            record = ReplayRecord.from_dict(raw_record)
            
        raw_tags = data.get("tags", [])
        if not isinstance(raw_tags, list):
             raise ValueError("tags must be a list")
             
        return cls(
            case_id=str(data["case_id"]),
            task_input=str(data["task_input"]),
            constraints=constraints,
            reference_answer=str(data["reference_answer"]) if data.get("reference_answer") is not None else None,
            replay_record=record,
            tags=frozenset(str(t) for t in raw_tags)
        )


@dataclass(frozen=True)
class DeterministicGateResult:
    passed: bool
    failures: List[str]
    replay_mismatch: bool = False


@dataclass(frozen=True)
class JudgeResult:
    correctness: float
    relevance: float
    instruction_adherence: float
    reasoning: str
    judge_model: str
    error: Optional[str] = None


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    deterministic: DeterministicGateResult
    probabilistic: Optional[JudgeResult]
    duration_ms: float


@dataclass(frozen=True)
class EvaluationSummary:
    total_cases: int
    passed_deterministic: int
    failed_deterministic: int
    average_scores: Dict[str, float]
    results: List[EvaluationResult]
