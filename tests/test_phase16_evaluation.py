import pytest
from unittest.mock import MagicMock
import os
import json
from app.evaluation.benchmark_dataset import BenchmarkDataset, BenchmarkCase
from app.evaluation.benchmark_metrics import DeterministicMetrics
from app.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkRunResult
from app.evaluation.benchmark_reporters import BenchmarkReporter
from app.agent.contracts import AgentResult, AgentRequest
from app.agent.state import AgentState

def test_dataset_loading(tmp_path):
    data = {"id": "c1", "question": "Q", "expected_answer": "A", "required_concepts": ["A"]}
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps(data))
    
    ds = BenchmarkDataset.load_jsonl(str(p))
    assert len(ds) == 1
    assert ds.cases[0].id == "c1"

def test_deterministic_metrics():
    assert DeterministicMetrics.exact_match("hello", "hello") == 1.0
    assert DeterministicMetrics.exact_match("hello", "world") == 0.0
    assert DeterministicMetrics.exact_match("hello", None) is None
    
    assert DeterministicMetrics.concept_coverage("apple and banana", ["apple", "banana"]) == 1.0
    assert DeterministicMetrics.concept_coverage("apple", ["apple", "banana"]) == 0.5
    assert DeterministicMetrics.concept_coverage("orange", []) is None
    
    assert DeterministicMetrics.recall_at_k(["d1", "d2"], ["d2", "d3"], 2) == 0.5
    assert DeterministicMetrics.precision_at_k(["d1", "d2"], ["d2", "d3"], 2) == 0.5
    assert DeterministicMetrics.mrr(["d1", "d2"], ["d2", "d3"]) == 0.5
    assert DeterministicMetrics.mrr(["d1", "d4"], ["d2", "d3"]) == 0.0

def test_benchmark_runner():
    mock_agent = MagicMock()
    req = AgentRequest(input_text="dummy")
    state = AgentState(messages=[])
    mock_agent.execute.return_value = AgentResult(request=req, state=state, success=True, output="apple and banana")
    
    ds = BenchmarkDataset([
        BenchmarkCase(id="c1", question="fruits?", expected_answer="apple and banana", required_concepts=["apple"])
    ])
    
    runner = BenchmarkRunner(agent=mock_agent)
    result = runner.run(ds, agent_config_meta={"model_id": "test_model"})
    
    assert result.summary["total_cases"] == 1
    assert result.summary["success_rate"] == 1.0
    assert result.summary["mean_concept_coverage"] == 1.0
    assert result.case_results[0]["metrics"]["exact_match"] == 1.0
    assert result.agent_config["model_id"] == "test_model"

def test_reporter(tmp_path):
    res1 = BenchmarkRunResult(
        run_id="run1", timestamp=123, agent_config={"model_id": "ModelA"},
        summary={"success_rate": 0.8, "mean_judge_correctness": 4.5, "mean_concept_coverage": 0.9, "mean_latency": 2.0},
        case_results=[{"operational": {"total_tokens": 100}}]
    )
    res2 = BenchmarkRunResult(
        run_id="run2", timestamp=124, agent_config={"model_id": "ModelB"},
        summary={"success_rate": 0.9, "mean_judge_correctness": 4.8, "mean_concept_coverage": 0.95, "mean_latency": 1.5},
        case_results=[{"operational": {"total_tokens": 80}}]
    )
    
    out = BenchmarkReporter.generate_comparison_report([res1, res2])
    assert "ModelA" in out
    assert "ModelB" in out
    assert "80.0%" in out
    assert "90.0%" in out
    
    p = tmp_path / "out.json"
    BenchmarkReporter.save_run(res1, str(p))
    assert os.path.exists(str(p))
