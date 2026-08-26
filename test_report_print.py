from app.evaluation.benchmark_runner import BenchmarkRunResult
from app.evaluation.benchmark_reporters import BenchmarkReporter

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
res3 = BenchmarkRunResult(
    run_id="run3", timestamp=125, agent_config={"model_id": "ModelC"},
    summary={"success_rate": 0.7, "mean_judge_correctness": 4.2, "mean_concept_coverage": 0.8, "mean_latency": 2.5},
    case_results=[{"operational": {"total_tokens": 120}}]
)

print(BenchmarkReporter.generate_comparison_report([res1, res2, res3]))
