import json
from typing import Dict, Any, List
from app.evaluation.benchmark_runner import BenchmarkRunResult

class BenchmarkReporter:
    @staticmethod
    def save_run(result: BenchmarkRunResult, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
            
    @staticmethod
    def generate_comparison_report(runs: List[BenchmarkRunResult]) -> str:
        if not runs:
            return "No runs to compare."
            
        report = ["# Model Comparison Report\n"]
        
        # Extract model names
        models = []
        for r in runs:
            meta = r.agent_config
            model_name = meta.get("model_id", r.run_id[:8])
            models.append(model_name)
            
        col_width = 35
        header = f"{'Metric':<25}" + "".join([f"{m:>{col_width}}" for m in models])
        report.append(header)
        report.append("-" * len(header))
        
        # Format: (Name, Key, Formatter, HigherIsBetter)
        metrics = [
            ("Success Rate", "success_rate", lambda x: f"{x*100:.1f}%" if x is not None else "N/A", True),
            ("Mean Correctness", "mean_judge_correctness", lambda x: f"{x:.1f}" if x is not None else "N/A", True),
            ("Mean Coverage", "mean_concept_coverage", lambda x: f"{x*100:.1f}%" if x is not None else "N/A", True),
            ("Mean Latency (s)", "mean_latency", lambda x: f"{x:.2f}s" if x is not None else "N/A", False),
        ]
        
        for name, key, fmt, higher_is_better in metrics:
            row = f"{name:<25}"
            baseline_val = runs[0].summary.get(key)
            
            for i, r in enumerate(runs):
                val = r.summary.get(key)
                formatted_val = fmt(val)
                
                if i == 0 or val is None or baseline_val is None:
                    row += f"{formatted_val:>{col_width}}"
                else:
                    diff = val - baseline_val
                    if abs(diff) < 1e-5:
                        status = "Unchanged"
                        sign = ""
                    else:
                        is_improvement = (diff > 0) if higher_is_better else (diff < 0)
                        status = "Improved" if is_improvement else "Regressed"
                        sign = "+" if diff > 0 else ""
                    
                    if "Rate" in name or "Coverage" in name:
                        diff_str = f"{sign}{diff*100:.1f}%"
                    elif "Latency" in name:
                        diff_str = f"{sign}{diff:.2f}s"
                    else:
                        diff_str = f"{sign}{diff:.1f}"
                        
                    cell = f"{formatted_val} [{diff_str} {status}]"
                    row += f"{cell:>{col_width}}"
            report.append(row)
            
        report.append("\n## Token Usage (Averages)")
        row_tokens = f"{'Mean Total Tokens':<25}"
        
        baseline_tokens = None
        for i, r in enumerate(runs):
            total_tokens = [c["operational"]["total_tokens"] for c in r.case_results]
            mean_tok = sum(total_tokens) / len(total_tokens) if total_tokens else 0
            
            if i == 0:
                baseline_tokens = mean_tok
                row_tokens += f"{int(mean_tok):>{col_width}}"
            else:
                diff = mean_tok - baseline_tokens
                if abs(diff) < 1e-5:
                    status = "Unchanged"
                    sign = ""
                else:
                    status = "Improved" if diff < 0 else "Regressed"
                    sign = "+" if diff > 0 else ""
                
                diff_str = f"{sign}{int(diff)}"
                cell = f"{int(mean_tok)} [{diff_str} {status}]"
                row_tokens += f"{cell:>{col_width}}"
                
        report.append(row_tokens)
        
        return "\n".join(report)
