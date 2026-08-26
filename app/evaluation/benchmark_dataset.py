from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json

@dataclass
class BenchmarkCase:
    id: str
    question: str
    expected_answer: Optional[str] = None
    reference_sources: List[str] = field(default_factory=list)
    required_concepts: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkCase":
        return cls(
            id=data["id"],
            question=data["question"],
            expected_answer=data.get("expected_answer"),
            reference_sources=data.get("reference_sources", []),
            required_concepts=data.get("required_concepts", []),
            difficulty=data.get("difficulty", "medium"),
            category=data.get("category", "general"),
            metadata=data.get("metadata", {})
        )

class BenchmarkDataset:
    def __init__(self, cases: List[BenchmarkCase]):
        self.cases = cases

    @classmethod
    def load_jsonl(cls, file_path: str) -> "BenchmarkDataset":
        cases = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                cases.append(BenchmarkCase.from_dict(data))
        return cls(cases)

    def __iter__(self):
        return iter(self.cases)
        
    def __len__(self):
        return len(self.cases)
