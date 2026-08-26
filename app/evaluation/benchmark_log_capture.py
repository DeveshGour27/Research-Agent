import logging
from typing import List, Dict, Any

class EvaluationLogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        
    def reset(self):
        self.records.clear()
        
    def get_token_usage(self) -> Dict[str, int]:
        inp, out, tot = 0, 0, 0
        for r in self.records:
            if "Model request completed" in r.getMessage():
                inp += getattr(r, "input_tokens", 0)
                out += getattr(r, "output_tokens", 0)
                tot += getattr(r, "total_tokens", 0)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}
        
    def get_llm_calls(self) -> int:
        return sum(1 for r in self.records if "Model request completed" in r.getMessage())

    def get_tool_calls(self) -> int:
        # Check event logs for ToolCalledEvent or similar
        return sum(1 for r in self.records if hasattr(r, "event") and r.event.get("event_type") == "ToolCalledEvent")
        
    def get_retrieval_calls(self) -> int:
        return sum(1 for r in self.records if hasattr(r, "event") and r.event.get("event_type") == "MemoryRetrievedEvent")
