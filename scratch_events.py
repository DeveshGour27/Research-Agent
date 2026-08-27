import os

def insert_after(file_path, search_str, content_to_insert):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if search_str in content:
        parts = content.split(search_str)
        new_content = parts[0] + search_str + content_to_insert + parts[1]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Modified {file_path}")
    else:
        print(f"Not found in {file_path}")

events_to_add = """

@dataclass
class LLMCallEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, provider: str, model: str, latency_s: float, input_tokens: int, output_tokens: int, total_tokens: int, task_type: str, success: bool = True, error_info: str | None = None):
        super().__init__(
            event_type="LLMCallEvent",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"provider": provider, "model": model, "latency_s": latency_s, "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "task_type": task_type, "success": success, "error_info": error_info},
        )

@dataclass
class ToolCallEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, tool_name: str, arguments: dict[str, Any], success: bool = True, latency_s: float | None = None, error_info: str | None = None):
        super().__init__(
            event_type="ToolCallEvent",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"tool_name": tool_name, "arguments": arguments, "success": success, "latency_s": latency_s, "error_info": error_info},
        )

@dataclass
class RetrievalEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, query: str, top_k: int, result_count: int, success: bool = True, latency_s: float | None = None, error_info: str | None = None):
        super().__init__(
            event_type="RetrievalEvent",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"query": query, "top_k": top_k, "result_count": result_count, "success": success, "latency_s": latency_s, "error_info": error_info},
        )

@dataclass
class RerankEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, query: str, input_count: int, output_count: int, success: bool = True, latency_s: float | None = None, error_info: str | None = None):
        super().__init__(
            event_type="RerankEvent",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"query": query, "input_count": input_count, "output_count": output_count, "success": success, "latency_s": latency_s, "error_info": error_info},
        )

@dataclass
class ReflectionEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, reflection_type: str, success: bool = True, latency_s: float | None = None, error_info: str | None = None):
        super().__init__(
            event_type="ReflectionEvent",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"reflection_type": reflection_type, "success": success, "latency_s": latency_s, "error_info": error_info},
        )

@dataclass
class AgentStepEvent(BaseEvent):
    def __init__(self, trace_id: str | None, run_id: str | None, span_id: str | None, parent_span_id: str | None, agent_name: str, step_name: str, success: bool = True, latency_s: float | None = None, error_info: str | None = None):
        super().__init__(
            event_type="AgentStepEvent",
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_data={"agent_name": agent_name, "step_name": step_name, "success": success, "latency_s": latency_s, "error_info": error_info},
        )
"""

insert_after('app/observability/events.py', 'event_data={"query": query, "top_k": top_k, "result_count": result_count},\n        )', events_to_add)

# Update benchmark_log_capture.py
blc_path = 'app/evaluation/benchmark_log_capture.py'
with open(blc_path, 'r', encoding='utf-8') as f:
    blc = f.read()

blc = blc.replace(
    'if "Model request completed" in r.getMessage():\n                inp += getattr(r, "input_tokens", 0)\n                out += getattr(r, "output_tokens", 0)\n                tot += getattr(r, "total_tokens", 0)',
    'if hasattr(r, "event") and r.event.get("event_type") == "LLMCallEvent":\n                inp += r.event.get("event_data", {}).get("input_tokens", 0)\n                out += r.event.get("event_data", {}).get("output_tokens", 0)\n                tot += r.event.get("event_data", {}).get("total_tokens", 0)'
)

blc = blc.replace(
    'return sum(1 for r in self.records if "Model request completed" in r.getMessage())',
    'return sum(1 for r in self.records if hasattr(r, "event") and r.event.get("event_type") == "LLMCallEvent")'
)

blc = blc.replace(
    'return sum(1 for r in self.records if hasattr(r, "event") and r.event.get("event_type") == "ToolCalledEvent")',
    'return sum(1 for r in self.records if hasattr(r, "event") and r.event.get("event_type") in ("ToolCalledEvent", "ToolCallEvent"))'
)

blc = blc.replace(
    'return sum(1 for r in self.records if hasattr(r, "event") and r.event.get("event_type") == "MemoryRetrievedEvent")',
    'return sum(1 for r in self.records if hasattr(r, "event") and r.event.get("event_type") in ("MemoryRetrievedEvent", "RetrievalEvent"))'
)

with open(blc_path, 'w', encoding='utf-8') as f:
    f.write(blc)

print(f"Modified {blc_path}")
