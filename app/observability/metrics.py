"""Deterministic, in-memory metrics registry for Phase 5.8 Step 4."""

from __future__ import annotations

import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from app.logger import get_logger

logger = get_logger(__name__)


class MetricsRegistry:
    """
    Lightweight, deterministic, in-memory metrics registry.
    Consumes observability events to produce execution-quality metrics.
    Safe for concurrent use. Never raises exceptions to the caller.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        
        # execution state tracking
        # run_id -> {"status": str, "root_agent": str, "start_time": datetime}
        self._run_states: dict[str, dict[str, Any]] = {}
        
        # handoff tracking
        # f"{run_id}:{target_task_type}" -> datetime
        self._handoff_starts: dict[str, datetime] = {}
        
        # counters
        self._counters: dict[str, Any] = {
            "executions": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "partial_success": 0,
                "cancelled": 0,
                "timed_out": 0
            },
            "retries": {
                "total": 0
            },
            "handoffs": {
                "total": 0,
                "successful": 0,
                "failed": 0
            },
            "agent_failures": 0,
            "step_failures": 0
        }
        
        # latency collections (in milliseconds)
        self._latencies: dict[str, list[float]] = {
            "execution": [],
            "step": [], # Kept empty as deterministic step start events don't exist
            "handoff": []
        }
        
        # agent specific metrics
        # agent_name -> {"executions": int, "failures": int, "retries": int}
        self._agent_metrics: dict[str, dict[str, int]] = {}

    def record_event(self, event: Any) -> None:
        """
        Consume a typed observability event and update metrics.
        Failure isolation: swallows all errors to avoid breaking execution.
        """
        try:
            with self._lock:
                self._process_event(event)
        except Exception as e:
            # Observability must NEVER become a control-flow dependency.
            logger.debug(f"Metrics recording failed: {e}")

    def snapshot(self) -> dict[str, Any]:
        """
        Return an immutable snapshot of the current metrics state.
        Calculates averages for latency arrays.
        """
        with self._lock:
            snap = deepcopy(self._counters)
            
            snap["latency"] = {}
            for key, times in self._latencies.items():
                count = len(times)
                total_ms = sum(times)
                avg_ms = total_ms / count if count > 0 else 0.0
                
                snap["latency"][key] = {
                    "count": count,
                    "total_ms": round(total_ms, 2),
                    "average_ms": round(avg_ms, 2)
                }
                
            snap["agents"] = deepcopy(self._agent_metrics)
            
            return snap

    def _process_event(self, event: Any) -> None:
        event_type = getattr(event, "event_type", None)
        if not event_type:
            return
            
        event_data = getattr(event, "event_data", {})
        run_id = getattr(event, "run_id", None)
        
        # Parse timestamp safely
        timestamp_str = getattr(event, "timestamp", None)
        timestamp = datetime.now(timezone.utc)
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                pass

        if event_type == "AgentExecutionStarted":
            agent = event_data.get("agent_name")
            is_root = getattr(event, "parent_span_id", None) is None
            
            if run_id and run_id not in self._run_states:
                self._run_states[run_id] = {
                    "status": "started",
                    "root_agent": agent,
                    "start_time": timestamp
                }
                self._counters["executions"]["total"] += 1
                
            if agent:
                if agent not in self._agent_metrics:
                    self._agent_metrics[agent] = {"executions": 0, "failures": 0, "retries": 0}
                self._agent_metrics[agent]["executions"] += 1

        elif event_type == "AgentExecutionCompleted":
            agent = event_data.get("agent_name")
            success = event_data.get("success", False)
            output = event_data.get("output")
            
            status = "failed"
            if success:
                status = "partial_success" if output == "partial_success" else "successful"
                
            if status == "failed":
                self._counters["agent_failures"] += 1
                if agent:
                    if agent not in self._agent_metrics:
                        self._agent_metrics[agent] = {"executions": 0, "failures": 0, "retries": 0}
                    self._agent_metrics[agent]["failures"] += 1

            if run_id and run_id in self._run_states:
                state = self._run_states[run_id]
                # Only transition root execution from 'started'
                if state["status"] == "started" and state["root_agent"] == agent:
                    state["status"] = status
                    self._counters["executions"][status] += 1
                    
                    start_time = state.get("start_time")
                    if start_time:
                        latency = (timestamp - start_time).total_seconds() * 1000.0
                        self._latencies["execution"].append(latency)

        elif event_type == "TimeoutCancellation":
            reason = event_data.get("reason", "")
            status = "cancelled" if "cancelled" in reason else "timed_out"
            
            if run_id and run_id in self._run_states:
                state = self._run_states[run_id]
                if state["status"] == "started":
                    state["status"] = status
                    self._counters["executions"][status] += 1

        elif event_type == "PlanStepCompleted":
            status = event_data.get("status")
            if status == "failed":
                self._counters["step_failures"] += 1

        elif event_type == "RetryAttempted":
            self._counters["retries"]["total"] += 1

        elif event_type == "HandoffInitiated":
            self._counters["handoffs"]["total"] += 1
            target = event_data.get("target_task_type")
            if run_id and target:
                self._handoff_starts[f"{run_id}:{target}"] = timestamp

        elif event_type == "HandoffResolved":
            success = event_data.get("success", False)
            if success:
                self._counters["handoffs"]["successful"] += 1
            else:
                self._counters["handoffs"]["failed"] += 1
                
            target = event_data.get("target_task_type")
            if run_id and target:
                key = f"{run_id}:{target}"
                start_time = self._handoff_starts.pop(key, None)
                if start_time:
                    latency = (timestamp - start_time).total_seconds() * 1000.0
                    self._latencies["handoff"].append(latency)
