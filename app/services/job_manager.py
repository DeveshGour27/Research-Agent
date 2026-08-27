"""Async job manager for background execution of research jobs."""

from __future__ import annotations

import asyncio
from typing import Callable

from sqlalchemy.orm import sessionmaker

from app.agent.supervisor import Supervisor
from app.agent.contracts import AgentRequest
from app.agent.execution_context import AgentExecutionContext
from app.db.repository import SQLJobRepository
from app.exceptions import AgentExecutionError, AgentCancellationError, AgentTimeoutError, InvalidStateTransitionError, AgentHITLPauseException
from app.logger import get_logger

import uuid

logger = get_logger(__name__)

class AsyncJobManager:
    """Manages the background execution of research jobs with bounded concurrency.
    
    This abstracts the synchronous Supervisor into an isolated thread, managing
    database state transitions safely outside the FastAPI event loop.
    
    Note: max_concurrent_jobs limits execution concurrency PER PROCESS.
    It does not create a global distributed concurrency limit.
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        supervisor_factory: Callable[[], Supervisor],
        max_concurrent_jobs: int = 10,
        job_timeout_seconds: float = 300.0,
        worker_id: str | None = None,
        job_heartbeat_interval_seconds: int = 10,
        job_stale_after_seconds: int = 60,
        job_recovery_poll_interval_seconds: int = 15,
        job_max_attempts: int = 3,
        hitl_service: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._supervisor_factory = supervisor_factory
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._job_timeout = job_timeout_seconds
        self._hitl_service = hitl_service
        
        self.worker_id = worker_id or uuid.uuid4().hex
        self._job_heartbeat_interval_seconds = job_heartbeat_interval_seconds
        self._job_stale_after_seconds = job_stale_after_seconds
        self._job_recovery_poll_interval_seconds = job_recovery_poll_interval_seconds
        self._job_max_attempts = job_max_attempts
        
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._job_contexts: dict[str, AgentExecutionContext] = {}
        self._is_shutting_down = False
        
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
            
        self._heartbeat_task = self._loop.create_task(self._heartbeat_loop())
        self._recovery_task = self._loop.create_task(self._recovery_loop())

    @property
    def is_ready(self) -> bool:
        """Read-only readiness indicator for health-check endpoints."""
        return not self._is_shutting_down

    def submit_job(self, job_id: str, user_id: str, goal: str) -> None:
        """Submit a job for background execution. Returns immediately."""
        if self._is_shutting_down:
            raise RuntimeError("JobManager is shutting down, cannot accept new jobs.")
            
        if job_id in self._running_tasks:
            return
            
        def _start_task():
            if job_id in self._running_tasks:
                return
            task = self._loop.create_task(self._run_job_with_lifecycle(job_id, user_id, goal))
            self._running_tasks[job_id] = task
            task.add_done_callback(lambda t: self._running_tasks.pop(job_id, None))
            task.add_done_callback(lambda t: self._job_contexts.pop(job_id, None))

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is self._loop:
            _start_task()
        else:
            self._loop.call_soon_threadsafe(_start_task)

    async def cancel_job(self, job_id: str, user_id: str) -> None:
        """Attempt to cancel a running or pending job safely."""
        def _cancel_in_db():
            with self._session_factory() as session:
                repo = SQLJobRepository(session)
                try:
                    res = repo.update_job_status(job_id, user_id, status="CANCELLED")
                    if res is None:
                        raise ValueError(f"Job {job_id} not found.")
                except InvalidStateTransitionError:
                    pass
                
        await asyncio.to_thread(_cancel_in_db)

        context = self._job_contexts.get(job_id)
        if context:
            context.mark_cancelled()

        task = self._running_tasks.get(job_id)
        if task and not task.done():
            task.cancel()

    async def shutdown(self) -> None:
        """Gracefully shutdown the manager and wait for tasks to finish or cancel."""
        self._is_shutting_down = True
        
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
            
        for job_id, task in list(self._running_tasks.items()):
            if not task.done():
                context = self._job_contexts.get(job_id)
                if context:
                    context.mark_cancelled()
                task.cancel()
        
        tasks_to_await = list(self._running_tasks.values())
        if self._heartbeat_task:
            tasks_to_await.append(self._heartbeat_task)
        if self._recovery_task:
            tasks_to_await.append(self._recovery_task)
            
        if tasks_to_await:
            await asyncio.gather(*tasks_to_await, return_exceptions=True)

    async def _heartbeat_loop(self) -> None:
        while not self._is_shutting_down:
            try:
                await asyncio.sleep(self._job_heartbeat_interval_seconds)
                active_job_ids = list(self._job_contexts.keys())
                if not active_job_ids:
                    continue
                    
                def _do_heartbeats(jobs: list[str]):
                    with self._session_factory() as session:
                        repo = SQLJobRepository(session)
                        for jid in jobs:
                            repo.heartbeat_job(jid, self.worker_id)
                            
                await asyncio.to_thread(_do_heartbeats, active_job_ids)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _recovery_loop(self) -> None:
        while not self._is_shutting_down:
            try:
                await asyncio.sleep(self._job_recovery_poll_interval_seconds)
                def _do_recovery() -> list[str]:
                    with self._session_factory() as session:
                        repo = SQLJobRepository(session)
                        recovered = repo.recover_stale_jobs(
                            stale_threshold_seconds=self._job_stale_after_seconds,
                            max_attempts=self._job_max_attempts,
                        )
                        if self._hitl_service:
                            self._hitl_service.expire_requests()
                        pending = repo.poll_pending_jobs(limit=10)
                        return list(set(recovered + pending))
                
                recovered_ids = await asyncio.to_thread(_do_recovery)
                if recovered_ids:
                    logger.info(f"Discovered {len(recovered_ids)} pending/stale jobs for execution.")
                    def _fetch_jobs(jids: list[str]):
                        with self._session_factory() as session:
                            from sqlalchemy import select
                            from app.db.models import Job
                            res = []
                            for jid in jids:
                                job = session.execute(select(Job).where(Job.job_id == jid)).scalar_one_or_none()
                                if job:
                                    res.append((job.job_id, job.user_id, job.goal))
                            return res
                    jobs = await asyncio.to_thread(_fetch_jobs, recovered_ids)
                    for jid, uid, goal in jobs:
                        from app.observability.events import RetryAttemptedEvent
                        try:
                            RetryAttemptedEvent(
                                trace_id=None,
                                run_id=None,
                                span_id=None,
                                parent_span_id=None,
                                error_type="StaleJobRecovery",
                                attempt_number=1
                            ).emit()
                        except Exception as e:
                            logger.error(f"Failed to emit RetryAttemptedEvent: {e}")
                        self.submit_job(jid, uid, goal)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")

    async def _run_job_with_lifecycle(self, job_id: str, user_id: str, goal: str) -> None:
        """Internal coroutine that handles semaphore, timeout, and state transitions."""
        async with self._semaphore:
            if self._is_shutting_down:
                return

            def _claim():
                try:
                    with self._session_factory() as session:
                        repo = SQLJobRepository(session)
                        return repo.claim_job(job_id, self.worker_id)
                except Exception as e:
                    logger.error(f"Failed to claim job {job_id}: {e}")
                    return False
            
            claimed = await asyncio.to_thread(_claim)
            if not claimed:
                logger.info(f"Worker {self.worker_id} failed to claim job {job_id} (already claimed/cancelled)")
                return

            context = AgentExecutionContext(task=goal, metadata={"job_id": job_id})
            self._job_contexts[job_id] = context
            request = AgentRequest(input_text=goal, context=context, metadata={"job_id": job_id})
            supervisor = self._supervisor_factory()

            def _update_result(status: str, result: str | None = None, error_message: str | None = None):
                try:
                    with self._session_factory() as session:
                        repo = SQLJobRepository(session)
                        repo.update_job_result(
                            job_id=job_id,
                            user_id=user_id,
                            status=status,
                            result=result,
                            error_message=error_message,
                            worker_id=self.worker_id,
                        )
                except Exception as e:
                    logger.error(f"Failed to update job {job_id} result to {status}: {e}")

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(supervisor.execute, request),
                    timeout=self._job_timeout
                )
                
                await asyncio.to_thread(
                    _update_result,
                    status="COMPLETED",
                    result=str(result.output) if result.output else "Success"
                )

            except asyncio.TimeoutError:
                context.mark_timed_out()
                
                # Emit structured telemetry for timeout
                from app.observability.events import TimeoutCancellationEvent
                try:
                    TimeoutCancellationEvent(
                        trace_id=getattr(context, "trace_id", None),
                        run_id=getattr(context, "run_id", None),
                        span_id=None,
                        parent_span_id=None,
                        reason="Job Execution Timeout"
                    ).emit()
                except Exception as e:
                    logger.error(f"Failed to emit TimeoutCancellationEvent: {e}")
                    
                await asyncio.to_thread(
                    _update_result,
                    status="FAILED",
                    error_message="Execution timed out"
                )
            except asyncio.CancelledError:
                pass
            except (AgentExecutionError, AgentCancellationError, AgentTimeoutError) as e:
                if isinstance(e, AgentCancellationError):
                    def _mark_cancel():
                        try:
                            with self._session_factory() as session:
                                repo = SQLJobRepository(session)
                                try:
                                    repo.update_job_status(job_id, user_id, status="CANCELLED", worker_id=self.worker_id)
                                except InvalidStateTransitionError:
                                    pass
                        except Exception as e:
                            logger.error(f"Failed to mark job {job_id} as cancelled: {e}")
                    await asyncio.to_thread(_mark_cancel)
                else:
                    await asyncio.to_thread(
                        _update_result,
                        status="FAILED",
                        error_message=str(e)
                    )
            except AgentHITLPauseException as e:
                def _mark_wait():
                    try:
                        with self._session_factory() as session:
                            repo = SQLJobRepository(session)
                            try:
                                # It's waiting for human, worker goes away, worker_id = None
                                # But we also don't clear worker_id if we want to ensure it isn't picked up?
                                # Actually, WAITING_FOR_HUMAN shouldn't be picked up anyway.
                                repo.update_job_status(job_id, user_id, status="WAITING_FOR_HUMAN", worker_id=self.worker_id)
                                # Remove worker association so it can be picked up by any worker when approved
                                from sqlalchemy import update
                                from app.db.models import Job
                                session.execute(update(Job).where(Job.job_id == job_id).values(worker_id=None, heartbeat_at=None))
                                session.commit()
                            except InvalidStateTransitionError:
                                pass
                    except Exception as e:
                        logger.error(f"Failed to mark job {job_id} as waiting: {e}")
                await asyncio.to_thread(_mark_wait)
                logger.info("job_paused_for_hitl", extra={"job_id": job_id, "request_id": e.request_id})

            except Exception as e:
                logger.exception("Unexpected error in background job execution", extra={"job_id": job_id})
                await asyncio.to_thread(
                    _update_result,
                    status="FAILED",
                    error_message="Internal execution error"
                )
