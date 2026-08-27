"""Policy enforcement for evidence sufficiency and retrieval retries."""

from typing import List

from app.config import settings
from app.logger import get_logger
from app.reflection.contracts import ReflectionDecision, ReflectionResult
from app.reflection.evaluator import ReflectionEvaluator
from rag.hybrid_search import Retrieved

logger = get_logger(__name__)

class EvidenceSufficiencyPolicy:
    """Deterministic policy bounding reflection outcomes."""

    def __init__(self, evaluator: ReflectionEvaluator | None = None) -> None:
        self.evaluator = evaluator or ReflectionEvaluator()

    def evaluate(
        self, 
        query: str, 
        retrieved_docs: List[Retrieved], 
        current_attempt: int
    ) -> ReflectionResult:
        """
        Evaluate evidence and enforce deterministic constraints.
        
        Args:
            query: The user's query.
            retrieved_docs: The candidate documents retrieved so far.
            current_attempt: 0-indexed count of retrieval attempts.
        """
        # Enforce max attempts
        if current_attempt > settings.max_retrieval_retries:
            logger.warning("Retry budget exhausted. Forcing INSUFFICIENT_EVIDENCE.")
            res = ReflectionResult(
                decision=ReflectionDecision.INSUFFICIENT_EVIDENCE,
                confidence=1.0,
                reason="Maximum retry budget exhausted.",
                retry_retrieval=False
            )
            from app.observability.events import ReflectionEvent
            ReflectionEvent(None, None, None, None, reflection_type=res.decision.name, success=True).emit()
            return res
            
        result = self.evaluator.evaluate(query, retrieved_docs)

        # Policy checks
        if result.decision == ReflectionDecision.ACCEPT:
            if not retrieved_docs:
                logger.warning("Reflection returned ACCEPT on empty evidence. Overriding to INSUFFICIENT_EVIDENCE.")
                res = ReflectionResult(
                    decision=ReflectionDecision.INSUFFICIENT_EVIDENCE,
                    confidence=1.0,
                    reason="Policy override: Cannot ACCEPT empty evidence.",
                    retry_retrieval=result.retry_retrieval
                )
                from app.observability.events import ReflectionEvent
                ReflectionEvent(None, None, None, None, reflection_type=res.decision.name, success=True).emit()
                return res

        if result.decision == ReflectionDecision.RETRY_RETRIEVAL or result.retry_retrieval:
            # Check if budget permits retry
            if current_attempt >= settings.max_retrieval_retries:
                logger.info("Reflection requested retry but budget exhausted. Yielding INSUFFICIENT_EVIDENCE.")
                res = ReflectionResult(
                    decision=ReflectionDecision.INSUFFICIENT_EVIDENCE,
                    confidence=result.confidence,
                    reason="Budget exhausted for retries.",
                    retry_retrieval=False
                )
                from app.observability.events import ReflectionEvent
                ReflectionEvent(None, None, None, None, reflection_type=res.decision.name, success=True).emit()
                return res
            # Normal retry allowed
            res = ReflectionResult(
                decision=ReflectionDecision.RETRY_RETRIEVAL,
                confidence=result.confidence,
                reason=result.reason,
                retry_retrieval=True
            )
            from app.observability.events import ReflectionEvent
            ReflectionEvent(None, None, None, None, reflection_type=res.decision.name, success=True).emit()
            return res

        from app.observability.events import ReflectionEvent
        ReflectionEvent(None, None, None, None, reflection_type=result.decision.name, success=True).emit()
        return result
