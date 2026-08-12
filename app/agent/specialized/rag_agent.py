"""RAGAgent implementation."""

from __future__ import annotations

import json

from app.agent.contracts import (
    AgentCapabilities,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
)
from app.agent.state import AgentState
from app.exceptions import AgentExecutionError
from app.retriever import Retriever


class RAGAgent(BaseAgent):
    """Specialized agent for internal RAG retrieval."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._identity = AgentIdentity(
            name="rag_agent",
            version="1.0.0",
            description="Specialized agent for retrieving information from the internal knowledge base.",
        )
        self._capabilities = AgentCapabilities(
            tool_use=False,
            memory=False,
            multi_turn=False,
            retrieval=True,
            task_types=frozenset({"rag_search"}),
        )
        self._retriever = retriever or Retriever()

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def execute(self, request: AgentRequest) -> AgentResult:
        normalized_input = request.input_text.strip()
        if not normalized_input:
            raise AgentExecutionError(
                "Agent request input_text must not be empty.",
                request=request,
            )

        user_id = request.context.user_id if request.context else None
        if not user_id:
            raise AgentExecutionError(
                "user_id is required for RAG retrieval.",
                request=request,
            )

        try:
            from app.reflection import EvidenceSufficiencyPolicy, ReflectionDecision
            from app.config import settings
            from app.retriever import _rewrite_query
            
            current_query = normalized_input
            
            if not getattr(settings, "reflection_enabled", True):
                results = self._retriever.retrieve(query=current_query, user_id=user_id)
                serialized_results = []
                for res in results:
                    serialized_results.append({
                        "chunk_id": res.chunk_id,
                        "score": round(res.score, 4),
                        "text": res.metadata.get("text", ""),
                    })
                
                if not serialized_results:
                    output = "No relevant documents found."
                else:
                    output = json.dumps(serialized_results, indent=2, ensure_ascii=False)
            else:
                policy = EvidenceSufficiencyPolicy()
                attempt = 0
                
                while attempt <= settings.max_retrieval_retries:
                    if attempt > 0:
                        current_query = _rewrite_query(current_query)
                        
                    results = self._retriever.retrieve(query=current_query, user_id=user_id)
                    reflection = policy.evaluate(normalized_input, results, current_attempt=attempt)
                    
                    if reflection.decision == ReflectionDecision.ACCEPT:
                        serialized_results = []
                        for res in results:
                            serialized_results.append({
                                "chunk_id": res.chunk_id,
                                "score": round(res.score, 4),
                                "text": res.metadata.get("text", ""),
                            })
                        
                        if not serialized_results:
                            output = "No relevant documents found."
                        else:
                            output = json.dumps(serialized_results, indent=2, ensure_ascii=False)
                        break
                        
                    elif reflection.decision == ReflectionDecision.INSUFFICIENT_EVIDENCE:
                        output = "No relevant documents found."
                        break
                        
                    elif reflection.decision == ReflectionDecision.RETRY_RETRIEVAL:
                        attempt += 1
                else:
                    output = "No relevant documents found."
                
            success = True
            error = None
        except Exception as e:
            output = None
            success = False
            error = AgentExecutionError(
                "RAG retrieval failed.",
                request=request,
                details={"error_type": type(e).__name__, "message": str(e)},
            )

        state = AgentState(
            finished=success,
            final_answer=output,
        )

        return AgentResult(
            request=request,
            state=state,
            output=output,
            success=success,
            context=request.context,
            error=error,
        )
