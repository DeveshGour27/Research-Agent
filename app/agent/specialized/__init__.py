"""Package for specialized worker agents."""

from app.agent.specialized.rag_agent import RAGAgent
from app.agent.specialized.web_agent import WebResearchAgent

__all__ = [
    "RAGAgent",
    "WebResearchAgent",
]
