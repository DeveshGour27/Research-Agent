"""Public surface for the Groq LLM integration.

Import :class:`GroqProvider` from here (``app.llm.groq``) or from the
legacy ``app.llm.groq_provider`` module — both resolve to the same class.
"""

from app.llm.groq_provider import GroqProvider  # noqa: F401

__all__ = ["GroqProvider"]
