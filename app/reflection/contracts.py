"""Contracts for the reflection module."""

from enum import Enum
from pydantic import BaseModel, Field

class ReflectionDecision(str, Enum):
    """Allowed decisions for evidence reflection."""
    ACCEPT = "ACCEPT"
    RETRY_RETRIEVAL = "RETRY_RETRIEVAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

class ReflectionResult(BaseModel):
    """Structured output for reflection evaluation."""
    decision: ReflectionDecision = Field(
        ..., 
        description="The decision made by the reflection evaluator."
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0.0 and 1.0."
    )
    reason: str = Field(
        ..., 
        description="Explanation for the decision."
    )
    retry_retrieval: bool = Field(
        default=False, 
        description="Whether a retry is explicitly requested by the model."
    )
