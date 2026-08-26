from typing import List, Optional, Set
import re

class DeterministicMetrics:
    @staticmethod
    def exact_match(prediction: str, reference: Optional[str]) -> Optional[float]:
        if reference is None:
            return None
        return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

    @staticmethod
    def concept_coverage(prediction: str, required_concepts: List[str]) -> Optional[float]:
        if not required_concepts:
            return None
        prediction_lower = prediction.lower()
        found = sum(1 for concept in required_concepts if concept.lower() in prediction_lower)
        return found / len(required_concepts)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> Optional[float]:
        if not relevant_ids:
            return None
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        intersection = retrieved_k.intersection(relevant_set)
        return len(intersection) / len(relevant_set)

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> Optional[float]:
        if not relevant_ids:
            return None
        if not retrieved_ids[:k]:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        intersection = retrieved_k.intersection(relevant_set)
        return len(intersection) / len(retrieved_k)

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> Optional[float]:
        if not relevant_ids:
            return None
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
