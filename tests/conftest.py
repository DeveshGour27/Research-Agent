"""Global test configuration and fixtures."""

import pytest
from unittest.mock import patch
from app.config import settings

@pytest.fixture(autouse=True)
def disable_reflection_for_tests():
    """Disable Phase 9 reflection by default in tests to prevent LLM calls."""
    with patch.object(settings, "reflection_enabled", False):
        yield

import numpy as np

@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    """Globally mock SentenceTransformers to prevent model downloads and slow initializations during tests."""
    from rag.embeddings import SentenceTransformersProvider
    
    def mock_init(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        
    def mock_embed(self, texts):
        return np.zeros((len(list(texts)), 384), dtype=np.float32)
        
    with patch.object(SentenceTransformersProvider, '__init__', mock_init):
        with patch.object(SentenceTransformersProvider, 'embed_texts', mock_embed):
            yield
