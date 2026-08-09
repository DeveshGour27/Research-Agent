from __future__ import annotations

import json
from typing import List

import pytest

from app.config import settings
from rag.embeddings import get_embedding_provider, SentenceTransformersProvider, MockEmbeddingProvider
from rag.vector_store import get_vector_store, ChromaVectorStore
from app.retriever import Retriever
from rag.reranker import NeuralReranker, LexicalReranker, NoopReranker


def test_embedding_default_in_production(monkeypatch):
    # Simulate production environment and fake SentenceTransformersProvider
    monkeypatch.setattr(settings, "environment", settings.environment.__class__.PRODUCTION)

    class FakeST:
        def __init__(self, model_name="m"):
            pass

        def embed_texts(self, texts: List[str]):
            import numpy as np

            return np.zeros((len(texts), 16), dtype=float)

    monkeypatch.setattr("rag.embeddings.SentenceTransformersProvider", lambda model_name=None: FakeST())

    provider = get_embedding_provider(None)
    assert not isinstance(provider, MockEmbeddingProvider)


def test_vector_store_factory_and_persistence(monkeypatch):
    # Simulate production env and fake ChromaVectorStore to allow persistence test
    monkeypatch.setattr(settings, "environment", settings.environment.__class__.PRODUCTION)
    # Create a fake ChromaVectorStore that persists to a module-level dict
    store_data = {}

    class FakeChroma:
        def __init__(self, dim, persist_directory=None, collection_name=None):
            self.collection = collection_name or "default"

        def add_batch(self, ids, vectors, metadatas=None):
            store_data.setdefault(self.collection, {})
            for i, _id in enumerate(ids):
                store_data[self.collection][_id] = {"embedding": vectors[i], "metadata": metadatas[i]}

        def query(self, vector, top_k=5):
            # naive: return stored ids
            coll = store_data.get(self.collection, {})
            out = []
            for _id, entry in coll.items():
                out.append((_id, 1.0, entry.get("metadata", {})))
            return out[:top_k]

    # Monkeypatch constructor
    monkeypatch.setattr("rag.vector_store.ChromaVectorStore", lambda dim, persist_directory=None, collection_name=None: FakeChroma(dim, persist_directory, collection_name))

    # Use Retriever with mock embeddings to avoid ST requirement
    r = Retriever(embedding_provider_name="mock")
    docs = [
        # simple small documents
        type("D", (), {"id": "d1", "content": "apple banana", "source": "s1", "metadata": {}})(),
    ]
    r.index_documents(docs)
    # Create a new retriever and ensure documents are available via vector store
    r2 = Retriever(embedding_provider_name="mock")
    # The vector store class will be instantiated anew but backing store_data is shared
    # Query using retriever internal vector store by generating embedding for 'apple'
    qres = r2.retrieve("apple")
    assert qres, "Expected retrieval results from fake Chroma store"


def test_reranker_wiring_and_fallback(monkeypatch):
    # Ensure settings affect retriever.reranker
    monkeypatch.setattr(settings, "reranking_enabled", True)
    monkeypatch.setattr(settings, "reranker_type", "neural")
    r = Retriever(embedding_provider_name="mock")
    assert isinstance(r.reranker, NeuralReranker)

    # Disable reranking
    monkeypatch.setattr(settings, "reranking_enabled", False)
    r2 = Retriever(embedding_provider_name="mock")
    assert isinstance(r2.reranker, NoopReranker)

    # Test neural reranker fallback to lexical when LLM fails: monkeypatch create_chat_provider to raise
    from app.llm.factory import create_chat_provider

    monkeypatch.setattr("app.llm.factory.create_chat_provider", lambda cfg: (_ for _ in ()).throw(Exception("fail")))
    nr = NeuralReranker(top_k=5)
    items = [("id1", 0.1, {"text": "alpha beta"}), ("id2", 0.2, {"text": "beta gamma alpha"})]
    out = nr.rerank("alpha", items)
    # Should fallback to lexical ordering (BM25-like) — ensure output is list of tuples
    assert isinstance(out, list)
    assert all(isinstance(t, tuple) for t in out)
