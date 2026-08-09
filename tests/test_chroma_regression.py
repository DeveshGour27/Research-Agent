from __future__ import annotations

import numpy as np
from app.config import settings
from app.config import Environment
from app.retriever import Retriever
from rag.documents import ingest_from_string


def test_chroma_persistence_across_retrievers(monkeypatch):
    """
    Regression test: Index a document with one Retriever instance and ensure
    a new Retriever instance can retrieve it when using the ChromaVectorStore
    backend. This test uses a fake ChromaVectorStore to avoid external deps.
    """
    # Simulate production environment so get_vector_store picks Chroma
    monkeypatch.setattr(settings, "environment", settings.environment.__class__.PRODUCTION)

    # Shared backing store to simulate persistent Chroma
    store = {}

    class FakeChromaCollection:
        def __init__(self, name):
            self.name = name

        def add(self, ids, embeddings, metadatas, documents):
            coll = store.setdefault(self.name, {})
            for i, _id in enumerate(ids):
                coll[_id] = {
                    "embedding": np.array(embeddings[i], dtype=float),
                    "metadata": metadatas[i],
                    "document": documents[i],
                }

        def query(self, query_embeddings=None, n_results=5, include=None):
            # Return chroma-like nested lists for ids, distances, metadatas, documents
            coll = store.get(self.name, {})
            ids = list(coll.keys())
            distances = [0.0 for _ in ids]
            metadatas = [entry["metadata"] for entry in coll.values()]
            documents = [entry["document"] for entry in coll.values()]
            return {"ids": [ids], "distances": [distances], "metadatas": [metadatas], "documents": [documents]}

    class FakeClient:
        def __init__(self, persist_directory=None):
            pass

        def get_collection(self, name=None):
            # If collection missing, raise to mimic chroma behavior
            if name not in store:
                raise Exception("not found")
            return FakeChromaCollection(name)

        def create_collection(self, name=None):
            # Create collection and return a collection wrapper
            store.setdefault(name, {})
            return FakeChromaCollection(name)

        def persist(self):
            # fake persist is a no-op
            return

    # Monkeypatch the ChromaVectorStore internals to use our fake client/collection
    import rag.vector_store as vs_mod

    orig_chroma_cls = vs_mod.ChromaVectorStore

    def fake_chroma_ctor(dim, persist_directory=None, collection_name=None):
        # Create a light wrapper object matching the required interface
        class Wrapper:
            def __init__(self, dim, persist_directory=None, collection_name=None):
                self.dim = dim
                self._client = FakeClient(persist_directory)
                # create collection if absent
                try:
                    self._collection = self._client.get_collection(name=collection_name)
                except Exception:
                    self._collection = self._client.create_collection(name=collection_name)

            def add(self, id, vector, metadata=None):
                self.add_batch([id], np.expand_dims(np.asarray(vector), 0), metadatas=[metadata or {}])

            def add_batch(self, ids, vectors, metadatas=None):
                metadatas = metadatas or [{} for _ in ids]
                documents = [md.get("text", "") for md in metadatas]
                # Enrich metadata to mimic real ChromaVectorStore.add_batch behavior
                enriched = []
                for i, _id in enumerate(ids):
                    md = dict(metadatas[i] if i < len(metadatas) else {})
                    md.setdefault("_chunk_id", _id)
                    enriched.append(md)
                self._collection.add(ids=ids, embeddings=vectors.tolist(), metadatas=enriched, documents=documents)

            def query(self, vector, top_k=5):
                # Delegate to fake collection; return list of tuples matching ChromaVectorStore.query output
                res = self._collection.query(query_embeddings=[np.asarray(vector).tolist()], n_results=top_k, include=["metadatas", "distances", "documents", "ids"])
                # res is chroma-like dict with nested lists; mimic ChromaVectorStore.query post-processing
                raw_ids = res.get("ids", [[]])[0] if isinstance(res.get("ids", None), list) else []
                distances = res.get("distances", [[]])[0] if isinstance(res.get("distances", None), list) else []
                metadatas = res.get("metadatas", [[]])[0] if isinstance(res.get("metadatas", None), list) else []
                out = []
                for i, _id in enumerate(raw_ids):
                    dist = float(distances[i]) if i < len(distances) else 0.0
                    score = max(0.0, 1.0 - dist)
                    md = metadatas[i] if i < len(metadatas) else {}
                    canonical_id = md.get("_chunk_id") if isinstance(md, dict) and md.get("_chunk_id") else _id
                    out.append((canonical_id, score, md))
                return out

            def persist(self):
                return

            def get(self, id):
                return None

            def all_ids(self):
                return list(store.get(collection_name, {}).keys())

            def size(self):
                return len(store.get(collection_name, {}))

        return Wrapper(dim, persist_directory=persist_directory, collection_name=collection_name)

    monkeypatch.setattr("rag.vector_store.ChromaVectorStore", fake_chroma_ctor)

    # Index a document with Retriever A
    r1 = Retriever(embedding_provider_name="mock")
    doc = ingest_from_string("/tmp/test_rfc.txt", "Access-token rotation interval is 30 days.", filename="test_rfc.txt")
    r1.index_documents([doc])

    # Create a new Retriever instance (simulating a restart) and ensure it can retrieve
    r2 = Retriever(embedding_provider_name="mock")
    results = r2.retrieve("What is the required access-token rotation interval in RFC-TEST-9700?", top_k=5)

    assert results, "Expected at least one retrieval result"
    # Ensure returned metadata contains the original text
    texts = [r.metadata.get("text", "") for r in results]
    assert any("Access-token rotation interval is 30 days." in t for t in texts)

    # Clean up monkeypatch
    monkeypatch.setattr("rag.vector_store.ChromaVectorStore", orig_chroma_cls)
