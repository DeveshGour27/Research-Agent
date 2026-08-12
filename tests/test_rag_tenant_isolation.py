from __future__ import annotations

import pytest

from app.agent.contracts import AgentRequest
from app.agent.execution_context import AgentExecutionContext
from app.agent.specialized.rag_agent import RAGAgent
from app.exceptions import AgentExecutionError
from app.retriever import Retriever
from rag.chunking import Document as ChunkDoc
from rag.embeddings import MockEmbeddingProvider
from rag.reranker import NoopReranker

def test_vector_store_isolation():
    r = Retriever(embedding_provider_name="mock", reranker=NoopReranker())
    
    # Ingest documents for User A and User B
    doc_a = ChunkDoc(id="a1", content="apple banana A", source="src", metadata={})
    doc_b = ChunkDoc(id="b1", content="apple banana B", source="src", metadata={})
    
    r.index_documents([doc_a], user_id="UserA")
    r.index_documents([doc_b], user_id="UserB")
    
    # Query vector search directly using User A
    # Since mock provider produces identical vectors for identical words, both docs would match globally
    res_a = r._vector_search("apple", top_k=5, user_id="UserA")
    assert len(res_a) == 1
    assert "A" in res_a[0][2]["text"]
    
    res_b = r._vector_search("apple", top_k=5, user_id="UserB")
    assert len(res_b) == 1
    assert "B" in res_b[0][2]["text"]

def test_bm25_isolation():
    r = Retriever(embedding_provider_name="mock", reranker=NoopReranker())
    
    doc_a = ChunkDoc(id="a1", content="apple banana A", source="src", metadata={})
    doc_b = ChunkDoc(id="b1", content="apple banana B", source="src", metadata={})
    
    r.index_documents([doc_a], user_id="UserA")
    r.index_documents([doc_b], user_id="UserB")
    
    res_a = r._bm25_search("apple", top_k=5, user_id="UserA")
    assert len(res_a) == 1
    assert "A" in res_a[0][2]["text"]
    
    res_b = r._bm25_search("apple", top_k=5, user_id="UserB")
    assert len(res_b) == 1
    assert "B" in res_b[0][2]["text"]

def test_hybrid_isolation():
    r = Retriever(embedding_provider_name="mock", reranker=NoopReranker())
    
    doc_a = ChunkDoc(id="a1", content="apple banana A", source="src", metadata={})
    doc_b = ChunkDoc(id="b1", content="apple banana B", source="src", metadata={})
    
    r.index_documents([doc_a], user_id="UserA")
    r.index_documents([doc_b], user_id="UserB")
    
    res_a = r.retrieve("apple", user_id="UserA")
    assert len(res_a) == 1
    assert "A" in res_a[0].metadata["text"]
    
    res_b = r.retrieve("apple", user_id="UserB")
    assert len(res_b) == 1
    assert "B" in res_b[0].metadata["text"]

def test_rag_agent_isolation():
    r = Retriever(embedding_provider_name="mock", reranker=NoopReranker())
    
    doc_a = ChunkDoc(id="a1", content="apple banana A", source="src", metadata={})
    doc_b = ChunkDoc(id="b1", content="apple banana B", source="src", metadata={})
    
    r.index_documents([doc_a], user_id="UserA")
    r.index_documents([doc_b], user_id="UserB")
    
    agent = RAGAgent(retriever=r)
    
    ctx_a = AgentExecutionContext(task="Test", user_id="UserA")
    req_a = AgentRequest(input_text="apple", context=ctx_a)
    res_a = agent.execute(req_a)
    assert res_a.success is True
    assert "apple banana A" in res_a.output
    assert "apple banana B" not in res_a.output

def test_rag_agent_missing_identity():
    r = Retriever(embedding_provider_name="mock", reranker=NoopReranker())
    agent = RAGAgent(retriever=r)
    
    req = AgentRequest(input_text="apple")
    
    with pytest.raises(AgentExecutionError, match="user_id is required"):
        agent.execute(req)

def test_persistence_restore_isolation(monkeypatch):
    import rag.vector_store as vs_mod
    orig_chroma_cls = vs_mod.ChromaVectorStore
    
    store = {}
    class FakeChromaCollection:
        def __init__(self, name): self.name = name
        def add(self, ids, embeddings, metadatas, documents):
            coll = store.setdefault(self.name, {})
            for i, _id in enumerate(ids):
                coll[_id] = {"embedding": embeddings[i], "metadata": metadatas[i], "document": documents[i]}
        def query(self, query_embeddings=None, n_results=5, where=None, include=None):
            coll = store.get(self.name, {})
            ids, dists, mds, docs = [], [], [], []
            uid = where.get("user_id") if where else None
            for _id, entry in coll.items():
                if uid and entry["metadata"].get("user_id") != uid:
                    continue
                ids.append(_id); dists.append(0.0); mds.append(entry["metadata"]); docs.append(entry["document"])
            return {"ids": [ids], "distances": [dists], "metadatas": [mds], "documents": [docs]}
        def get(self, where=None, include=None):
            coll = store.get(self.name, {})
            ids, mds, docs = [], [], []
            uid = where.get("user_id") if where else None
            for _id, entry in coll.items():
                if uid and entry["metadata"].get("user_id") != uid:
                    continue
                ids.append(_id); mds.append(entry["metadata"]); docs.append(entry["document"])
            return {"ids": ids, "metadatas": mds, "documents": docs}

    class FakeClient:
        def get_collection(self, name=None):
            if name not in store: raise Exception("not found")
            return FakeChromaCollection(name)
        def create_collection(self, name=None):
            store.setdefault(name, {})
            return FakeChromaCollection(name)

    def fake_chroma_ctor(dim, persist_directory=None, collection_name=None, provider_id=None):
        class Wrapper:
            def __init__(self, dim):
                self._client = FakeClient()
                try: self._collection = self._client.get_collection(name=collection_name)
                except Exception: self._collection = self._client.create_collection(name=collection_name)
            def add_batch(self, ids, vectors, metadatas=None):
                mds = []
                for i, _id in enumerate(ids):
                    md = dict(metadatas[i] if metadatas and i < len(metadatas) else {})
                    md.setdefault("_chunk_id", _id)
                    mds.append(md)
                self._collection.add(ids=ids, embeddings=vectors.tolist(), metadatas=mds, documents=[m.get("text", "") for m in mds])
            def query(self, vector, top_k=5, *, user_id):
                res = self._collection.query(query_embeddings=[list(vector)], n_results=top_k, where={"user_id": user_id}, include=["metadatas", "distances", "documents", "ids"])
                raw_ids = res.get("ids", [[]])[0] if isinstance(res.get("ids", None), list) else []
                dists = res.get("distances", [[]])[0] if isinstance(res.get("distances", None), list) else []
                mds = res.get("metadatas", [[]])[0] if isinstance(res.get("metadatas", None), list) else []
                out = []
                for i, _id in enumerate(raw_ids):
                    out.append((mds[i].get("_chunk_id", _id), max(0.0, 1.0 - float(dists[i])), mds[i]))
                return out
            def all_items(self, user_id):
                res = self._collection.get(where={"user_id": user_id}, include=["metadatas", "documents"])
                ids = res.get("ids", [])
                mds = res.get("metadatas", [])
                return [(mds[i].get("_chunk_id", ids[i]), mds[i]) for i in range(len(ids))]
        return Wrapper(dim)

    monkeypatch.setattr("rag.vector_store.ChromaVectorStore", fake_chroma_ctor)
    from app.config import settings
    monkeypatch.setattr(settings, "environment", settings.environment.__class__.PRODUCTION)

    r1 = Retriever(embedding_provider_name="mock")
    doc_a = ChunkDoc(id="a1", content="apple A", source="s", metadata={})
    doc_b = ChunkDoc(id="b1", content="apple B", source="s", metadata={})
    r1.index_documents([doc_a], user_id="UserA")
    r1.index_documents([doc_b], user_id="UserB")

    r2 = Retriever(embedding_provider_name="mock")
    # Query with A, it should restore A's BM25 but NOT B's
    res_a = r2.retrieve("apple", user_id="UserA")
    assert len(res_a) == 1
    assert "A" in res_a[0].metadata["text"]

    # Query with B
    res_b = r2.retrieve("apple", user_id="UserB")
    assert len(res_b) == 1
    assert "B" in res_b[0].metadata["text"]

    monkeypatch.setattr("rag.vector_store.ChromaVectorStore", orig_chroma_cls)
