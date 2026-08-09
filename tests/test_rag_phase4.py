from __future__ import annotations

import pytest

from rag.chunking import Document as ChunkDoc, chunk_document
from rag.embeddings import MockEmbeddingProvider
from rag.vector_store import InMemoryVectorStore
from rag.hybrid_search import combine_scores
from rag.reranker import LexicalReranker, NoopReranker
from app.retriever import Retriever


def test_chunking_basic():
    content = "This is a simple document. " * 200
    doc = ChunkDoc(id="doc1", content=content, source="test", metadata={"title": "t"})
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    # deterministic ids
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_mock_embeddings_and_vector_store_query():
    provider = MockEmbeddingProvider(dim=16)
    texts = ["apple banana fruit", "car truck vehicle"]
    emb = provider.embed_texts(texts)
    assert emb.shape == (2, 16)
    vs = InMemoryVectorStore(dim=16)
    vs.add_batch(["a", "b"], emb, metadatas=[{"text": texts[0]}, {"text": texts[1]}])
    q = provider.embed_texts(["apple fruit"][0:1])
    # q is 1d array
    qvec = provider.embed_texts(["apple fruit"])[0]
    res = vs.query(qvec, top_k=2)
    assert res[0][0] == "a"


def test_hybrid_combine_scores():
    bm25 = [("c1", 3.0, {}), ("c2", 1.0, {})]
    vec = [("c2", 0.9, {}), ("c3", 0.8, {})]
    combined = combine_scores(bm25, vec, bm25_weight=0.6, vector_weight=0.4, top_k=3)
    ids = [r.chunk_id for r in combined]
    assert "c1" in ids and "c2" in ids and "c3" in ids


def test_reranker_lexical():
    reranker = LexicalReranker()
    items = [("id1", 0.1, {"text": "alpha beta"}), ("id2", 0.2, {"text": "beta gamma alpha"})]
    out = reranker.rerank("alpha", items)
    assert out[0][0] in ("id1", "id2")


def test_retriever_end_to_end():
    # create small docs
    docs = [ChunkDoc(id=f"d{i}", content=("apple banana " * 50), source=f"s{i}", metadata={}) for i in range(3)]
    # Make one doc about cars
    docs.append(ChunkDoc(id="dcar", content=("car truck engine " * 50), source="s_car", metadata={}))
    r = Retriever(embedding_provider_name="mock", reranker=NoopReranker())
    r.index_documents(docs)
    # pick a relevant chunk id (first in index)
    assert r.chunk_index
    some_id = next(iter(r.chunk_index.keys()))
    results = r.agentic_retrieval_loop("apple", relevant_ids=[some_id], max_iters=2)
    assert isinstance(results, list)
    assert results[0].iteration == 1
    # stop_reason should be one of allowed
    assert results[0].stop_reason in ("sufficient_evidence", "max_iterations_reached", "no_new_evidence")
