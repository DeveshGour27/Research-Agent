import pytest
from typing import List, Optional

from app.config import Settings
from rag.hybrid_search import Retrieved, combine_scores
from rag.reranker import LexicalReranker, NeuralReranker
from app.retriever import _evaluate_retrieval


def test_top_k_hierarchy_validation():
    # Valid
    settings = Settings(top_k_retrieval=50, top_k_rerank=25, top_k_final=10)
    assert settings.top_k_retrieval == 50

    # Invalid
    with pytest.raises(ValueError, match="Invalid top-k hierarchy"):
        Settings(top_k_retrieval=10, top_k_rerank=50, top_k_final=10)

    with pytest.raises(ValueError, match="Input should be greater than 0"):
        Settings(top_k_retrieval=50, top_k_rerank=50, top_k_final=0)



def test_retrieved_dataclass_score_semantics():
    r = Retrieved(
        chunk_id="1",
        score=0.9,
        source_score=0.1,
        vector_score=0.2,
        metadata={"a": 1}
    )
    # Defaults must be backward compatible
    assert r.hybrid_score == 0.0
    assert r.rerank_score is None

    r2 = Retrieved(
        chunk_id="2",
        score=0.8,
        source_score=0.5,
        vector_score=0.5,
        metadata={},
        hybrid_score=0.8,
        rerank_score=0.9
    )
    assert r2.hybrid_score == 0.8
    assert r2.rerank_score == 0.9


def test_lexical_reranker_tie_breaking():
    reranker = LexicalReranker()
    # Mock items: (chunk_id, hybrid_score, metadata)
    # Give them identical text so BM25 score is identical.
    # The tie-breaker should use hybrid_score DESC -> chunk_id ASC.
    items = [
        ("id2", 0.5, {"text": "apple", "hybrid_score": 0.5}),
        ("id1", 0.9, {"text": "apple", "hybrid_score": 0.9}),
        ("id3", 0.5, {"text": "apple", "hybrid_score": 0.5}),
    ]
    
    # query is "apple"
    results = reranker.rerank("apple", items)
    
    # All get same BM25 score.
    # Sorting sequence: rerank_score DESC (BM25 score), hybrid_score DESC, chunk_id ASC
    # So order should be: id1 (0.9), id2 (0.5), id3 (0.5)
    assert results[0][0] == "id1"
    assert results[1][0] == "id2"
    assert results[2][0] == "id3"


def test_evaluate_retrieval_metrics():
    # evaluate_retrieval computes recall@k and mrr
    r_list = [
        Retrieved("id1", 0.9, 0.0, 0.0, {}),
        Retrieved("id2", 0.8, 0.0, 0.0, {}),
        Retrieved("id3", 0.7, 0.0, 0.0, {}),
    ]
    
    # test 1: perfect ranking
    metrics = _evaluate_retrieval(r_list, ["id1"], k=3)
    assert metrics["mrr"] == 1.0
    assert metrics["recall@k"] == 1.0
    
    # test 2: relevant at rank 2
    metrics2 = _evaluate_retrieval(r_list, ["id2"], k=3)
    assert metrics2["mrr"] == 0.5
    assert metrics2["recall@k"] == 1.0
    
    # test 3: no relevant in top k
    metrics3 = _evaluate_retrieval(r_list, ["id4"], k=3)
    assert metrics3["mrr"] == 0.0
    assert metrics3["recall@k"] == 0.0


class MockProvider:
    def __init__(self, output: str):
        self.output = output
        
    def generate(self, messages):
        class MockResponse:
            content = self.output
        return MockResponse()


def test_neural_reranker_fallback_deterministic_sort(monkeypatch):
    import app.llm.factory
    # NeuralReranker shouldn't fail if we give it bad JSON, it should fallback to lexical reranker.
    # If the text doesn't contain the keyword, lexical reranker score is 0.
    # It will fallback to hybrid score DESC -> chunk_id ASC.
    monkeypatch.setattr(app.llm.factory, "create_model_gateway", lambda s: MockProvider("Bad Output"))
    
    nr = NeuralReranker()
    
    items = [
        ("id3", 0.1, {"text": "no match", "hybrid_score": 0.1}),
        ("id1", 0.9, {"text": "no match", "hybrid_score": 0.9}),
        ("id2", 0.5, {"text": "no match", "hybrid_score": 0.5}),
    ]
    
    results = nr.rerank("keyword", items)
    
    # BM25 scores will all be 0.
    # LexicalReranker sorts by: (-bm25_score, -hybrid_score, chunk_id).
    assert results[0][0] == "id1"
    assert results[1][0] == "id2"
    assert results[2][0] == "id3"
    
def test_neural_reranker_missing_score(monkeypatch):
    import app.llm.factory
    # LLM outputs ordering but no numeric scores: ["id2", "id1", "id3"]
    monkeypatch.setattr(app.llm.factory, "create_model_gateway", lambda s: MockProvider('["id2", "id1", "id3"]'))
    nr = NeuralReranker()
    items = [
        ("id1", 0.9, {"text": "A", "hybrid_score": 0.9}),
        ("id2", 0.5, {"text": "B", "hybrid_score": 0.5}),
        ("id3", 0.1, {"text": "C", "hybrid_score": 0.1}),
    ]
    results = nr.rerank("query", items)
    
    # results should maintain the LLM's ordering because NeuralReranker appends in order.
    # NeuralReranker itself does not resort, it just validates output.
    assert results[0][0] == "id2"
    assert results[1][0] == "id1"
    assert results[2][0] == "id3"
    
    # And the scores assigned should be explicitly None because the LLM provided none.
    assert results[0][1] is None
    assert results[1][1] is None
    assert results[2][1] is None


def test_adversarial_keyword_trap(monkeypatch):
    from app.retriever import Retriever
    from app.config import settings
    
    # Build a synthetic dataset with a keyword trap.
    docs = [
        {"id": "trap", "text": "This document contains the EXACT KEYWORDS but means something else.", "score": 0.9},
        {"id": "relevant", "text": "This discusses the actual semantic concept intended by the query.", "score": 0.6},
        {"id": "distractor", "text": "Just another unrelated document.", "score": 0.2},
    ]
    
    class MockRetriever(Retriever):
        def _vector_search(self, query, top_k, user_id):
            # Semantic search favors the relevant one slightly, but keyword trap is also high
            return [("relevant", 0.8, {"text": docs[1]["text"]}), ("trap", 0.7, {"text": docs[0]["text"]}), ("distractor", 0.2, {"text": docs[2]["text"]})]
            
        def _bm25_search(self, query, top_k, user_id):
            # BM25 heavily favors the keyword trap
            return [("trap", 1.0, {"text": docs[0]["text"]}), ("distractor", 0.0, {"text": docs[2]["text"]}), ("relevant", 0.0, {"text": docs[1]["text"]})]
            
        def _restore_bm25_from_vector_store(self, user_id):
            pass

    import app.llm.factory
    # Reranker sees the text and places the semantically relevant one first!
    monkeypatch.setattr(app.llm.factory, "create_model_gateway", lambda s: MockProvider('[{"id": "relevant", "score": 0.99}, {"id": "trap", "score": 0.1}]'))
    
    # Force defaults
    monkeypatch.setattr(settings, "top_k_retrieval", 3)
    monkeypatch.setattr(settings, "top_k_rerank", 3)
    monkeypatch.setattr(settings, "top_k_final", 3)
    
    retriever = MockRetriever(reranker=NeuralReranker())
    
    # 1. Baseline: Hybrid only (No reranking)
    baseline_results = retriever.retrieve("EXACT KEYWORDS", "u1", rerank=False)
    
    # Hybrid gives Trap a huge boost (1.0 bm25 + 0.7 vec) vs Relevant (0.0 bm25 + 0.8 vec)
    # So baseline MRR should put Trap first.
    baseline_metrics = _evaluate_retrieval(baseline_results, ["relevant"], k=3)
    
    # 2. Reranked: Hybrid + Reranking
    reranked_results = retriever.retrieve("EXACT KEYWORDS", "u1", rerank=True)
    
    reranked_metrics = _evaluate_retrieval(reranked_results, ["relevant"], k=3)
    
    # Verify the keyword trap causes baseline to rank the relevant document lower (rank 2)
    assert baseline_metrics["mrr"] < 1.0
    
    # Verify the reranker fixed it
    assert reranked_metrics["mrr"] == 1.0
    
    # This explicitly satisfies the requirement to measure and assert improvement
    # only where explicitly expected (in a keyword trap!).
    assert reranked_metrics["mrr"] > baseline_metrics["mrr"]
    
    # Verify score semantics in final output
    assert reranked_results[0].chunk_id == "relevant"
    assert reranked_results[0].rerank_score == 0.99
    assert reranked_results[0].score == 0.99  # authoritative score mimics rerank score
    assert reranked_results[0].hybrid_score > 0.0  # preserved original hybrid score
