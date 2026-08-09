from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from app.llm.base import ChatMessage, ChatResponse, LLMProvider
from app.config import settings
from rag.reranker import NeuralReranker
from app.retriever import _rewrite_query, _decompose_query


@dataclass
class FakeProvider(LLMProvider):
    # simple scripted responses
    responses: Sequence[str]

    def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        # pop the first response-like behaviour
        text = self.responses[0] if self.responses else ""
        return ChatResponse(content=text, model="fake")


def test_neural_reranker_with_fake_provider(monkeypatch):
    # prepare a fake provider that returns a JSON ranking
    ranked = ["id2", "id1"]
    fake = FakeProvider(responses=[json.dumps(ranked)])

    monkeypatch.setattr("app.llm.factory.create_chat_provider", lambda cfg: fake)

    items = [("id1", 0.1, {"text": "alpha beta"}), ("id2", 0.2, {"text": "beta gamma alpha"})]
    nr = NeuralReranker(top_k=5)
    out = nr.rerank("alpha query", items)
    assert [o[0] for o in out][:2] == ranked


def test_llm_rewrite_and_decompose_with_fake_provider(monkeypatch):
    # rewrite: return a short normalized string
    fake_rewrite = FakeProvider(responses=["rewritten query"])
    monkeypatch.setattr("app.llm.factory.create_chat_provider", lambda cfg: fake_rewrite)
    out = _rewrite_query("   Rewrite THIS   Query")
    assert out == "rewritten query"

    # decomposition: return JSON
    fake_decomp = FakeProvider(responses=[json.dumps(["sub1", "sub2"])])
    monkeypatch.setattr("app.llm.factory.create_chat_provider", lambda cfg: fake_decomp)
    parts = _decompose_query("Complex query that needs decomposition")
    assert parts == ["sub1", "sub2"]
