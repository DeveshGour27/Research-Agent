from __future__ import annotations

import pytest
from rag.reranker import NeuralReranker


class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeProvider:
    def __init__(self, content):
        self._content = content

    def generate(self, messages):
        return FakeResp(self._content)


def make_items():
    return [
        ("id1", 0.1, {"text": "alpha beta"}),
        ("id2", 0.2, {"text": "beta gamma"}),
        ("id3", 0.05, {"text": "gamma delta"}),
    ]


@pytest.mark.parametrize("content,expected_order", [
    ('["id2","id1"]', ["id2", "id1", "id3"]),
    ('```json\n["id2","id1"]\n```', ["id2", "id1", "id3"]),
    ('  \n ["id2","id1"] \n ', ["id2", "id1", "id3"]),
    ('["id3","id2","id1"]', ["id3", "id2", "id1"]),
    ('["id2","id2"]', ["id2", "id1", "id3"]),
    ('["unknown","id2"]', ["id2", "id1", "id3"]),
])
def test_neural_reranker_parsing_and_ordering(monkeypatch, content, expected_order):
    monkeypatch.setattr("app.llm.factory.create_model_gateway", lambda cfg: FakeProvider(content))
    nr = NeuralReranker()
    items = make_items()
    out = nr.rerank("query", items)
    assert [o[0] for o in out] == expected_order


def test_neural_reranker_empty_or_invalid(monkeypatch):
    # empty -> fallback
    monkeypatch.setattr("app.llm.factory.create_model_gateway", lambda cfg: FakeProvider(""))
    nr = NeuralReranker()
    items = make_items()
    out = nr.rerank("query", items)
    assert set(o[0] for o in out) == set(i[0] for i in items)

    # invalid JSON -> fallback
    monkeypatch.setattr("app.llm.factory.create_model_gateway", lambda cfg: FakeProvider("not json"))
    out2 = nr.rerank("query", items)
    assert set(o[0] for o in out2) == set(i[0] for i in items)
