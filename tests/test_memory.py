"""Unit tests for persistent user/chat-scoped memory."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.agent import Agent
from app.exceptions import MemoryReadError
from app.llm.models import ModelResponse
from app.llm.base import (
    ChatMessage,
    ChatResponse,
    LLMProvider,
    LLMResponse,
)
from app.memory import JsonFileMemoryStore, MemoryExtractor
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry


class ScriptedProvider(LLMProvider):
    @property
    def provider_id(self) -> str: return "scripted"

    def __init__(
        self,
        responses: Sequence[LLMResponse],
        generation_responses: Sequence[ModelResponse] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._generation_responses = list(generation_responses or [])
        self.calls = 0
        self.generation_calls = 0

    def generate(self, request, model_id="test") -> LLMResponse:
        from app.llm.models import TaskType
        if getattr(request, "task_type", None) == TaskType.REFLECTION:
            if self.generation_calls >= len(self._generation_responses):
                raise AssertionError("No scripted response left for this provider generation call.")
            response = self._generation_responses[self.generation_calls]
            self.generation_calls += 1
            return response
        else:
            if self.calls >= len(self._responses):
                raise AssertionError("No scripted response left for this provider call.")
            response = self._responses[self.calls]
            self.calls += 1
            return response


def _store(path: Path) -> JsonFileMemoryStore:
    return JsonFileMemoryStore(storage_path=path)


def _registry_with_calculator() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


def test_user_creation_and_retrieval(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    created = store.create_user(email="user@example.com")
    fetched = store.get_user(created.user_id)

    assert fetched is not None
    assert fetched.user_id == created.user_id
    assert fetched.email == "user@example.com"


def test_memory_persists_across_store_reload(tmp_path: Path) -> None:
    storage_file = tmp_path / "memory.json"
    store_a = _store(storage_file)
    user = store_a.create_user(email="persist@example.com")
    store_a.save_memory(user.user_id, "remember this")

    store_b = _store(storage_file)
    memories = store_b.get_memories(user.user_id)

    assert [memory.content for memory in memories] == ["remember this"]


def test_user_retrieves_own_memory(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="owner@example.com")
    store.save_memory(user.user_id, "my memory")

    memories = store.get_memories(user.user_id)

    assert len(memories) == 1
    assert memories[0].content == "my memory"
    assert memories[0].user_id == user.user_id


def test_user_cannot_retrieve_another_users_memory(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user_a = store.create_user(email="a@example.com")
    user_b = store.create_user(email="b@example.com")

    store.save_memory(user_a.user_id, "a secret")
    store.save_memory(user_b.user_id, "b secret")

    memories_for_a = store.get_memories(user_a.user_id)

    assert [memory.content for memory in memories_for_a] == ["a secret"]
    assert all(
        memory.user_id == user_a.user_id
        for memory in memories_for_a
    )


def test_multiple_chats_for_one_user_are_separate(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="chats@example.com")

    store.create_chat(user.user_id, chat_id="chat-1")
    store.create_chat(user.user_id, chat_id="chat-2")

    store.save_chat_messages(
        user.user_id,
        "chat-1",
        [{"role": "user", "content": "chat one"}],
    )

    store.save_chat_messages(
        user.user_id,
        "chat-2",
        [{"role": "user", "content": "chat two"}],
    )

    chat_one_messages = store.get_chat_messages(
        user.user_id,
        "chat-1",
    )

    chat_two_messages = store.get_chat_messages(
        user.user_id,
        "chat-2",
    )

    assert chat_one_messages == [
        {"role": "user", "content": "chat one"}
    ]

    assert chat_two_messages == [
        {"role": "user", "content": "chat two"}
    ]


def test_chat_history_isolation_across_users(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user_a = store.create_user(email="isolate-a@example.com")
    user_b = store.create_user(email="isolate-b@example.com")
    chat_id = "shared-name"

    store.save_chat_messages(
        user_a.user_id,
        chat_id,
        [{"role": "user", "content": "from A"}],
    )

    store.save_chat_messages(
        user_b.user_id,
        chat_id,
        [{"role": "user", "content": "from B"}],
    )

    assert store.get_chat_messages(
        user_a.user_id,
        chat_id,
    ) == [{"role": "user", "content": "from A"}]

    assert store.get_chat_messages(
        user_b.user_id,
        chat_id,
    ) == [{"role": "user", "content": "from B"}]


def test_reset_clears_chat_but_preserves_user_memory(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="reset@example.com")

    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                content="Before reset response.",
            ),
            LLMResponse(
                model="fake-model",
                content="After reset response.",
            ),
        ]
    )

    agent = Agent(
        provider,
        _registry_with_calculator(),
        user_id=user.user_id,
        chat_id="chat-reset",
        memory_store=store,
    )

    agent.run("Before reset")
    agent.save_user_memory("keep this")
    agent.reset()

    state = agent.run("After reset")

    memories = store.get_memories(user.user_id)
    persisted_chat = store.get_chat_messages(
        user.user_id,
        "chat-reset",
    )

    assert [memory.content for memory in memories] == ["keep this"]

    assert {
        "role": "assistant",
        "content": "Before reset response.",
    } not in persisted_chat

    assert persisted_chat[-1] == {
        "role": "assistant",
        "content": "After reset response.",
    }

    assert state.final_answer == "After reset response."


def test_relevant_memory_is_retrieved(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="relevant@example.com")

    store.save_memory(
        user.user_id,
        "User prefers Python examples.",
    )

    store.save_memory(
        user.user_id,
        "User likes concise replies.",
    )

    retrieved = store.retrieve_relevant_memories(
        user.user_id,
        query="Can you give Python code?",
        top_k=3,
    )

    assert [
        memory.content
        for memory in retrieved
    ] == ["User prefers Python examples."]


def test_irrelevant_memory_is_excluded(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(
        email="irrelevant-memory@example.com"
    )

    store.save_memory(
        user.user_id,
        "User prefers Java examples.",
    )

    retrieved = store.retrieve_relevant_memories(
        user.user_id,
        query="What is the weather today?",
        top_k=3,
    )

    assert retrieved == []


def test_memory_ranking_prefers_more_relevant_match(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="ranking@example.com")

    store.save_memory(
        user.user_id,
        "User prefers Python and pytest examples.",
    )

    store.save_memory(
        user.user_id,
        "User prefers Python.",
    )

    retrieved = store.retrieve_relevant_memories(
        user.user_id,
        query="Need Python pytest help",
        top_k=2,
    )

    assert [
        memory.content
        for memory in retrieved
    ] == [
        "User prefers Python and pytest examples.",
        "User prefers Python.",
    ]


def test_memory_retrieval_top_k_works(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="topk@example.com")

    store.save_memory(user.user_id, "User likes Python.")
    store.save_memory(user.user_id, "User likes pytest.")
    store.save_memory(user.user_id, "User likes linters.")

    retrieved = store.retrieve_relevant_memories(
        user.user_id,
        query="python pytest lint",
        top_k=2,
    )

    assert len(retrieved) == 2


def test_memory_retrieval_respects_user_isolation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")
    user_a = store.create_user(
        email="retrieve-a@example.com"
    )
    user_b = store.create_user(
        email="retrieve-b@example.com"
    )

    store.save_memory(
        user_a.user_id,
        "User A likes tea.",
    )

    store.save_memory(
        user_b.user_id,
        "User B likes coffee.",
    )

    retrieved_a = store.retrieve_relevant_memories(
        user_a.user_id,
        query="coffee tea",
        top_k=5,
    )

    assert [
        memory.content
        for memory in retrieved_a
    ] == ["User A likes tea."]


# ---------------------------------------------------------------------------
# Phase 3.5 — Memory Update & Conflict Resolution
# ---------------------------------------------------------------------------


def test_update_memory_preserves_id_and_created_at(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="update@example.com")

    original = store.save_memory(
        user.user_id,
        "User prefers Python examples.",
    )

    updated = store.update_memory(
        user.user_id,
        original.memory_id,
        "User prefers Rust examples.",
    )

    assert updated.memory_id == original.memory_id
    assert updated.user_id == original.user_id
    assert updated.created_at == original.created_at
    assert updated.content == "User prefers Rust examples."


def test_updated_memory_persists_after_store_reload(
    tmp_path: Path,
) -> None:
    storage_file = tmp_path / "memory.json"

    store_a = _store(storage_file)
    user = store_a.create_user(email="update-persist@example.com")

    original = store_a.save_memory(
        user.user_id,
        "User uses OpenAI.",
    )

    store_a.update_memory(
        user.user_id,
        original.memory_id,
        "User uses Groq.",
    )

    store_b = _store(storage_file)

    memories = store_b.get_memories(user.user_id)

    assert len(memories) == 1
    assert memories[0].memory_id == original.memory_id
    assert memories[0].content == "User uses Groq."
    assert memories[0].created_at == original.created_at


def test_update_does_not_modify_unrelated_memories(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="unrelated@example.com")

    target = store.save_memory(
        user.user_id,
        "User prefers Python examples.",
    )

    unrelated = store.save_memory(
        user.user_id,
        "User prefers concise replies.",
    )

    updated = store.update_memory(
        user.user_id,
        target.memory_id,
        "User prefers Rust examples.",
    )

    memories = store.get_memories(user.user_id)

    assert updated.content == "User prefers Rust examples."

    assert [
        memory.content
        for memory in memories
    ] == [
        "User prefers Rust examples.",
        "User prefers concise replies.",
    ]

    assert memories[1].memory_id == unrelated.memory_id


def test_invalid_memory_id_cannot_modify_any_memory(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="invalid-id@example.com")

    original = store.save_memory(
        user.user_id,
        "User prefers Python.",
    )

    try:
        store.update_memory(
            user.user_id,
            "does-not-exist",
            "User prefers Rust.",
        )
    except MemoryReadError:
        pass
    else:
        raise AssertionError(
            "Expected MemoryReadError for invalid memory ID."
        )

    memories = store.get_memories(user.user_id)

    assert len(memories) == 1
    assert memories[0].memory_id == original.memory_id
    assert memories[0].content == "User prefers Python."


def test_user_cannot_update_another_users_memory(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")

    user_a = store.create_user(email="owner-a@example.com")
    user_b = store.create_user(email="owner-b@example.com")

    memory_a = store.save_memory(
        user_a.user_id,
        "User A prefers Python.",
    )

    try:
        store.update_memory(
            user_b.user_id,
            memory_a.memory_id,
            "User B prefers Rust.",
        )
    except MemoryReadError:
        pass
    else:
        raise AssertionError(
            "Expected MemoryReadError for cross-user update."
        )

    memories_a = store.get_memories(user_a.user_id)
    memories_b = store.get_memories(user_b.user_id)

    assert [
        memory.content
        for memory in memories_a
    ] == ["User A prefers Python."]

    assert memories_b == []


def test_upsert_exact_duplicate_does_not_create_duplicate(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="duplicate@example.com")

    original = store.save_memory(
        user.user_id,
        "User prefers Python examples.",
    )

    result = store.upsert_memory(
        user.user_id,
        "User prefers Python examples.",
    )

    memories = store.get_memories(user.user_id)

    assert len(memories) == 1
    assert result.memory_id == original.memory_id
    assert memories[0].content == original.content


def test_upsert_related_memory_updates_existing_memory(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="upsert@example.com")

    original = store.save_memory(
        user.user_id,
        "User prefers Python examples.",
    )

    result = store.upsert_memory(
        user.user_id,
        "User prefers Python code examples.",
    )

    memories = store.get_memories(user.user_id)

    assert len(memories) == 1
    assert result.memory_id == original.memory_id
    assert memories[0].content == "User prefers Python code examples."
    assert memories[0].created_at == original.created_at


def test_extractor_parses_create_operation() -> None:
    provider = ScriptedProvider(
        [],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content=(
                    '{"should_store": true, '
                    '"operation": "create", '
                    '"memory": "User prefers Rust examples.", '
                    '"memory_id": ""}'
                ),
            )
        ],
    )

    extractor = MemoryExtractor(provider)

    result = extractor.extract(
        user_input="From now on, use Rust in examples.",
        assistant_response="Understood.",
        existing_memories=[],
    )

    assert result.should_store is True
    assert result.operation == "create"
    assert result.memory == "User prefers Rust examples."
    assert result.memory_id == ""


def test_extractor_parses_update_operation() -> None:
    provider = ScriptedProvider(
        [],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content=(
                    '{"should_store": true, '
                    '"operation": "update", '
                    '"memory": "User prefers Rust examples.", '
                    '"memory_id": "memory-123"}'
                ),
            )
        ],
    )

    extractor = MemoryExtractor(provider)

    from app.memory import Memory

    existing = [
        Memory(
            memory_id="memory-123",
            user_id="user-123",
            content="User prefers Python examples.",
            created_at="2026-01-01T00:00:00+00:00",
        )
    ]

    result = extractor.extract(
        user_input="I now prefer Rust examples.",
        assistant_response="Understood.",
        existing_memories=existing,
    )

    assert result.should_store is True
    assert result.operation == "update"
    assert result.memory == "User prefers Rust examples."
    assert result.memory_id == "memory-123"


def test_extractor_parses_ignore_operation() -> None:
    provider = ScriptedProvider(
        [],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content=(
                    '{"should_store": false, '
                    '"operation": "ignore", '
                    '"memory": "", '
                    '"memory_id": ""}'
                ),
            )
        ],
    )

    extractor = MemoryExtractor(provider)

    result = extractor.extract(
        user_input="What is 25 * 30?",
        assistant_response="750",
        existing_memories=[],
    )

    assert result.should_store is False
    assert result.operation == "ignore"
    assert result.memory == ""
    assert result.memory_id == ""


def test_extractor_rejects_invalid_update_memory_id() -> None:
    provider = ScriptedProvider(
        [],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content=(
                    '{"should_store": true, '
                    '"operation": "update", '
                    '"memory": "User prefers Rust.", '
                    '"memory_id": "unknown-id"}'
                ),
            )
        ],
    )

    extractor = MemoryExtractor(provider)

    from app.memory import Memory

    existing = [
        Memory(
            memory_id="memory-123",
            user_id="user-123",
            content="User prefers Python.",
            created_at="2026-01-01T00:00:00+00:00",
        )
    ]

    result = extractor.extract(
        user_input="I prefer Rust now.",
        assistant_response="Understood.",
        existing_memories=existing,
    )

    assert result.should_store is False
    assert result.operation == "ignore"
    assert result.memory == ""
    assert result.memory_id == ""


def test_extractor_invalid_json_fails_safely() -> None:
    provider = ScriptedProvider(
        [],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content="this is not json",
            )
        ],
    )

    extractor = MemoryExtractor(provider)

    result = extractor.extract(
        user_input="Remember that I use Rust.",
        assistant_response="Okay.",
        existing_memories=[],
    )

    assert result.should_store is False
    assert result.operation == "ignore"
    assert result.memory == ""
    assert result.memory_id == ""


def test_extractor_invalid_operation_fails_safely() -> None:
    provider = ScriptedProvider(
        [],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content=(
                    '{"should_store": true, '
                    '"operation": "delete", '
                    '"memory": "User prefers Rust.", '
                    '"memory_id": ""}'
                ),
            )
        ],
    )

    extractor = MemoryExtractor(provider)

    result = extractor.extract(
        user_input="I prefer Rust.",
        assistant_response="Understood.",
        existing_memories=[],
    )

    assert result.should_store is False
    assert result.operation == "ignore"


def test_agent_applies_update_operation(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="agent-update@example.com")

    original = store.save_memory(
        user.user_id,
        "User uses OpenAI.",
    )

    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                content="The project now uses Groq.",
            )
        ],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content=(
                    '{"should_store": true, '
                    '"operation": "update", '
                    '"memory": "User uses Groq.", '
                    f'"memory_id": "{original.memory_id}"}}'
                ),
            )
        ],
    )

    agent = Agent(
        provider,
        _registry_with_calculator(),
        user_id=user.user_id,
        chat_id="update-chat",
        memory_store=store,
    )

    agent.run("I switched the project from OpenAI to Groq.")

    memories = store.get_memories(user.user_id)

    assert len(memories) == 1
    assert memories[0].memory_id == original.memory_id
    assert memories[0].content == "User uses Groq."
    assert memories[0].created_at == original.created_at


def test_agent_applies_create_operation(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="agent-create@example.com")

    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                content="Understood.",
            )
        ],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content=(
                    '{"should_store": true, '
                    '"operation": "create", '
                    '"memory": "User prefers concise explanations.", '
                    '"memory_id": ""}'
                ),
            )
        ],
    )

    agent = Agent(
        provider,
        _registry_with_calculator(),
        user_id=user.user_id,
        chat_id="create-chat",
        memory_store=store,
    )

    agent.run("Please keep explanations concise from now on.")

    memories = store.get_memories(user.user_id)

    assert len(memories) == 1
    assert memories[0].content == (
        "User prefers concise explanations."
    )


def test_agent_ignores_ignore_operation(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="agent-ignore@example.com")

    existing = store.save_memory(
        user.user_id,
        "User prefers concise explanations.",
    )

    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                content="750",
            )
        ],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content=(
                    '{"should_store": false, '
                    '"operation": "ignore", '
                    '"memory": "", '
                    '"memory_id": ""}'
                ),
            )
        ],
    )

    agent = Agent(
        provider,
        _registry_with_calculator(),
        user_id=user.user_id,
        chat_id="ignore-chat",
        memory_store=store,
    )

    agent.run("What is 25 * 30?")

    memories = store.get_memories(user.user_id)

    assert len(memories) == 1
    assert memories[0].memory_id == existing.memory_id
    assert memories[0].content == existing.content


def test_agent_memory_update_failure_does_not_fail_run(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="agent-failure@example.com")

    provider = ScriptedProvider(
        [
            LLMResponse(
                model="fake-model",
                content="Normal answer.",
            )
        ],
        generation_responses=[
            ModelResponse(
                model="fake-model",
                content=(
                    '{"should_store": true, '
                    '"operation": "update", '
                    '"memory": "User prefers Rust.", '
                    '"memory_id": "invalid-memory-id"}'
                ),
            )
        ],
    )

    agent = Agent(
        provider,
        _registry_with_calculator(),
        user_id=user.user_id,
        chat_id="failure-chat",
        memory_store=store,
    )

    state = agent.run("I now prefer Rust.")

    assert state.final_answer == "Normal answer."
    assert store.get_memories(user.user_id) == []


def test_sensitivity_filter_redacts_credentials_and_secrets(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="secrets@example.com")

    # API key redaction
    mem1 = store.save_memory(user.user_id, "User key is sk-abcdefghijklmnopqrstuvwxyz123456")
    assert "sk-" not in mem1.content
    assert "[REDACTED_OPENAI_API_KEY]" in mem1.content

    # Bearer token redaction
    mem2 = store.save_memory(user.user_id, "Auth token is Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz")
    assert "Bearer" not in mem2.content or "[REDACTED_BEARER_TOKEN]" in mem2.content

    # Private key redaction
    private_key_content = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
    mem3 = store.save_memory(user.user_id, f"SSH key: {private_key_content}")
    assert "BEGIN RSA PRIVATE KEY" not in mem3.content
    assert "[REDACTED_PRIVATE_KEY]" in mem3.content

    # Password redaction
    mem4 = store.save_memory(user.user_id, "User password: MySuperSecretPassword123!")
    assert "MySuperSecretPassword123!" not in mem4.content
    assert "[REDACTED_SECRET]" in mem4.content


def test_memory_deletion_and_gdpr_right_to_be_forgotten(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="gdpr@example.com")

    m1 = store.save_memory(user.user_id, "Memory 1")
    m2 = store.save_memory(user.user_id, "Memory 2")
    m3 = store.save_memory(user.user_id, "Memory 3")

    assert len(store.get_memories(user.user_id)) == 3

    # Delete single memory
    deleted = store.delete_memory(user.user_id, m2.memory_id)
    assert deleted is True
    remaining = store.get_memories(user.user_id)
    assert len(remaining) == 2
    assert m2.memory_id not in [m.memory_id for m in remaining]

    # Deleting nonexistent memory returns False
    assert store.delete_memory(user.user_id, "nonexistent-id") is False

    # GDPR delete all user memories
    deleted_count = store.delete_user_memories(user.user_id)
    assert deleted_count == 2
    assert store.get_memories(user.user_id) == []


def test_memory_expiration_and_retention_purge(tmp_path: Path) -> None:
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="retention@example.com")

    # Save memory with TTL of 30 days (future)
    future_mem = store.save_memory(user.user_id, "Future memory", ttl_days=30)
    assert future_mem.expires_at is not None

    # Save memory already expired
    import datetime
    expired_timestamp = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat()
    raw_data = store._read_data()
    raw_data["memories"][user.user_id].append({
        "memory_id": "expired-mem-id",
        "user_id": user.user_id,
        "content": "Expired memory",
        "created_at": expired_timestamp,
        "expires_at": expired_timestamp,
    })
    store._write_data(raw_data)

    # get_memories should filter out expired
    active = store.get_memories(user.user_id)
    assert len(active) == 1
    assert active[0].memory_id == future_mem.memory_id

    # purge_expired_memories should remove it from underlying store
    purged = store.purge_expired_memories()
    assert purged == 1


def test_concurrent_memory_writes_are_thread_safe(tmp_path: Path) -> None:
    import concurrent.futures
    store = _store(tmp_path / "memory.json")
    user = store.create_user(email="threads@example.com")

    def _write_item(i: int):
        store.save_memory(user.user_id, f"Thread memory {i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_write_item, i) for i in range(25)]
        concurrent.futures.wait(futures)

    memories = store.get_memories(user.user_id)
    assert len(memories) == 25