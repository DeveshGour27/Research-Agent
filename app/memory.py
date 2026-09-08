"""Persistent user-scoped memory and chat history storage."""

from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from app.constants import MEMORY_DIR
from app.exceptions import MemoryReadError, MemoryWriteError
from app.llm.gateway import ModelGateway
from app.llm.models import ModelRequest, TaskType
from app.logger import get_logger

logger = get_logger(__name__)


MemoryOperation = Literal["create", "update", "ignore"]

_VALID_MEMORY_OPERATIONS: frozenset[str] = frozenset(
    {"create", "update", "ignore"}
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class User:
    """Application user identified by internal ``user_id``."""

    user_id: str
    email: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class Memory:
    """Long-term user memory item."""

    memory_id: str
    user_id: str
    content: str
    created_at: str
    expires_at: str | None = None


class SensitivityFilter:
    """Detects and redacts sensitive credentials, secrets, and private keys."""

    _SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE)),
        ("openai_api_key", re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b")),
        ("github_token", re.compile(r"\bgh[pousr]_[a-zA-Z0-9]{20,}\b")),
        ("bearer_token", re.compile(r"\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b", re.IGNORECASE)),
        ("generic_secret", re.compile(r"(?:api[_-]?key|auth[_-]?token|secret|password|passwd|pwd)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?", re.IGNORECASE)),
        ("credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    )

    @classmethod
    def contains_sensitive_data(cls, text: str) -> bool:
        """Return True if any secret or credential pattern is found."""
        return any(pattern.search(text) is not None for _, pattern in cls._SECRET_PATTERNS)

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Redact detected credentials and secrets with placeholder markers."""
        sanitized = text
        for name, pattern in cls._SECRET_PATTERNS:
            if name == "generic_secret":
                def _redact_generic(m: re.Match) -> str:
                    prefix = m.group(0)[:m.start(1) - m.start(0)]
                    suffix = m.group(0)[m.end(1) - m.start(0):]
                    return f"{prefix}[REDACTED_SECRET]{suffix}"
                sanitized = pattern.sub(_redact_generic, sanitized)
            else:
                sanitized = pattern.sub(f"[REDACTED_{name.upper()}]", sanitized)
        return sanitized


@dataclass(frozen=True, slots=True)
class MemoryExtractionResult:
    """Structured extractor output for automatic long-term memory decisions."""

    should_store: bool
    memory: str
    operation: MemoryOperation = "ignore"
    memory_id: str = ""


class MemoryStore(ABC):
    """Interface for user/chat-scoped persistent memory backends."""

    @abstractmethod
    def create_user(
        self,
        user_id: str | None = None,
        email: str | None = None,
    ) -> User:
        """Create a new user if missing and return it."""

    @abstractmethod
    def get_user(self, user_id: str) -> User | None:
        """Return user by ``user_id`` or ``None`` when missing."""

    @abstractmethod
    def get_or_create_user_by_email(self, email: str) -> User:
        """Resolve user by email or create a new mapped user."""

    @abstractmethod
    def create_chat(
        self,
        user_id: str,
        chat_id: str | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create a chat for ``user_id`` and return ``chat_id``."""

    @abstractmethod
    def get_chat_messages(
        self,
        user_id: str,
        chat_id: str,
    ) -> list[dict[str, Any]]:
        """Return chat messages for one user/chat pair."""

    @abstractmethod
    def save_chat_messages(
        self,
        user_id: str,
        chat_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Persist full chat messages for one user/chat pair."""

    @abstractmethod
    def reset_chat_history(
        self,
        user_id: str,
        chat_id: str,
        initial_messages: list[dict[str, Any]],
    ) -> None:
        """Reset one chat while keeping user long-term memories intact."""

    @abstractmethod
    def save_memory(self, user_id: str, content: str) -> Memory:
        """Store one long-term memory for ``user_id``."""

    @abstractmethod
    def get_memories(self, user_id: str) -> list[Memory]:
        """Return long-term memories for ``user_id`` only."""

    @abstractmethod
    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        content: str,
    ) -> Memory:
        """Update one existing memory owned by ``user_id``."""

    @abstractmethod
    def upsert_memory(self, user_id: str, content: str) -> Memory:
        """Create or update a related long-term memory for ``user_id``."""

    @abstractmethod
    def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        top_k: int,
    ) -> list[Memory]:
        """Return top-k user memories ranked by lexical relevance to ``query``."""

    @abstractmethod
    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete one specific memory owned by user_id."""

    @abstractmethod
    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for user_id (GDPR right-to-be-forgotten)."""

    @abstractmethod
    def purge_expired_memories(self) -> int:
        """Purge all expired memories across all users."""


class MemoryExtractor:
    """LLM-backed extractor for long-term memory decisions."""

    def __init__(self, provider: ModelGateway) -> None:
        self._provider = provider

    def extract(
        self,
        *,
        user_input: str,
        assistant_response: str,
        existing_memories: list[Memory],
    ) -> MemoryExtractionResult:
        """Extract a durable memory operation from one completed interaction."""
        memories_text = (
            "\n".join(
                (
                    f"- memory_id={memory.memory_id} | "
                    f"content={memory.content}"
                )
                for memory in existing_memories
            )
            if existing_memories
            else "(none)"
        )

        system_prompt = (
            "You extract stable user-specific long-term memories.\n"
            "Store only durable preferences, profile facts, or standing "
            "constraints.\n"
            "Ignore transient requests, one-off tasks, and general questions.\n\n"
            "Choose exactly one operation:\n"
            "- create: the information is genuinely new and should become "
            "a new memory.\n"
            "- update: the information changes, corrects, replaces, or "
            "conflicts with an existing memory.\n"
            "- ignore: the information should not be stored as long-term "
            "memory.\n\n"
            "For update, memory_id MUST be the exact ID of the existing "
            "memory being replaced.\n"
            "For create, memory_id MUST be an empty string.\n"
            "For ignore, memory and memory_id MUST both be empty strings.\n"
            "When the new information contradicts an existing memory, "
            "prefer update over create.\n"
            "Do not invent memory IDs. Only use IDs provided in the "
            "existing memories list.\n"
            "When an existing memory is updated, write the complete "
            "canonical replacement memory rather than describing the change.\n"
            "Avoid storing temporary task instructions or answers to "
            "one-off questions.\n\n"
            "Respond with JSON only using exactly this structure:\n"
            '{"should_store": true|false, '
            '"operation": "create"|"update"|"ignore", '
            '"memory": "...", "memory_id": "..."}\n'
            "If operation is ignore, should_store must be false.\n"
            "If operation is create or update, should_store must be true "
            "and memory must be non-empty."
        )

        user_prompt = (
            f"User input: {user_input}\n"
            f"Assistant response: {assistant_response}\n\n"
            f"Existing memories:\n{memories_text}"
        )

        try:
            response = self._provider.generate(
                ModelRequest(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    task_type=TaskType.REFLECTION,
                )
            )
        except Exception as error:
            logger.warning(
                "Automatic memory extraction failed during provider call",
                extra={"error_type": type(error).__name__},
            )
            return MemoryExtractionResult(
                should_store=False,
                memory="",
                operation="ignore",
                memory_id="",
            )

        parsed = self._parse_response(response.content)
        if parsed is None:
            logger.warning("Automatic memory extraction returned invalid JSON")
            return MemoryExtractionResult(
                should_store=False,
                memory="",
                operation="ignore",
                memory_id="",
            )

        return self._build_result(parsed, existing_memories)

    @staticmethod
    def _build_result(
        parsed: dict[str, Any],
        existing_memories: list[Memory],
    ) -> MemoryExtractionResult:
        """Validate parsed extractor output before it reaches storage."""
        should_store = parsed.get("should_store", False)
        if not isinstance(should_store, bool):
            return MemoryExtractionResult(
                should_store=False,
                memory="",
                operation="ignore",
                memory_id="",
            )

        operation_value = parsed.get("operation")

        if operation_value is None:
            # Backward compatibility with the previous extractor contract.
            # Old responses only contained should_store + memory.
            operation: MemoryOperation = (
                "create" if should_store else "ignore"
            )
        elif isinstance(operation_value, str):
            normalized_operation = operation_value.strip().casefold()
            if normalized_operation not in _VALID_MEMORY_OPERATIONS:
                logger.warning(
                    "Automatic memory extraction returned invalid operation",
                    extra={"operation": normalized_operation},
                )
                return MemoryExtractionResult(
                    should_store=False,
                    memory="",
                    operation="ignore",
                    memory_id="",
                )
            operation = normalized_operation  # type: ignore[assignment]
        else:
            return MemoryExtractionResult(
                should_store=False,
                memory="",
                operation="ignore",
                memory_id="",
            )

        memory_value = parsed.get("memory", "")
        memory = memory_value.strip() if isinstance(memory_value, str) else ""

        memory_id_value = parsed.get("memory_id", "")
        memory_id = (
            memory_id_value.strip()
            if isinstance(memory_id_value, str)
            else ""
        )

        if operation == "ignore":
            return MemoryExtractionResult(
                should_store=False,
                memory="",
                operation="ignore",
                memory_id="",
            )

        if not should_store or not memory:
            return MemoryExtractionResult(
                should_store=False,
                memory="",
                operation="ignore",
                memory_id="",
            )

        if operation == "create":
            if memory_id:
                logger.warning(
                    "Create memory operation unexpectedly contained "
                    "memory_id; ignoring the supplied ID",
                )
            return MemoryExtractionResult(
                should_store=True,
                memory=memory,
                operation="create",
                memory_id="",
            )

        # Update requires a real existing memory ID. Validate it here as an
        # additional safety boundary before the storage layer is called.
        if not memory_id:
            logger.warning(
                "Update memory operation missing memory_id",
            )
            return MemoryExtractionResult(
                should_store=False,
                memory="",
                operation="ignore",
                memory_id="",
            )

        existing_ids = {
            existing.memory_id
            for existing in existing_memories
        }

        if memory_id not in existing_ids:
            logger.warning(
                "Update memory operation referenced unknown memory_id",
                extra={"memory_id": memory_id},
            )
            return MemoryExtractionResult(
                should_store=False,
                memory="",
                operation="ignore",
                memory_id="",
            )

        return MemoryExtractionResult(
            should_store=True,
            memory=memory,
            operation="update",
            memory_id=memory_id,
        )

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any] | None:
        """Parse a strict JSON object returned by the model."""
        text = content.strip()
        if not text:
            return None

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None

        return payload if isinstance(payload, dict) else None


class JsonFileMemoryStore(MemoryStore):
    """JSON-backed persistent memory store.

    This implementation is intentionally simple and replaceable: it provides a
    narrow storage contract now and can later be swapped with SQL/vector
    backends without changing agent orchestration code.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._storage_path = storage_path or (MEMORY_DIR / "memory_store.json")
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._storage_path.exists():
            self._write_data(self._empty_data())

    @staticmethod
    def _empty_data() -> dict[str, Any]:
        return {
            "users": {},
            "emails": {},
            "chats": {},
            "memories": {},
        }

    @staticmethod
    def _normalize_email(email: str) -> str:
        normalized = email.strip().casefold()
        if not normalized:
            raise MemoryWriteError("Email cannot be blank.")
        return normalized

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.casefold()))

    def _read_data(self) -> dict[str, Any]:
        with self._lock:
            try:
                with self._storage_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except FileNotFoundError as error:
                raise MemoryReadError(
                    "Memory store file is missing.",
                    details={"path": str(self._storage_path)},
                ) from error
            except json.JSONDecodeError as error:
                raise MemoryReadError(
                    "Memory store file is corrupted.",
                    details={"path": str(self._storage_path)},
                ) from error
            except OSError as error:
                raise MemoryReadError(
                    "Unable to read memory store file.",
                    details={
                        "path": str(self._storage_path),
                        "error": str(error),
                    },
                ) from error

            if not isinstance(payload, dict):
                raise MemoryReadError(
                    "Memory store format is invalid.",
                    details={"path": str(self._storage_path)},
                )

            return payload

    def _write_data(self, payload: dict[str, Any]) -> None:
        with self._lock:
            temp_path = self._storage_path.with_suffix(".tmp")

            try:
                with temp_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=True, indent=2)

                temp_path.replace(self._storage_path)
            except OSError as error:
                raise MemoryWriteError(
                    "Unable to write memory store file.",
                    details={
                        "path": str(self._storage_path),
                        "error": str(error),
                    },
                ) from error

    def _ensure_user_exists(
        self,
        data: dict[str, Any],
        user_id: str,
    ) -> None:
        users = data.get("users", {})

        if user_id not in users:
            raise MemoryReadError(
                f"User '{user_id}' does not exist.",
                details={"user_id": user_id},
            )

    def create_user(
        self,
        user_id: str | None = None,
        email: str | None = None,
    ) -> User:
        data = self._read_data()
        users = data.setdefault("users", {})
        emails = data.setdefault("emails", {})
        resolved_user_id = user_id or str(uuid4())

        existing = users.get(resolved_user_id)

        if existing is not None:
            return User(
                user_id=existing["user_id"],
                email=existing.get("email"),
                created_at=existing["created_at"],
            )

        normalized_email: str | None = None

        if email is not None:
            normalized_email = self._normalize_email(email)
            mapped_user_id = emails.get(normalized_email)

            if mapped_user_id:
                mapped_user = users.get(mapped_user_id)

                if mapped_user is None:
                    raise MemoryReadError(
                        "Email mapping points to missing user.",
                        details={
                            "email": normalized_email,
                            "user_id": mapped_user_id,
                        },
                    )

                return User(
                    user_id=mapped_user["user_id"],
                    email=mapped_user.get("email"),
                    created_at=mapped_user["created_at"],
                )

        created_at = _utc_now_iso()

        users[resolved_user_id] = {
            "user_id": resolved_user_id,
            "email": normalized_email,
            "created_at": created_at,
        }

        data.setdefault("chats", {}).setdefault(resolved_user_id, {})
        data.setdefault("memories", {}).setdefault(resolved_user_id, [])

        if normalized_email:
            emails[normalized_email] = resolved_user_id

        self._write_data(data)

        return User(
            user_id=resolved_user_id,
            email=normalized_email,
            created_at=created_at,
        )

    def get_user(self, user_id: str) -> User | None:
        data = self._read_data()
        users = data.get("users", {})
        record = users.get(user_id)

        if record is None:
            return None

        return User(
            user_id=record["user_id"],
            email=record.get("email"),
            created_at=record["created_at"],
        )

    def get_or_create_user_by_email(self, email: str) -> User:
        normalized_email = self._normalize_email(email)
        data = self._read_data()
        users = data.get("users", {})
        emails = data.get("emails", {})

        mapped_user_id = emails.get(normalized_email)

        if mapped_user_id:
            record = users.get(mapped_user_id)

            if record is None:
                raise MemoryReadError(
                    "Email mapping points to missing user.",
                    details={
                        "email": normalized_email,
                        "user_id": mapped_user_id,
                    },
                )

            return User(
                user_id=record["user_id"],
                email=record.get("email"),
                created_at=record["created_at"],
            )

        return self.create_user(email=normalized_email)

    def create_chat(
        self,
        user_id: str,
        chat_id: str | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        resolved_chat_id = (chat_id or str(uuid4())).strip()

        if not resolved_chat_id:
            raise MemoryWriteError("Chat ID cannot be blank.")

        data = self._read_data()
        self._ensure_user_exists(data, user_id)

        chats_by_user = data.setdefault("chats", {}).setdefault(
            user_id,
            {},
        )

        if resolved_chat_id not in chats_by_user:
            now = _utc_now_iso()

            chats_by_user[resolved_chat_id] = {
                "chat_id": resolved_chat_id,
                "created_at": now,
                "updated_at": now,
                "messages": deepcopy(initial_messages or []),
            }

            self._write_data(data)

        return resolved_chat_id

    def get_chat_messages(
        self,
        user_id: str,
        chat_id: str,
    ) -> list[dict[str, Any]]:
        data = self._read_data()
        self._ensure_user_exists(data, user_id)

        chats_by_user = data.get("chats", {}).get(user_id, {})
        chat_record = chats_by_user.get(chat_id)

        if chat_record is None:
            return []

        return deepcopy(chat_record.get("messages", []))

    def save_chat_messages(
        self,
        user_id: str,
        chat_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        data = self._read_data()
        self._ensure_user_exists(data, user_id)

        chats_by_user = data.setdefault("chats", {}).setdefault(
            user_id,
            {},
        )

        existing = chats_by_user.get(chat_id)

        created_at = (
            existing.get("created_at")
            if isinstance(existing, dict)
            else _utc_now_iso()
        )

        chats_by_user[chat_id] = {
            "chat_id": chat_id,
            "created_at": created_at,
            "updated_at": _utc_now_iso(),
            "messages": deepcopy(messages),
        }

        self._write_data(data)

    def reset_chat_history(
        self,
        user_id: str,
        chat_id: str,
        initial_messages: list[dict[str, Any]],
    ) -> None:
        self.save_chat_messages(
            user_id,
            chat_id,
            deepcopy(initial_messages),
        )

    def save_memory(
        self,
        user_id: str,
        content: str,
        ttl_days: int | None = None,
    ) -> Memory:
        normalized_content = content.strip()

        if not normalized_content:
            raise MemoryWriteError("Memory content cannot be blank.")

        sanitized_content = SensitivityFilter.sanitize(normalized_content)

        with self._lock:
            data = self._read_data()
            self._ensure_user_exists(data, user_id)

            memories_by_user = data.setdefault("memories", {}).setdefault(
                user_id,
                [],
            )

            expires_at = None
            if ttl_days is not None and ttl_days > 0:
                expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

            memory = Memory(
                memory_id=str(uuid4()),
                user_id=user_id,
                content=sanitized_content,
                created_at=_utc_now_iso(),
                expires_at=expires_at,
            )

            memories_by_user.append(
                {
                    "memory_id": memory.memory_id,
                    "user_id": memory.user_id,
                    "content": memory.content,
                    "created_at": memory.created_at,
                    "expires_at": memory.expires_at,
                }
            )

            self._write_data(data)

            return memory

    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        content: str,
    ) -> Memory:
        """Update an existing memory owned by the supplied user.

        The memory ID is resolved only inside the user's own memory
        collection. The original ID and creation timestamp are preserved.
        """
        normalized_content = content.strip()

        if not normalized_content:
            raise MemoryWriteError("Memory content cannot be blank.")

        normalized_memory_id = memory_id.strip()

        if not normalized_memory_id:
            raise MemoryWriteError("Memory ID cannot be blank.")

        sanitized_content = SensitivityFilter.sanitize(normalized_content)

        with self._lock:
            data = self._read_data()
            self._ensure_user_exists(data, user_id)

            memories_by_user = data.setdefault("memories", {}).setdefault(
                user_id,
                [],
            )

            for record in memories_by_user:
                if record.get("memory_id") != normalized_memory_id:
                    continue

                record_user_id = record.get("user_id")

                if record_user_id != user_id:
                    raise MemoryReadError(
                        "Memory does not belong to the requested user.",
                        details={
                            "memory_id": normalized_memory_id,
                            "user_id": user_id,
                        },
                    )

                record["content"] = sanitized_content

                self._write_data(data)

                return Memory(
                    memory_id=str(record["memory_id"]),
                    user_id=str(record["user_id"]),
                    content=str(record["content"]),
                    created_at=str(record["created_at"]),
                    expires_at=record.get("expires_at"),
                )

            raise MemoryReadError(
                f"Memory '{normalized_memory_id}' does not exist.",
                details={
                    "memory_id": normalized_memory_id,
                    "user_id": user_id,
                },
            )

    def upsert_memory(
        self,
        user_id: str,
        content: str,
        ttl_days: int | None = None,
    ) -> Memory:
        """Create or update a related long-term memory.

        This method remains the compatibility fallback for callers that do
        not have an explicit memory operation. Exact duplicates are treated
        as existing memories. For related content, the existing deterministic
        lexical matching behavior is retained.
        """
        normalized_content = content.strip()

        if not normalized_content:
            raise MemoryWriteError("Memory content cannot be blank.")

        sanitized_content = SensitivityFilter.sanitize(normalized_content)

        with self._lock:
            data = self._read_data()
            self._ensure_user_exists(data, user_id)

            memories_by_user = data.setdefault("memories", {}).setdefault(
                user_id,
                [],
            )

            related_index = self._find_related_memory_index(
                memories_by_user,
                sanitized_content,
            )

            if related_index is None:
                expires_at = None
                if ttl_days is not None and ttl_days > 0:
                    expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

                memory = Memory(
                    memory_id=str(uuid4()),
                    user_id=user_id,
                    content=sanitized_content,
                    created_at=_utc_now_iso(),
                    expires_at=expires_at,
                )

                memories_by_user.append(
                    {
                        "memory_id": memory.memory_id,
                        "user_id": memory.user_id,
                        "content": memory.content,
                        "created_at": memory.created_at,
                        "expires_at": memory.expires_at,
                    }
                )

                self._write_data(data)

                return memory

            existing = memories_by_user[related_index]

            existing["content"] = sanitized_content

            self._write_data(data)

            return Memory(
                memory_id=str(existing["memory_id"]),
                user_id=str(existing["user_id"]),
                content=str(existing["content"]),
                created_at=str(existing["created_at"]),
                expires_at=existing.get("expires_at"),
            )

    @staticmethod
    def _find_related_memory_index(
        records: list[dict[str, Any]],
        candidate: str,
    ) -> int | None:
        """Find the strongest lexical relation for compatibility fallback."""
        candidate_tokens = JsonFileMemoryStore._tokenize(candidate)

        best_index: int | None = None
        best_score = 0.0

        for index, record in enumerate(records):
            existing = str(record.get("content", "")).strip()

            if not existing:
                continue

            if existing.casefold() == candidate.casefold():
                return index

            existing_tokens = JsonFileMemoryStore._tokenize(existing)

            if not existing_tokens or not candidate_tokens:
                continue

            intersection = len(existing_tokens & candidate_tokens)
            union = len(existing_tokens | candidate_tokens)

            if union == 0:
                continue

            score = intersection / union

            if score > best_score:
                best_score = score
                best_index = index

        if best_score >= 0.35:
            return best_index

        return None

    def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        top_k: int,
    ) -> list[Memory]:
        if top_k <= 0:
            return []

        all_memories = self.get_memories(user_id)
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scored: list[tuple[float, Memory]] = []

        for memory in all_memories:
            memory_tokens = self._tokenize(memory.content)

            if not memory_tokens:
                continue

            overlap = len(query_tokens & memory_tokens)

            if overlap == 0:
                continue

            score = overlap / len(query_tokens | memory_tokens)

            scored.append((score, memory))

        scored.sort(
            key=lambda item: (-item[0], item[1].created_at)
        )

        return [memory for _, memory in scored[:top_k]]

    def get_memories(self, user_id: str) -> list[Memory]:
        now_iso = _utc_now_iso()
        with self._lock:
            data = self._read_data()
            self._ensure_user_exists(data, user_id)

            records = data.get("memories", {}).get(user_id, [])

            memories: list[Memory] = []
            for record in records:
                exp = record.get("expires_at")
                if exp and exp <= now_iso:
                    continue
                memories.append(
                    Memory(
                        memory_id=record["memory_id"],
                        user_id=record["user_id"],
                        content=record["content"],
                        created_at=record["created_at"],
                        expires_at=record.get("expires_at"),
                    )
                )
            return memories

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete one specific memory owned by user_id."""
        normalized_memory_id = memory_id.strip()
        if not normalized_memory_id:
            raise MemoryWriteError("Memory ID cannot be blank.")

        with self._lock:
            data = self._read_data()
            self._ensure_user_exists(data, user_id)

            memories_by_user = data.setdefault("memories", {}).setdefault(user_id, [])
            initial_count = len(memories_by_user)
            data["memories"][user_id] = [
                m for m in memories_by_user if m.get("memory_id") != normalized_memory_id
            ]

            if len(data["memories"][user_id]) < initial_count:
                self._write_data(data)
                return True
            return False

    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for user_id (GDPR right-to-be-forgotten)."""
        with self._lock:
            data = self._read_data()
            self._ensure_user_exists(data, user_id)

            memories_by_user = data.setdefault("memories", {}).get(user_id, [])
            count = len(memories_by_user)
            data["memories"][user_id] = []
            if count > 0:
                self._write_data(data)
            return count

    def purge_expired_memories(self) -> int:
        """Purge all expired memories across all users."""
        now_iso = _utc_now_iso()
        purged = 0
        with self._lock:
            data = self._read_data()
            all_memories = data.setdefault("memories", {})
            for uid, mem_list in all_memories.items():
                kept = []
                for m in mem_list:
                    exp = m.get("expires_at")
                    if exp and exp <= now_iso:
                        purged += 1
                    else:
                        kept.append(m)
                all_memories[uid] = kept

            if purged > 0:
                self._write_data(data)
        return purged
