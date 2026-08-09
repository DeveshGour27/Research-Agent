"""
Vector Store module.

Responsibilities:

- In-memory and Chroma-backed vector stores with compatible APIs.
- Cosine-similarity retrieval with top-k.
- ChromaDB provides persistent vector storage.
- Persistent Chroma collections are isolated by embedding configuration.

Important invariant:

    Same embedding configuration
        -> same collection

    Different embedding configuration
        -> different collection

This prevents incompatible embedding spaces from being mixed inside a
persistent Chroma collection.

Collection selection is deterministic and identity-first:

    1. The provider-scoped collection (named from a hash of the embedding
       provider identity, see `_build_collection_name`) is used directly
       when it exists and already has data.
    2. Otherwise, the legacy/base collection (`settings.chroma_collection_name`,
       used before provider-scoping existed) is considered. It is only
       adopted when its *stored* identity metadata
       (`embedding_provider_id` / `embedding_dim`) provably matches the
       currently configured provider and dimension.
    3. If the legacy collection has data but predates this scheme (no
       identity metadata was ever stored on it), a dimension-only probe is
       used as a last-resort fallback. This is a weaker guarantee than (2)
       -- it can only confirm dimension compatibility, not provider
       identity -- and is logged accordingly.
    4. If nothing compatible is found, the provider-scoped collection is
       created (or reused if it already exists empty), and its identity is
       recorded in its metadata so future runs can use path (2) instead of
       the fallback probe.

This guarantees indexing and retrieval always resolve to the same
collection for a given embedding configuration, and that existing
compatible data is never skipped in favor of an empty collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VectorItem:
    id: str
    vector: np.ndarray
    metadata: dict


class InMemoryVectorStore:
    """
    Simple deterministic vector store used for tests.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._items: Dict[str, VectorItem] = {}

    def add(
        self,
        id: str,
        vector: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if (
            vector.ndim != 1
            or vector.shape[0] != self.dim
        ):
            raise ValueError(
                f"Embedding vector has wrong shape: "
                f"expected ({self.dim},), "
                f"got {vector.shape}"
            )

        self._items[id] = VectorItem(
            id=id,
            vector=vector,
            metadata=dict(metadata or {}),
        )

    def add_batch(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadatas: List[dict] | None = None,
    ) -> None:
        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Vectors must be a 2D array."
            )

        if vectors.shape[0] != len(ids):
            raise ValueError(
                "IDs and vectors length mismatch."
            )

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.dim}, "
                f"got {vectors.shape[1]}"
            )

        for i, item_id in enumerate(ids):
            metadata = (
                metadatas[i]
                if (
                    metadatas is not None
                    and i < len(metadatas)
                )
                else {}
            )

            self.add(
                item_id,
                vectors[i],
                metadata,
            )

    @staticmethod
    def _cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        denom = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denom == 0:
            return 0.0

        return float(
            np.dot(a, b) / denom
        )

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float, dict]]:
        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if (
            vector.ndim != 1
            or vector.shape[0] != self.dim
        ):
            raise ValueError(
                "Query vector has wrong shape."
            )

        results = []

        for item_id, item in self._items.items():
            score = self._cosine_similarity(
                vector,
                item.vector,
            )

            results.append(
                (
                    item_id,
                    score,
                    item.metadata,
                )
            )

        results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return results[:top_k]

    def get(
        self,
        id: str,
    ) -> Optional[VectorItem]:
        return self._items.get(id)

    def all_ids(self) -> List[str]:
        return list(self._items.keys())

    def all_items(self) -> List[Tuple[str, dict]]:
        """
        Return all stored IDs and metadata.
        """
        return [
            (
                item_id,
                dict(item.metadata),
            )
            for item_id, item in self._items.items()
        ]

    def size(self) -> int:
        return len(self._items)


class ChromaVectorStore:
    """
    Persistent ChromaDB-backed vector store.

    A fresh instance connects to the same on-disk database, allowing
    retrieval after application restart.
    """

    def __init__(
        self,
        dim: int,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_id: str | None = None,
    ):
        try:
            import chromadb
        except Exception as exc:
            raise ImportError(
                "chromadb is required for ChromaVectorStore."
            ) from exc

        self.dim = int(dim)
        self.provider_id = provider_id

        persist_directory = (
            persist_directory
            or settings.chroma_db_dir
        )

        persist_directory = os.path.abspath(
            os.path.expanduser(
                persist_directory
            )
        )

        requested_collection = (
            collection_name
            or settings.chroma_collection_name
        )

        base_collection = settings.chroma_collection_name

        self.persist_directory = persist_directory
        # self.collection_name will be set to the actual collection used below

        os.makedirs(
            self.persist_directory,
            exist_ok=True,
        )

        # Explicitly use Chroma's persistent client.
        self._client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # Deterministic, identity-metadata-driven collection selection.
        # See module docstring for the full resolution order.
        self._collection, chosen = _resolve_chroma_collection(
            client=self._client,
            requested_name=requested_collection,
            base_name=base_collection,
            provider_id=self.provider_id,
            dim=self.dim,
        )

        self.collection_name = chosen

    def add(
        self,
        id: str,
        vector: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        self.add_batch(
            [id],
            np.expand_dims(
                np.asarray(
                    vector,
                    dtype=np.float32,
                ),
                axis=0,
            ),
            metadatas=[
                metadata or {}
            ],
        )

    def add_batch(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadatas: List[dict] | None = None,
    ) -> None:
        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Vectors must be a 2D array."
            )

        if vectors.shape[0] != len(ids):
            raise ValueError(
                "IDs and vectors length mismatch."
            )

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"store expects {self.dim}, "
                f"received {vectors.shape[1]}."
            )

        metadatas = metadatas or [
            {}
            for _ in ids
        ]

        if len(metadatas) != len(ids):
            raise ValueError(
                "IDs and metadatas length mismatch."
            )

        enriched_metadatas: List[dict] = []
        documents: List[str] = []

        for i, item_id in enumerate(ids):
            metadata = dict(
                metadatas[i]
                if i < len(metadatas)
                else {}
            )

            # Preserve canonical application chunk ID.
            metadata["_chunk_id"] = str(
                metadata.get(
                    "_chunk_id",
                    item_id,
                )
            )

            # Chroma documents contain the actual chunk text.
            text = str(
                metadata.get(
                    "text",
                    "",
                )
            )

            enriched_metadatas.append(
                metadata
            )

            documents.append(text)

        # upsert makes repeated indexing safe.
        self._collection.upsert(
            ids=[
                str(item_id)
                for item_id in ids
            ],
            embeddings=vectors.tolist(),
            metadatas=enriched_metadatas,
            documents=documents,
        )

        # Persist to disk when using a persistent client to ensure data is
        # visible to new client instances created later.
        try:
            if hasattr(
                self._client,
                "persist",
            ):
                self._client.persist()
        except Exception:
            # Don't fail indexing if persistence isn't supported.
            pass

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float, dict]]:
        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "Query vector must be one-dimensional."
            )

        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Query embedding dimension mismatch: "
                f"store expects {self.dim}, "
                f"received {vector.shape[0]}."
            )

        if self.size() == 0:
            return []

        top_k = max(
            1,
            int(top_k),
        )

        # Never ask Chroma for more results than exist.
        top_k = min(
            top_k,
            self.size(),
        )

        result = self._collection.query(
            query_embeddings=[
                vector.tolist()
            ],
            n_results=top_k,
            include=[
                "metadatas",
                "distances",
                "documents",
            ],
        )

        raw_ids = result.get("ids") or []
        raw_distances = (
            result.get("distances") or []
        )
        raw_metadatas = (
            result.get("metadatas") or []
        )
        raw_documents = (
            result.get("documents") or []
        )

        # Chroma normally returns nested lists:
        # [[id1, id2, ...]]
        ids = (
            raw_ids[0]
            if (
                raw_ids
                and isinstance(
                    raw_ids[0],
                    (list, tuple),
                )
            )
            else raw_ids
        )

        distances = (
            raw_distances[0]
            if (
                raw_distances
                and isinstance(
                    raw_distances[0],
                    (list, tuple),
                )
            )
            else raw_distances
        )

        metadatas = (
            raw_metadatas[0]
            if (
                raw_metadatas
                and isinstance(
                    raw_metadatas[0],
                    (list, tuple),
                )
            )
            else raw_metadatas
        )

        documents = (
            raw_documents[0]
            if (
                raw_documents
                and isinstance(
                    raw_documents[0],
                    (list, tuple),
                )
            )
            else raw_documents
        )

        results: List[
            Tuple[str, float, dict]
        ] = []

        for i, raw_id in enumerate(ids):
            metadata = (
                dict(metadatas[i])
                if (
                    i < len(metadatas)
                    and isinstance(
                        metadatas[i],
                        dict,
                    )
                )
                else {}
            )

            # Chroma's documents field is the canonical fallback for text
            # if metadata does not contain it.
            if (
                "text" not in metadata
                and i < len(documents)
                and documents[i] is not None
            ):
                metadata["text"] = str(
                    documents[i]
                )

            distance = (
                float(distances[i])
                if i < len(distances)
                else 0.0
            )

            # Chroma collection uses:
            #     hnsw:space = cosine
            #
            # Therefore Chroma returns cosine distance.
            # Convert distance to similarity:
            #
            #     similarity = 1 - distance
            score = max(
                0.0,
                min(
                    1.0,
                    1.0 - distance,
                ),
            )

            canonical_id = str(
                metadata.get(
                    "_chunk_id",
                    raw_id,
                )
            )

            results.append(
                (
                    canonical_id,
                    score,
                    metadata,
                )
            )

        # Guarantee descending similarity order.
        results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return results

    def get(
        self,
        id: str,
    ) -> Optional[VectorItem]:
        """
        Retrieve metadata for one stored vector.

        Chroma's stable API does not require exposing the raw embedding here.
        """
        try:
            result = self._collection.get(
                ids=[
                    str(id)
                ],
                include=[
                    "metadatas",
                    "documents",
                ],
            )
        except Exception:
            return None

        ids = result.get("ids") or []

        if not ids:
            return None

        metadata_list = (
            result.get("metadatas") or []
        )

        documents = (
            result.get("documents") or []
        )

        metadata = (
            dict(metadata_list[0])
            if (
                metadata_list
                and isinstance(
                    metadata_list[0],
                    dict,
                )
            )
            else {}
        )

        if (
            "text" not in metadata
            and documents
            and documents[0] is not None
        ):
            metadata["text"] = str(
                documents[0]
            )

        canonical_id = str(
            metadata.get(
                "_chunk_id",
                ids[0],
            )
        )

        return VectorItem(
            id=canonical_id,
            vector=np.empty(
                self.dim,
                dtype=np.float32,
            ),
            metadata=metadata,
        )

    def all_ids(self) -> List[str]:
        try:
            result = self._collection.get()
            ids = result.get("ids") or []

            return [
                str(item_id)
                for item_id in ids
            ]

        except Exception:
            return []

    def all_items(self) -> List[Tuple[str, dict]]:
        """
        Return all persisted IDs and metadata.

        This is used by the Retriever to reconstruct lexical/BM25 retrieval
        after creating a fresh Retriever instance.
        """
        try:
            result = self._collection.get(
                include=[
                    "metadatas",
                    "documents",
                ]
            )

        except Exception:
            return []

        ids = result.get("ids") or []

        metadatas = (
            result.get("metadatas") or []
        )

        documents = (
            result.get("documents") or []
        )

        items: List[
            Tuple[str, dict]
        ] = []

        for i, raw_id in enumerate(ids):
            metadata = (
                dict(metadatas[i])
                if (
                    i < len(metadatas)
                    and isinstance(
                        metadatas[i],
                        dict,
                    )
                )
                else {}
            )

            # Restore text from Chroma's document field.
            if (
                "text" not in metadata
                and i < len(documents)
                and documents[i] is not None
            ):
                metadata["text"] = str(
                    documents[i]
                )

            canonical_id = str(
                metadata.get(
                    "_chunk_id",
                    raw_id,
                )
            )

            items.append(
                (
                    canonical_id,
                    metadata,
                )
            )

        return items

    def size(self) -> int:
        try:
            # Prefer native count when available.
            if hasattr(
                self._collection,
                "count",
            ):
                return int(
                    self._collection.count()
                )

            # Fallback to querying ids.
            # NOTE: "ids" is always returned by Chroma's get()/query() and
            # is not a valid `include` value in current chromadb versions
            # (passing it raises ValueError). Ask for nothing extra.
            result = (
                self._collection.get(
                    include=[]
                )
                or {}
            )

            ids = result.get("ids") or []

            if (
                isinstance(ids, list)
                and ids
                and isinstance(
                    ids[0],
                    (list, tuple),
                )
            ):
                ids = ids[0]

            return len(ids)

        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Chroma collection identity helpers
# ---------------------------------------------------------------------------

def _chroma_collection_size(collection) -> int:
    """
    Return the number of items stored in a Chroma collection.

    Prefers the native `count()` API and falls back to fetching ids when
    it's unavailable. Never raises: an unreadable collection is treated as
    empty, since callers use this purely to decide whether a collection is
    safe/worthwhile to adopt.
    """
    try:
        if hasattr(collection, "count"):
            return int(collection.count())

        # NOTE: "ids" is always returned by Chroma's get() and is not a
        # valid `include` value in current chromadb versions (passing it
        # raises ValueError). Ask for nothing extra.
        data = collection.get(include=[]) or {}
        ids = data.get("ids") or []

        if (
            isinstance(ids, list)
            and ids
            and isinstance(ids[0], (list, tuple))
        ):
            ids = ids[0]

        return len(ids)

    except Exception:
        return 0


def _stored_collection_identity(
    collection,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Read back the embedding identity a collection was created with, if any.

    Collections created by this module store `embedding_provider_id` and
    `embedding_dim` in their Chroma metadata at creation time (see
    `_resolve_chroma_collection`). Collections that predate this scheme
    (e.g. a legacy collection created before provider-scoping existed) will
    have neither key, which callers must treat as "identity unknown" rather
    than "identity empty".
    """
    try:
        metadata = collection.metadata or {}
    except Exception:
        metadata = {}

    provider_id = metadata.get("embedding_provider_id")

    stored_dim = metadata.get("embedding_dim")
    try:
        stored_dim = (
            int(stored_dim)
            if stored_dim is not None
            else None
        )
    except (TypeError, ValueError):
        stored_dim = None

    return provider_id, stored_dim


def _dimension_only_probe_compatible(
    collection,
    dim: int,
) -> bool:
    """
    Best-effort dimension compatibility check for legacy collections that
    have no stored identity metadata.

    This can only confirm that a zero-vector of the given dimension is
    accepted by Chroma's index for this collection -- it cannot confirm
    that the collection's embedding *provider* matches the currently
    configured provider. Two different providers that happen to emit the
    same dimension would both pass this probe. It exists purely as a
    degraded fallback for collections written before this module started
    recording provider identity, and every successful use of it is logged
    so it's visible when the weaker guarantee was relied on.
    """
    if _chroma_collection_size(collection) == 0:
        return False

    try:
        test_vector = [0.0] * int(dim)
        # NOTE: "ids" is always returned by Chroma's query() and is not a
        # valid `include` value in current chromadb versions (passing it
        # raises ValueError, which previously made this probe -- and thus
        # legacy-collection adoption -- always fail). Ask for nothing
        # extra; a dimension mismatch still raises and is still caught
        # below.
        collection.query(
            query_embeddings=[test_vector],
            n_results=1,
            include=[],
        )
        return True
    except Exception:
        return False


def _resolve_chroma_collection(
    client,
    requested_name: str,
    base_name: str,
    provider_id: str | None,
    dim: int,
):
    """
    Deterministically resolve which Chroma collection a given embedding
    configuration (`provider_id`, `dim`) should read from and write to.

    Resolution order:

    1. The provider-scoped collection (`requested_name`) if it exists and
       already has data.
    2. The legacy/base collection (`base_name`), only if it exists, has
       data, and its *stored* identity metadata provably matches
       (`provider_id`, `dim`).
    3. The legacy/base collection as a degraded fallback, only if it has
       data but no stored identity metadata at all (true pre-scoping
       legacy data) and a dimension-only probe succeeds. This path is
       logged because it cannot verify provider identity, only dimension.
    4. Otherwise, create (or reuse, if already present but empty) the
       provider-scoped collection, recording its identity in metadata so
       future calls can use path (1) or (2) instead of a fallback probe.

    Returns (collection, chosen_collection_name).
    """

    def get_if_exists(name: str):
        try:
            return client.get_collection(name=name)
        except Exception:
            return None

    requested = get_if_exists(requested_name)

    if (
        requested is not None
        and _chroma_collection_size(requested) > 0
    ):
        return requested, requested_name

    if requested_name != base_name:
        base = get_if_exists(base_name)

        if base is not None and _chroma_collection_size(base) > 0:
            stored_provider_id, stored_dim = (
                _stored_collection_identity(base)
            )

            if (
                stored_provider_id is not None
                or stored_dim is not None
            ):
                # Base collection has recorded identity: only adopt on an
                # exact, provable match. A mismatch here means the base
                # collection is provably incompatible -- never adopt it,
                # regardless of dimension.
                if (
                    stored_provider_id == provider_id
                    and stored_dim == int(dim)
                ):
                    return base, base_name
            else:
                # True legacy collection with no identity metadata at all:
                # fall back to a dimension-only probe as a last resort.
                if _dimension_only_probe_compatible(base, dim):
                    logger.warning(
                        "Adopting legacy Chroma collection '%s' via "
                        "dimension-only compatibility probe: no "
                        "embedding_provider_id/embedding_dim metadata was "
                        "found on it, so only dimension (%d) could be "
                        "verified, not embedding-provider identity. "
                        "Consider re-indexing to migrate this data into a "
                        "provider-scoped collection with recorded identity.",
                        base_name,
                        int(dim),
                    )

                    # Claim the legacy collection for this embedding
                    # configuration by stamping its identity now. Without
                    # this, a *different* provider that happens to share
                    # the same dimension would pass this same weak probe
                    # on a later run and also get silently adopted into
                    # the same collection -- exactly the kind of mixing
                    # this module exists to prevent. Stamping identity on
                    # first legitimate adoption means every subsequent
                    # call goes through the strict metadata match above
                    # instead of the probe.
                    try:
                        # Chroma's modify(metadata=...) replaces the whole
                        # metadata dict rather than merging into it, so we
                        # rebuild it explicitly. Reserved configuration keys
                        # such as "hnsw:space" must NOT be round-tripped
                        # through this call: re-sending even the unchanged
                        # existing value trips Chroma's "distance function
                        # cannot be changed" guard. Only carry forward
                        # ordinary (non-reserved) keys plus our two identity
                        # fields.
                        existing_metadata = {
                            key: value
                            for key, value in dict(
                                base.metadata or {}
                            ).items()
                            if not key.startswith("hnsw:")
                        }
                        existing_metadata.update(
                            {
                                "embedding_dim": int(dim),
                                "embedding_provider_id": provider_id,
                            }
                        )
                        base.modify(metadata=existing_metadata)
                    except Exception:
                        logger.warning(
                            "Could not persist embedding identity metadata "
                            "onto legacy collection '%s' after adopting it "
                            "via dimension-only probe; future runs will "
                            "repeat the weaker probe-based check for this "
                            "collection.",
                            base_name,
                        )

                    return base, base_name

    # Nothing compatible found: create (or reuse-if-empty) the
    # provider-scoped collection, recording its identity for next time.
    collection = client.get_or_create_collection(
        name=requested_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_dim": int(dim),
            "embedding_provider_id": provider_id,
        },
    )

    return collection, requested_name


def _collection_suffix(
    provider_id: str,
) -> str:
    """
    Convert an arbitrary provider identity into a deterministic compact
    collection suffix.

    We intentionally hash the identity instead of placing the raw provider
    configuration in the Chroma collection name.

    This provides:

    - stable names
    - bounded length
    - safe characters
    - no accidental exposure of long model/configuration strings
    - collision resistance suitable for collection names
    """
    normalized = str(
        provider_id
    ).strip()

    if not normalized:
        raise ValueError(
            "provider_id must not be empty when "
            "creating a provider-scoped Chroma collection."
        )

    digest = sha256(
        normalized.encode("utf-8")
    ).hexdigest()

    return digest[:24]


def _build_collection_name(
    base_name: str,
    provider_id: str | None,
) -> str:
    """
    Build the concrete Chroma collection name.

    Legacy callers that do not provide a provider identity retain the original
    collection name for backward compatibility.

    Provider-aware callers receive an isolated deterministic collection.
    """
    base_name = str(
        base_name
    ).strip()

    if not base_name:
        raise ValueError(
            "Chroma collection name must not be blank."
        )

    if not provider_id:
        return base_name

    suffix = _collection_suffix(
        provider_id
    )

    return f"{base_name}__{suffix}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store(
    dim: int,
    backend: str | None = None,
    provider_id: str | None = None,
) -> InMemoryVectorStore | ChromaVectorStore:
    """
    Construct the configured vector store.

    Testing uses the in-memory backend.

    Development/production uses Chroma by default.

    When provider_id is supplied, Chroma collection identity is scoped to the
    embedding configuration so incompatible embedding spaces cannot share the
    same persistent collection.
    """

    backend = (
        backend
        or settings.vector_store
        or "chroma"
    ).lower()

    environment = getattr(
        settings,
        "environment",
        None,
    )

    environment_name = getattr(
        environment,
        "name",
        str(environment),
    ).lower()

    if environment_name == "testing":
        backend = "inmemory"

    if backend == "inmemory":
        return InMemoryVectorStore(
            dim=dim
        )

    if backend == "chroma":
        collection_name = _build_collection_name(
            settings.chroma_collection_name,
            provider_id,
        )

        return ChromaVectorStore(
            dim=dim,
            persist_directory=(
                settings.chroma_db_dir
            ),
            collection_name=collection_name,
            provider_id=provider_id,
        )

    raise ValueError(
        f"Unknown vector store backend: {backend}"
    )