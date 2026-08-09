from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from hashlib import sha256
import re
from typing import Iterable, List

import numpy as np

from app.config import settings


class EmbeddingProvider(ABC):
    """
    Base interface for all embedding providers.

    Providers expose a stable collection identity used by persistent vector
    stores. The identity describes the embedding implementation/configuration,
    while the actual dimensionality is supplied by the caller after embedding.
    """

    @abstractmethod
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Return a 2D numpy array of shape (len(texts), dim)."""

    def collection_identity(self, dimension: int) -> str:
        """
        Return a stable identity for the embedding configuration.

        The dimension is intentionally supplied by the caller because the
        actual output dimension is authoritative. This prevents a provider's
        declared/default configuration from disagreeing with the vectors it
        actually returns.

        Subclasses with model/configuration information should override this
        method and include that information.
        """
        return (
            f"{self.__class__.__name__.lower()}"
            f":dim={int(dimension)}"
        )


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic hash-based embeddings for tests.

    Produces fixed-size vectors by hashing text and expanding the digest.
    Deterministic and requires no external services, suitable for tests.
    """

    def __init__(self, dim: int = 128):
        self.dim = int(dim)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        out = []

        for t in texts:
            digest = sha256(t.encode("utf-8")).digest()

            vals = []
            i = 0

            while len(vals) < self.dim:
                block = sha256(
                    digest + i.to_bytes(4, "little")
                ).digest()

                vals.extend(block)
                i += 1

            arr = np.frombuffer(
                bytes(vals[: self.dim]),
                dtype=np.uint8,
            ).astype(np.float32)

            norm = np.linalg.norm(arr)

            out.append(
                arr if norm == 0 else arr / norm
            )

        return np.vstack(out)

    def collection_identity(self, dimension: int) -> str:
        """
        Identify the deterministic mock embedding configuration.

        The configured dimension and the actual dimension are both represented
        through the final dimension value supplied by the caller.
        """
        return (
            f"mock"
            f":dim={int(dimension)}"
        )


class HashingEmbeddingProvider(EmbeddingProvider):
    """
    Lightweight local embedding based on hashed token counts.

    Not a neural embedding but deterministic, fast, and local. Useful as a
    free production fallback when no neural provider is configured.
    """

    def __init__(self, dim: int = 128):
        self.dim = int(dim)

        # Simple tokenizer.
        self._token_re = re.compile(
            r"\w+",
            flags=re.UNICODE,
        )

    def _tokens(self, text: str) -> List[str]:
        return [
            token.lower()
            for token in self._token_re.findall(text)
        ]

    def embed_texts(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        out = []

        for t in texts:
            tokens = self._tokens(t)
            counts = Counter(tokens)

            vec = np.zeros(
                self.dim,
                dtype=np.float32,
            )

            if counts:
                for token, count in counts.items():
                    hashed = int(
                        sha256(
                            token.encode("utf-8")
                        ).hexdigest(),
                        16,
                    )

                    index = hashed % self.dim
                    vec[index] += float(count)

                norm = np.linalg.norm(vec)

                if norm > 0:
                    vec = vec / norm

            out.append(vec)

        return np.vstack(out)

    def collection_identity(self, dimension: int) -> str:
        """
        Identify the hashing embedding configuration.
        """
        return (
            f"hashing"
            f":dim={int(dimension)}"
        )


class SentenceTransformersProvider(EmbeddingProvider):
    """
    Wrapper for sentence-transformers.

    The model name is retained explicitly because two models can have the
    same embedding dimension while representing completely different
    embedding spaces.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence_transformers is not installed"
            ) from exc

        normalized_model_name = str(
            model_name
        ).strip()

        if not normalized_model_name:
            raise ValueError(
                "Sentence-transformers model name must not be blank."
            )

        self.model_name = normalized_model_name
        self._model = SentenceTransformer(
            self.model_name
        )

    def embed_texts(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def collection_identity(
        self,
        dimension: int,
    ) -> str:
        """
        Identify the exact sentence-transformers model and dimension.

        The model name is included because two models may produce the same
        dimensionality while having incompatible embedding spaces.
        """
        return (
            f"sentence_transformers"
            f":model={self.model_name}"
            f":dim={int(dimension)}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedding_provider(
    name: str | None = None,
    **kwargs,
) -> EmbeddingProvider:
    """
    Create the configured embedding provider.
    """

    # Determine default provider based on runtime environment.
    env = getattr(
        settings,
        "environment",
        None,
    )

    if name is None:
        if (
            env
            and getattr(
                env,
                "name",
                str(env),
            ).lower()
            == "testing"
        ):
            name = "mock"
        else:
            name = (
                settings.embedding_provider
                or "sentence_transformers"
            )

    name = name.lower()

    if name in (
        "mock",
        "test-mock",
    ):
        return MockEmbeddingProvider(
            dim=kwargs.get(
                "dim",
                settings.embedding_dim,
            )
        )

    if name in (
        "hashing",
        "local",
    ):
        return HashingEmbeddingProvider(
            dim=kwargs.get(
                "dim",
                settings.embedding_dim,
            )
        )

    if name in (
        "sentence_transformers",
        "st",
    ):
        try:
            return SentenceTransformersProvider(
                model_name=kwargs.get(
                    "model_name",
                    "all-MiniLM-L6-v2",
                )
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers provider requested "
                "but the package is not installed. "
                "Install 'sentence-transformers' or set "
                "EMBEDDING_PROVIDER=hashing for a local fallback."
            )

    # Last resort: mock.
    return MockEmbeddingProvider(
        dim=kwargs.get(
            "dim",
            settings.embedding_dim,
        )
    )