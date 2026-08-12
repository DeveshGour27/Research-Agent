"""
Retriever module.

Provides the high-level Agentic retrieval loop:

Query
 -> rewrite
 -> decompose
 -> retrieve
 -> rerank
 -> evaluate
 -> refine
 -> retrieve again
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import logging

from app.config import settings

from rag.chunking import (
    Document,
    chunk_documents,
    Chunk,
)

from rag.embeddings import (
    get_embedding_provider,
)

from rag.vector_store import (
    get_vector_store,
)

from rag.hybrid_search import (
    combine_scores,
    Retrieved,
)

from rag.reranker import (
    LexicalReranker,
    NoopReranker,
    Reranker,
    NeuralReranker,
)


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    retrieved: List[Retrieved]
    iteration: int
    stop_reason: Optional[str]


def _rewrite_query(query: str) -> str:
    try:
        if not query or (
            len(query) < 40
            and query.count(" ") < 3
        ):
            return " ".join(
                query.strip().split()
            )

        if (
            getattr(
                settings,
                "rewriting_enabled",
                True,
            )
            and getattr(
                settings,
                "rag_enabled",
                True,
            )
        ):
            try:
                from app.llm.factory import create_chat_provider
                from app.llm.base import ChatMessage

                provider = create_chat_provider(
                    settings
                )

                prompt = (
                    "Rewrite the user's query to "
                    "a concise, retrieval-optimized "
                    "query. Return only the rewritten "
                    "query as plain text.\n\n"
                    f"Original: {query}"
                )

                response = provider.generate(
                    [
                        ChatMessage(
                            role="user",
                            content=prompt,
                        )
                    ]
                )

                content = (
                    response.content or ""
                ).strip()

                if content:
                    return content

            except Exception:
                logger.exception(
                    "LLM rewrite failed; "
                    "falling back to original query."
                )

        return " ".join(
            query.strip().split()
        )

    except Exception as exc:
        logger.warning(
            "Query rewrite failed: %s",
            exc,
        )

        return query


def _decompose_query(
    query: str,
) -> List[str]:
    try:
        if (
            len(query) < 30
            and query.count(" ") < 3
        ):
            return (
                [
                    query.strip()
                ]
                if query.strip()
                else [query]
            )

        if (
            getattr(
                settings,
                "decomposition_enabled",
                True,
            )
            and getattr(
                settings,
                "rag_enabled",
                True,
            )
        ):
            try:
                import json

                from app.llm.factory import create_chat_provider
                from app.llm.base import ChatMessage

                provider = create_chat_provider(
                    settings
                )

                prompt = (
                    "Decompose the following complex "
                    "user query into a JSON array of "
                    "short independent subqueries suitable "
                    "for retrieval. Return only a JSON "
                    "array of strings. If the query is "
                    "already simple, return an array with "
                    "the original query.\n\n"
                    f"Query: {query}"
                )

                response = provider.generate(
                    [
                        ChatMessage(
                            role="user",
                            content=prompt,
                        )
                    ]
                )

                content = (
                    response.content or ""
                ).strip()

                try:
                    values = json.loads(
                        content
                    )

                    if isinstance(
                        values,
                        list,
                    ):
                        output: List[str] = []
                        seen = set()

                        for value in values:
                            if not isinstance(
                                value,
                                str,
                            ):
                                continue

                            value = value.strip()

                            if (
                                not value
                                or value in seen
                            ):
                                continue

                            seen.add(value)
                            output.append(value)

                        if output:
                            return output

                except Exception:
                    logger.exception(
                        "Failed to parse LLM "
                        "decomposition response."
                    )

            except Exception:
                logger.exception(
                    "LLM decomposition failed; "
                    "falling back to heuristic."
                )

        parts = [
            part.strip()
            for part in query.split("?")
            if part.strip()
        ]

        output: List[str] = []

        for part in parts:
            if " and " in part:
                output.extend(
                    [
                        item.strip()
                        for item in part.split(
                            " and "
                        )
                        if item.strip()
                    ]
                )
            else:
                output.append(part)

        return output or [query]

    except Exception:
        return [query]


def _evaluate_retrieval(
    retrieved: List[Retrieved],
    relevant_ids: Iterable[str],
    k: int,
) -> Dict[str, float]:
    relevant_set = set(
        relevant_ids
    )

    ids = [
        result.chunk_id
        for result in retrieved[:k]
    ]

    recall = (
        len(
            [
                item
                for item in ids
                if item in relevant_set
            ]
        )
        / max(
            1,
            len(relevant_set),
        )
    )

    precision = (
        len(
            [
                item
                for item in ids
                if item in relevant_set
            ]
        )
        / max(
            1,
            k,
        )
    )

    mrr = 0.0

    for rank, chunk_id in enumerate(
        ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            mrr = 1.0 / rank
            break

    return {
        "recall@k": recall,
        "precision@k": precision,
        "mrr": mrr,
    }


class Retriever:

    def __init__(
        self,
        embedding_provider_name: Optional[str] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.embedding_provider = (
            get_embedding_provider(
                embedding_provider_name
            )
        )

        if reranker is not None:
            self.reranker = reranker

        elif not getattr(
            settings,
            "reranking_enabled",
            True,
        ):
            self.reranker = NoopReranker()

        else:
            reranker_type = getattr(
                settings,
                "reranker_type",
                "neural",
            ).lower()

            if reranker_type == "neural":
                self.reranker = NeuralReranker()

            elif reranker_type == "lexical":
                self.reranker = LexicalReranker()

            else:
                self.reranker = NoopReranker()

        self.chunk_index: Dict[
            str,
            Chunk,
        ] = {}

        self._bm25: Dict[str, Any] = {}
        self._bm25_ids: Dict[str, List[str]] = {}

        # Created lazily using the actual embedding dimension.
        self._vector_store = None

    def index_documents(
        self,
        docs: Iterable[Document],
        user_id: str,
    ) -> None:
        chunks = chunk_documents(
            docs
        )

        self.chunk_index.update({
            chunk.id: chunk
            for chunk in chunks
        })

        self._bm25_ids[user_id] = [
            chunk.id
            for chunk in chunks
        ]

        tokenized = [
            chunk.text.split()
            for chunk in chunks
        ]

        if tokenized:
            from rag.reranker import BM25Okapi

            self._bm25[user_id] = BM25Okapi(
                tokenized
            )

        else:
            self._bm25.pop(user_id, None)

        texts = [
            chunk.text
            for chunk in chunks
        ]

        if not texts:
            logger.warning(
                "No chunks were produced for indexing."
            )
            return

        embeddings = (
            self.embedding_provider.embed_texts(
                texts
            )
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding provider must return "
                "a 2D array."
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Number of embeddings does not "
                "match number of chunks."
            )

        dimension = embeddings.shape[1]

        # The embedding provider owns the knowledge of what makes its
        # embedding space unique. The actual output dimension is supplied
        # here because the generated vectors are authoritative.
        provider_id = (
            self.embedding_provider.collection_identity(
                dimension
            )
        )

        vector_store = get_vector_store(
            dim=dimension,
            provider_id=provider_id,
        )

        metadata = [
            {
                "text": chunk.text,
                "user_id": user_id,
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        vector_store.add_batch(
            [
                chunk.id
                for chunk in chunks
            ],
            embeddings,
            metadatas=metadata,
        )

        self._vector_store = vector_store

        logger.info(
            "Indexed %d chunks into %s",
            len(chunks),
            type(vector_store).__name__,
        )

    def _restore_bm25_from_vector_store(
        self,
        user_id: str,
    ) -> None:
        """
        Rebuild the in-memory BM25 index from persisted vector-store
        metadata when using a fresh Retriever.
        """

        if (
            user_id in self._bm25
            or self._vector_store is None
        ):
            return

        try:
            items = (
                self._vector_store.all_items(user_id=user_id)
            )

        except Exception as exc:
            logger.warning(
                "Could not restore BM25 from vector store: %s",
                exc,
            )
            return

        if not items:
            return

        from rag.reranker import BM25Okapi

        ids: List[str] = []
        tokenized: List[
            List[str]
        ] = []

        for chunk_id, metadata in items:
            metadata = metadata or {}

            text = str(
                metadata.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            ids.append(
                str(chunk_id)
            )

            tokenized.append(
                text.split()
            )

        if not tokenized:
            return

        self._bm25_ids[user_id] = ids
        self._bm25[user_id] = BM25Okapi(
            tokenized
        )

        logger.info(
            "Restored BM25 index from vector store: %d chunks",
            len(ids),
        )

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        user_id: str,
    ) -> List[
        Tuple[str, float, dict]
    ]:
        bm25 = self._bm25.get(user_id)
        if bm25 is None:
            return []

        scores = bm25.get_scores(
            query.split()
        )

        pairs = list(
            zip(
                self._bm25_ids.get(user_id, []),
                scores,
            )
        )

        pairs.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        persisted_metadata: Dict[
            str,
            dict,
        ] = {}

        if self._vector_store is not None:
            try:
                persisted_metadata = {
                    str(chunk_id): dict(
                        metadata or {}
                    )
                    for chunk_id, metadata
                    in self._vector_store.all_items(user_id=user_id)
                }

            except Exception:
                persisted_metadata = {}

        results = []

        for chunk_id, score in pairs[:top_k]:
            chunk = self.chunk_index.get(
                chunk_id
            )

            if chunk is not None:
                metadata = {
                    "text": chunk.text,
                    **chunk.metadata,
                }

            else:
                metadata = (
                    persisted_metadata.get(
                        str(chunk_id),
                        {},
                    )
                )

            results.append(
                (
                    chunk_id,
                    float(score),
                    metadata,
                )
            )

        return results

    def _vector_search(
        self,
        query: str,
        top_k: int,
        user_id: str,
    ) -> List[
        Tuple[str, float, dict]
    ]:
        query_vector = (
            self.embedding_provider.embed_texts(
                [query]
            )[0]
        )

        if self._vector_store is None:
            # Use the embedding provider's stable configuration identity so
            # retrieval resolves to exactly the same collection identity used
            # during indexing.
            provider_id = (
                self.embedding_provider.collection_identity(
                    query_vector.shape[0]
                )
            )

            self._vector_store = get_vector_store(
                dim=query_vector.shape[0],
                provider_id=provider_id,
            )

        results = self._vector_store.query(
            query_vector,
            top_k=top_k,
            user_id=user_id,
        )

        logger.debug(
            "Vector retrieval returned %d results.",
            len(results),
        )

        return results

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        hybrid: bool = True,
        rerank: bool = True,
    ) -> List[Retrieved]:

        top_k = (
            top_k
            or settings.top_k_retrieval
        )

        # =====================================================
        # VECTOR-ONLY RETRIEVAL
        # =====================================================

        if not hybrid:
            vector_results = self._vector_search(
                query,
                top_k,
                user_id=user_id,
            )

            results = [
                Retrieved(
                    chunk_id=item[0],
                    score=float(item[1]),
                    source_score=0.0,
                    vector_score=float(item[1]),
                    metadata=item[2] or {},
                )
                for item in vector_results
            ]

        # =====================================================
        # HYBRID RETRIEVAL
        # =====================================================

        else:
            vector_results = self._vector_search(
                query,
                top_k=top_k * 3,
                user_id=user_id,
            )

            # A fresh Retriever does not have the BM25 index in memory.
            # Restore it from persisted metadata.
            self._restore_bm25_from_vector_store(user_id=user_id)

            bm25_results = self._bm25_search(
                query,
                top_k=top_k * 3,
                user_id=user_id,
            )

            # -------------------------------------------------
            # IMPORTANT HYBRID RETRIEVAL FIX
            #
            # If BM25 has no positive lexical match but the
            # semantic search found results, do NOT reduce
            # those semantic scores using a zero BM25 score.
            #
            # Previous behavior:
            #
            # vector = 1.0
            # BM25   = 0.0
            #
            # combined:
            #
            # 1.0 * 0.5 + 0.0 * 0.5 = 0.5
            #
            # This makes a strong semantic match appear weaker
            # even though BM25 simply had no lexical match.
            # -------------------------------------------------

            positive_bm25 = [
                result
                for result in bm25_results
                if float(result[1]) > 0.0
            ]

            if (
                not positive_bm25
                and vector_results
            ):
                results = [
                    Retrieved(
                        chunk_id=item[0],
                        score=float(item[1]),
                        source_score=0.0,
                        vector_score=float(item[1]),
                        metadata=item[2] or {},
                    )
                    for item in vector_results[
                        :top_k
                    ]
                ]

                logger.debug(
                    "BM25 produced no positive lexical "
                    "matches; using semantic vector "
                    "results directly."
                )

            else:
                results = combine_scores(
                    bm25_results,
                    vector_results,
                    bm25_weight=getattr(
                        settings,
                        "bm25_weight",
                        0.5,
                    ),
                    vector_weight=getattr(
                        settings,
                        "vector_weight",
                        0.5,
                    ),
                    top_k=top_k,
                )

        # =====================================================
        # NO RERANKING
        # =====================================================

        if (
            not rerank
            or not self.reranker
        ):
            return results

        # =====================================================
        # PREPARE RERANKER INPUT
        # =====================================================

        rerank_items = []

        for result in results:
            chunk_id = result.chunk_id

            if chunk_id in self.chunk_index:
                metadata = {
                    "text": self.chunk_index[
                        chunk_id
                    ].text,
                    **self.chunk_index[
                        chunk_id
                    ].metadata,
                }

            else:
                metadata = (
                    result.metadata
                    or {}
                )

            rerank_items.append(
                (
                    chunk_id,
                    result.score,
                    metadata,
                )
            )

        if not rerank_items:
            return results

        # =====================================================
        # RERANK
        # =====================================================

        reranked = self.reranker.rerank(
            query,
            rerank_items,
        )

        if not reranked:
            logger.warning(
                "Reranker returned no results; "
                "preserving original retrieval results."
            )

            return results

        # =====================================================
        # PRESERVE ORIGINAL RETRIEVAL SCORES
        # =====================================================

        original_by_id = {
            result.chunk_id: result
            for result in results
        }

        final_results: List[
            Retrieved
        ] = []

        for item in reranked:
            chunk_id = item[0]
            rerank_score = float(
                item[1]
            )
            metadata = item[2] or {}

            original = original_by_id.get(
                chunk_id
            )

            if original is not None:
                final_results.append(
                    Retrieved(
                        chunk_id=chunk_id,
                        score=rerank_score,
                        source_score=original.source_score,
                        vector_score=original.vector_score,
                        metadata=metadata,
                    )
                )

            else:
                final_results.append(
                    Retrieved(
                        chunk_id=chunk_id,
                        score=rerank_score,
                        source_score=0.0,
                        vector_score=0.0,
                        metadata=metadata,
                    )
                )

        return final_results

    def agentic_retrieval_loop(
        self,
        query: str,
        *,
        user_id: str,
        relevant_ids: Iterable[str] = (),
        max_iters: Optional[int] = None,
    ) -> List[RetrievalResult]:

        max_iters = (
            max_iters
            or getattr(
                settings,
                "max_retrieval_iterations",
                3,
            )
        )

        seen_chunk_ids = set()

        results: List[
            RetrievalResult
        ] = []

        current_query = query

        for iteration in range(
            1,
            max_iters + 1,
        ):
            if getattr(
                settings,
                "rewriting_enabled",
                True,
            ):
                rewritten = _rewrite_query(
                    current_query
                )

            else:
                rewritten = current_query

            if getattr(
                settings,
                "decomposition_enabled",
                True,
            ):
                subqueries = _decompose_query(
                    rewritten
                )

            else:
                subqueries = [
                    rewritten
                ]

            iteration_retrieved = []

            for subquery in subqueries:
                retrieved = self.retrieve(
                    subquery,
                    user_id=user_id,
                    top_k=settings.top_k_retrieval,
                    hybrid=True,
                    rerank=getattr(
                        settings,
                        "reranking_enabled",
                        True,
                    ),
                )

                new_results = [
                    result
                    for result in retrieved
                    if result.chunk_id
                    not in seen_chunk_ids
                ]

                for result in new_results:
                    seen_chunk_ids.add(
                        result.chunk_id
                    )

                iteration_retrieved.extend(
                    new_results
                )

            metrics = _evaluate_retrieval(
                iteration_retrieved,
                relevant_ids,
                k=settings.top_k_retrieval,
            )

            if (
                metrics["recall@k"] >= 0.95
                or len(iteration_retrieved)
                >= settings.top_k_retrieval
            ):
                stop_reason = (
                    "sufficient_evidence"
                )

            elif iteration == max_iters:
                stop_reason = (
                    "max_iterations_reached"
                )

            else:
                stop_reason = None

            results.append(
                RetrievalResult(
                    query=current_query,
                    retrieved=iteration_retrieved,
                    iteration=iteration,
                    stop_reason=stop_reason,
                )
            )

            if stop_reason is not None:
                break

            if iteration_retrieved:
                evidence = (
                    iteration_retrieved[
                        0
                    ].metadata.get(
                        "text",
                        "",
                    )[:200]
                )

                current_query = (
                    f"{query} {evidence}"
                )

            else:
                results[-1].stop_reason = (
                    "no_new_evidence"
                )

                break

        return results