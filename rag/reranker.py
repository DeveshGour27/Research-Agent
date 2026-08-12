"""
Reranker module.

Responsibilities:

- Reorder retrieved candidates by relevance.
- Provide lexical BM25 reranking.
- Provide optional LLM-based reranking.
- Provide a no-op reranker for disabling reranking.
- Fail safely when an LLM reranker cannot produce valid output.

Design:

The LLM reranker is an ordering component. It does not invent
candidate IDs and does not remove valid candidates.

All candidates are preserved in the final output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# JSON extraction / repair helpers for LLM responses
#
# These are intentionally standalone functions (not methods) so they can be
# reasoned about and tested in isolation from the reranking logic itself.
# ---------------------------------------------------------------------------

def _try_parse_json_candidate(candidate: str):
    """
    Attempt to parse a single candidate substring as JSON.

    Applies two bounded, deterministic repairs only when the strict
    parse fails -- this does not loosen validation of IDs/scores
    downstream, it only widens what counts as "recoverable JSON text":

    1. Strip a single trailing comma immediately before a closing
       ``]``/``}`` (a common benign LLM formatting slip).
    2. Fall back to :func:`ast.literal_eval`, which safely parses
       Python-literal syntax (e.g. single-quoted strings) without
       executing any code. The result is rejected unless it is a
       ``list`` or ``dict`` made only of JSON-compatible primitive
       types, so this cannot smuggle through arbitrary objects.

    Returns the parsed value, or ``None`` if nothing could be recovered.
    """

    import json
    import re

    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Repair 1: trailing comma before a closing bracket/brace.
    repaired = re.sub(r",\s*([\]}])", r"\1", candidate)
    if repaired != candidate:
        try:
            return json.loads(repaired)
        except Exception:
            pass

    # Repair 2: Python-literal-style (e.g. single-quoted) pseudo-JSON.
    try:
        import ast

        value = ast.literal_eval(candidate)
    except Exception:
        return None

    def _is_json_safe(obj) -> bool:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return True
        if isinstance(obj, list):
            return all(_is_json_safe(item) for item in obj)
        if isinstance(obj, dict):
            return all(
                isinstance(k, str) and _is_json_safe(v)
                for k, v in obj.items()
            )
        return False

    if isinstance(value, (list, dict)) and _is_json_safe(value):
        return value

    return None


def _extract_fenced_json(content: str):
    """
    Extract and parse JSON from the first ```/```json fenced code block
    in ``content``, if any. Returns the parsed value, or ``None``.
    """

    import re

    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        content,
        re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return None

    return _try_parse_json_candidate(match.group(1).strip())


def _extract_json_arrays(content: str) -> Optional[list]:
    """
    Find every complete, balanced top-level ``[...]`` substring in
    ``content`` via a bracket-depth scan (not a greedy regex, which can
    merge unrelated bracket groups spread across the text into one
    invalid blob) and return the first one -- scanning from the last
    substring found backwards -- that parses as a JSON array.

    The last complete bracket group is tried first because the actual
    answer normally follows any preamble/commentary the model added
    despite being asked not to.

    Returns ``None`` if no candidate substring parses successfully.
    """

    candidates: List[str] = []
    depth = 0
    start = None

    for index, char in enumerate(content):
        if char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(content[start : index + 1])
                    start = None

    for candidate in reversed(candidates):
        parsed = _try_parse_json_candidate(candidate)
        if isinstance(parsed, list):
            return parsed

    return None


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    class BM25Okapi:
        """
        Small deterministic fallback implementation.

        This is intentionally simple and exists only so the project
        remains usable when rank_bm25 is unavailable.
        """

        def __init__(self, docs: List[List[str]]):
            self.docs = docs

        def get_scores(
            self,
            query_tokens: List[str],
        ) -> List[float]:
            scores = []

            for doc in self.docs:
                score = 0.0

                for token in query_tokens:
                    score += doc.count(token)

                scores.append(score)

            return scores


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class Reranker(ABC):
    """
    Common reranker interface.

    Each item is:

        (chunk_id, retrieval_score, metadata)
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        items: Iterable[
            Tuple[str, float, dict]
        ],
    ) -> List[
        Tuple[str, Optional[float], dict]
    ]:
        """
        Return candidates reordered by relevance.

        The returned score may represent the reranker's own
        relevance score.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Lexical reranker
# ---------------------------------------------------------------------------

class LexicalReranker(Reranker):
    """
    BM25-based lexical reranker.

    Expects:

        metadata["text"]

    to contain the candidate document text.
    """

    def rerank(
        self,
        query: str,
        items: Iterable[
            Tuple[str, float, dict]
        ],
    ) -> List[
        Tuple[str, Optional[float], dict]
    ]:
        items = list(items)

        if not items:
            return []

        texts = [
            str(item[2].get("text", ""))
            for item in items
        ]

        tokenized = [
            text.split()
            for text in texts
        ]

        if not any(tokenized):
            return items

        bm25 = BM25Okapi(tokenized)

        scores = bm25.get_scores(
            query.split()
        )

        scored = []

        for index, item in enumerate(items):
            scored.append(
                (
                    item[0],
                    float(scores[index]),
                    item[2],
                )
            )

        # Tie-breaking: rerank_score DESC -> hybrid_score DESC -> chunk_id ASC
        scored.sort(
            key=lambda item: (
                -item[1],
                -item[2].get("hybrid_score", 0.0),
                item[0]
            )
        )

        return scored


# ---------------------------------------------------------------------------
# LLM-backed reranker
# ---------------------------------------------------------------------------

class NeuralReranker(Reranker):
    """
    LLM-backed reranker.

    Despite the historical class name, this implementation uses
    the configured chat model to rank candidate IDs.

    The model is asked to return JSON, but the implementation is
    intentionally defensive because LLM output is not guaranteed
    to follow the requested format.

    Failure behavior:

        LLM valid JSON
            -> use LLM ordering

        LLM invalid/empty response
            -> fall back to lexical reranking

        LLM returns unknown IDs
            -> ignore unknown IDs

        LLM returns duplicate IDs
            -> ignore duplicates

        LLM omits valid candidates
            -> append omitted candidates in original order
    """

    def rerank(
        self,
        query: str,
        items: Iterable[
            Tuple[str, float, dict]
        ],
    ) -> List[
        Tuple[str, Optional[float], dict]
    ]:
        from app.config import settings
        from app.llm.factory import (
            create_chat_provider,
        )
        from app.llm.base import ChatMessage
        from app.logger import get_logger

        logger = get_logger(__name__)

        candidates = list(items)

        if not candidates:
            return []

        try:
            provider = create_chat_provider(
                settings
            )

            # ---------------------------------------------------------
            # Build candidate list
            # ---------------------------------------------------------

            snippet_lines = []

            for chunk_id, score, metadata in candidates:
                text = str(
                    metadata.get(
                        "text",
                        "",
                    )
                )

                text = (
                    text[:500]
                    .replace("\n", " ")
                    .replace("\r", " ")
                )

                snippet_lines.append(
                    f"ID: {chunk_id}\n"
                    f"TEXT: {text}"
                )

            # Strong structured-output prompt: require only a JSON array with no
            # surrounding prose or markdown. Be explicit about forbidden output.
            prompt = (
                "You are a retrieval reranker.\n\n"
                "Task: rank the candidate document IDs by relevance to the query.\n\n"
                "IMPORTANT OUTPUT FORMAT (read carefully):\n"
                "- Return ONLY a single JSON array and NOTHING ELSE.\n"
                "- The array must contain the candidate IDs (strings) in descending order of relevance.\n"
                "- Do NOT include explanations, headings, or markdown code fences.\n"
                "- Do NOT create or invent IDs that are not in the provided candidates.\n"
                "- If you cannot produce a valid JSON array, return an empty response (so the system can fall back).\n\n"
                f"QUERY:\n{query}\n\n"
                "CANDIDATES (ID followed by a short text snippet):\n\n"
                + "\n\n".join(
                    snippet_lines
                )
                + "\n\n"
                "Return the JSON array now (eg. [\"id2\", \"id1\"])."
            )

            response = provider.generate(
                [
                    ChatMessage(
                        role="user",
                        content=prompt,
                    )
                ]
            )

            content = (getattr(response, "content", "") or "")

            # Normalize whitespace but preserve raw content for parsing attempts.
            content_str = content.strip()

            # ---------------------------------------------------------
            # Empty response
            # ---------------------------------------------------------

            if not content_str:
                raise ValueError(
                    "LLM reranker returned an empty response."
                )

            # ---------------------------------------------------------
            # Parse JSON robustly: support raw JSON, JSON inside ``` fences,
            # or a JSON array appearing somewhere in surrounding text.
            # ---------------------------------------------------------

            import json

            ranked_ids = None

            # 1) Try direct JSON parse of the whole (trimmed) response.
            try:
                parsed = json.loads(content_str)
                if isinstance(parsed, (list, dict)):
                    ranked_ids = parsed
            except Exception:
                ranked_ids = None

            # 2) Try extracting a fenced code block (``` or ```json).
            if ranked_ids is None:
                ranked_ids = _extract_fenced_json(content)

            # 3) Bracket-depth scan for complete top-level array substrings
            if ranked_ids is None:
                ranked_ids = _extract_json_arrays(content)

            if ranked_ids is None:
                logger.warning(
                    "NeuralReranker could not parse any JSON from the "
                    "LLM response. Raw content (truncated to 1000 "
                    "chars): %r",
                    content[:1000],
                )
                raise ValueError(
                    "LLM reranker returned invalid or unparsable JSON."
                )

            # ---------------------------------------------------------
            # Unwrap a JSON-object envelope, e.g. {"ranking": [...]}.
            # ---------------------------------------------------------

            if isinstance(ranked_ids, dict):
                envelope_keys = (
                    "ranking",
                    "rankings",
                    "ranked_ids",
                    "ranked_ids_list",
                    "ids",
                    "results",
                    "order",
                )

                unwrapped = None

                for key in envelope_keys:
                    value = ranked_ids.get(key)
                    if isinstance(value, list):
                        unwrapped = value
                        break

                if unwrapped is None:
                    logger.warning(
                        "NeuralReranker received a JSON object response "
                        "with no recognized array field. Keys present: "
                        "%r. Raw content (truncated to 1000 chars): %r",
                        list(ranked_ids.keys()),
                        content[:1000],
                    )
                    raise ValueError(
                        "LLM reranker response must be a JSON array "
                        "(or an object containing one under a "
                        "recognized key)."
                    )

                ranked_ids = unwrapped

            if not isinstance(ranked_ids, list):
                raise ValueError(
                    "LLM reranker response must be a JSON array."
                )

            # ---------------------------------------------------------
            # Validate IDs
            # ---------------------------------------------------------

            id_map = {item[0]: item for item in candidates}

            ranked: List[Tuple[str, Optional[float], dict]] = []
            seen = set()

            for entry in ranked_ids:
                rid: str | None = None
                entry_score: float | None = None

                if isinstance(entry, str):
                    rid = entry.strip()

                elif isinstance(entry, dict):
                    raw_id = entry.get("id")

                    if not isinstance(raw_id, str) or not raw_id.strip():
                        logger.warning(
                            "LLM reranker returned a ranking object "
                            "with a missing/invalid 'id': %r",
                            entry,
                        )
                        continue

                    rid = raw_id.strip()

                    if "score" in entry:
                        raw_score = entry.get("score")

                        if isinstance(raw_score, bool) or not isinstance(
                            raw_score, (int, float)
                        ):
                            logger.warning(
                                "LLM reranker returned a non-numeric "
                                "score for id %s; ignoring the score.",
                                rid,
                            )
                        else:
                            entry_score = float(raw_score)

                else:
                    logger.warning(
                        "LLM reranker returned a ranking entry of an "
                        "unsupported type: %r",
                        entry,
                    )
                    continue

                if not rid or rid in seen:
                    continue

                if rid not in id_map:
                    logger.warning(
                        "LLM reranker returned unknown candidate ID: %s",
                        rid,
                    )
                    continue

                seen.add(rid)

                chunk_id, original_score, metadata = id_map[rid]

                # Semantic score mapping:
                # If a valid numeric score was provided, use it.
                # Otherwise, rerank_score is explicitly None, signifying NO score.
                final_score = entry_score

                ranked.append(
                    (chunk_id, final_score, metadata)
                )

            # Append any omitted candidates
            for item in candidates:
                if item[0] not in seen:
                    # original item[1] is hybrid score. But for rerank output,
                    # we must return rerank_score=None when no score is given.
                    ranked.append((item[0], None, item[2]))

            if not ranked:
                logger.warning(
                    "NeuralReranker parsed valid JSON but no entries "
                    "survived ID/score validation. Raw content "
                    "(truncated to 1000 chars): %r",
                    content[:1000],
                )
                raise ValueError("LLM reranker produced no valid candidates.")

            return ranked

        except Exception as exc:
            logger.warning(
                "NeuralReranker failed; "
                "falling back to lexical: %s",
                exc,
            )

            # Important:
            # The lexical fallback receives the same candidates
            # and therefore remains deterministic.
            return LexicalReranker().rerank(
                query,
                candidates,
            )


# ---------------------------------------------------------------------------
# No-op reranker
# ---------------------------------------------------------------------------

class NoopReranker(Reranker):
    """
    Disables reranking while preserving retrieval order and scores.
    """

    def rerank(
        self,
        query: str,
        items: Iterable[
            Tuple[str, float, dict]
        ],
    ) -> List[
        Tuple[str, Optional[float], dict]
    ]:
        return list(items)