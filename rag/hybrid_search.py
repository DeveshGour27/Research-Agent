"""
Hybrid Search module.

Responsibilities:
- Combine BM25 lexical retrieval with vector similarity scores
- Deduplicate and rank results by configurable weights
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Retrieved:
    chunk_id: str
    score: float
    source_score: float
    vector_score: float
    metadata: dict


def combine_scores(bm25_results: List[Tuple[str, float, dict]],
                   vector_results: List[Tuple[str, float, dict]],
                   bm25_weight: float = 0.5,
                   vector_weight: float = 0.5,
                   top_k: int = 5) -> List[Retrieved]:
    """Combine two ranked lists into a single ranked list.

    bm25_results and vector_results are lists of tuples (chunk_id, score, metadata).
    We normalize each source's scores to [0,1] and then compute weighted sum.
    """
    # Build score maps
    bm25_map = {r[0]: float(r[1]) for r in bm25_results}
    vec_map = {r[0]: float(r[1]) for r in vector_results}

    # Normalise
    def _norm(m):
        if not m:
            return {}
        vals = list(m.values())
        mn = min(vals)
        mx = max(vals)
        if mx == mn:
            return {k: 1.0 for k in m}
        return {k: (v - mn) / (mx - mn) for k, v in m.items()}

    nb = _norm(bm25_map)
    nv = _norm(vec_map)

    # metadata maps to preserve per-source metadata
    bm25_meta = {r[0]: (r[2] if len(r) > 2 else {}) for r in bm25_results}
    vec_meta = {r[0]: (r[2] if len(r) > 2 else {}) for r in vector_results}

    all_ids = list(set(list(nb.keys()) + list(nv.keys())))

    results: List[Retrieved] = []
    for cid in all_ids:
        bscore = nb.get(cid, 0.0)
        vscore = nv.get(cid, 0.0)
        score = bm25_weight * bscore + vector_weight * vscore
        # prefer bm25 metadata when available, otherwise use vector metadata
        metadata = bm25_meta.get(cid) if cid in bm25_meta else vec_meta.get(cid, {})
        results.append(Retrieved(chunk_id=cid, score=score, source_score=bscore, vector_score=vscore, metadata=metadata))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
