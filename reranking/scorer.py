import numpy as np
from typing import List, Optional
from query.encoder import EncodedQuery


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _level_score(
    query_tags: List[tuple],   
    doc_tags: List[tuple],     
) -> float:
    """
    score = Σ_i Σ_j cos(query[i], doc[j]) x doc_conf[j]
    then normalized by no of pairs to be bounded
    """
    if not query_tags or not doc_tags:
        return 0.0

    total = 0.0
    for _, _, qvec in query_tags:
        for _, doc_conf, dvec in doc_tags:
            total += _cosine(qvec, dvec) * doc_conf

    return total / (len(query_tags) * len(doc_tags))


def _temporal_score(
    query_anchor: Optional[int],
    doc_anchor: Optional[int],
    lam: float,
) -> float:
    if lam == 0.0 or query_anchor is None or doc_anchor is None:
        return 1.0
    return float(np.exp(-lam * abs(query_anchor - doc_anchor)))


def score_chunk(
    eq: EncodedQuery,
    doc_level_embeddings: dict[str, List[tuple]], 
    doc_anchor_year: Optional[int],
) -> float:
    w = eq.weights

    coarse_score = _level_score(eq.level_embeddings["coarse"], doc_level_embeddings["coarse"])
    mid_score    = _level_score(eq.level_embeddings["mid"],    doc_level_embeddings["mid"])
    fine_score   = _level_score(eq.level_embeddings["fine"],   doc_level_embeddings["fine"])

    semantic_score = w.alpha * coarse_score + w.beta * mid_score + w.gamma * fine_score

    temporal = _temporal_score(
        query_anchor=eq.analysis.anchor_year,
        doc_anchor=doc_anchor_year,
        lam=w.lam,
    )

    return semantic_score + temporal