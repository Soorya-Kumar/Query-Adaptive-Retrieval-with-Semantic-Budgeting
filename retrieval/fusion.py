from typing import List

RRF_K = 60  # standard constant, higher = smoother rank fusion


def rrf_fusion(
    dense_results: List[dict],
    sparse_results: List[dict],
    top_k: int = 500,
) -> List[dict]:
    """
    Reciprocal Rank Fusion:
        score = 1/(k + rank_dense) + 1/(k + rank_sparse)

    Chunks only in one list still get a partial score.
    """
    
    print(f"Fusing {len(dense_results)} dense and {len(sparse_results)} sparse results with RRF (k={RRF_K})")
    scores: dict[str, float] = {}

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    for rank, item in enumerate(sparse_results, start=1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [{"chunk_id": cid, "rrf_score": score} for cid, score in ranked[:top_k]]