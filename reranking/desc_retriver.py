from typing import List
from query.encoder import EncodedQuery
from reranking.scorer import score_chunk
from storage.store_postgres import fetch_descriptor_embeddings, fetch_chunk_metadata


def rerank(eq: EncodedQuery, candidates: List[dict], top_k: int = 20) -> List[dict]:
    """
    candidates: output of retrieval pipeline [{"chunk_id": ..., "rrf_score": ...}]
    Returns top_k chunks sorted by final_score descending.
    """
    scored = []

    for item in candidates:
        chunk_id = item["chunk_id"]

        # Fetch individual descriptor embeddings from Postgres
        doc_embeddings = fetch_descriptor_embeddings(chunk_id)

        # Fetch anchor_year for temporal scoring
        metadata = fetch_chunk_metadata(chunk_id)
        doc_anchor_year = metadata.get("anchor_year") if metadata else None

        final_score = score_chunk(
            eq=eq,
            doc_level_embeddings=doc_embeddings,
            doc_anchor_year=doc_anchor_year,
        )

        scored.append({
            "chunk_id": chunk_id,
            "rrf_score": item["rrf_score"],
            "final_score": final_score,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k]