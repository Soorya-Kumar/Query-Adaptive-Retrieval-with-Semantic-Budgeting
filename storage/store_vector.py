from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)
from typing import List, Optional
import numpy as np

QDRANT_URL = "http://localhost:6333"
COLLECTION = "chunks"
VECTOR_DIM = 768  # match your embed model


client = QdrantClient(url=QDRANT_URL)


def init_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def upsert_chunk(
    chunk_id: str,
    pooled_vector: np.ndarray,
    doc_id: str,
    anchor_year: Optional[int],
    relativity_class: str,
):
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=_id_from_str(chunk_id),
                vector=pooled_vector.tolist(),
                payload={
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "anchor_year": anchor_year,
                    "relativity_class": relativity_class,
                }
            )
        ]
    )


def search(
    query_vector: np.ndarray,
    top_k: int = 500,
    doc_id: Optional[str] = None,
    relativity_class: Optional[str] = None,
    anchor_year_range: Optional[tuple[int, int]] = None,
) -> List[dict]:
    must = []

    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

    if relativity_class:
        must.append(FieldCondition(key="relativity_class", match=MatchValue(value=relativity_class)))

    if anchor_year_range:
        must.append(FieldCondition(
            key="anchor_year",
            range=Range(gte=anchor_year_range[0], lte=anchor_year_range[1])
        ))

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector.tolist(),
        limit=top_k,
        query_filter=Filter(must=must) if must else None,
        with_payload=True,
    )
    return [
        {"chunk_id": r.payload["chunk_id"], "score": r.score}
        for r in results.points
    ]
    

def _id_from_str(chunk_id: str) -> int:
    """Qdrant needs int/uuid IDs — hash the string."""
    return abs(hash(chunk_id)) % (2**53)