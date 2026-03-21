from query.encoder import EncodedQuery
from storage.store_postgres import bm25_search
from typing import List


def sparse_retrieve(eq: EncodedQuery, top_k: int = 500) -> List[dict]:
    """BM25 search over descriptor tags using the raw query string."""
    return bm25_search(query=eq.raw, top_k=top_k)