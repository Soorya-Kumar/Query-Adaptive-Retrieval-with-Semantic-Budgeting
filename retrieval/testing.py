from query.encoder import EncodedQuery
from retrieval.dense_retriever import dense_retrieve
from retrieval.sparse_retriever import sparse_retrieve
from retrieval.fusion import rrf_fusion
from typing import List

from utils import color_print

def retrieve(eq: EncodedQuery, top_k: int = 500) -> List[dict]:
    color_print("retrieving candidates...")
    dense = dense_retrieve(eq, top_k=top_k)
    
    color_print("dense retrieval complete.")
    sparse = sparse_retrieve(eq, top_k=top_k)
    
    color_print("sparse retrieval complete.")
    return rrf_fusion(dense, sparse, top_k=top_k)


