from query.encoder import EncodedQuery
from storage.store_vector import search
from typing import List


def dense_retrieve(eq: EncodedQuery, top_k: int = 500) -> List[dict]:
    analysis = eq.analysis

    anchor_year_range = None
    if analysis.anchor_year:
        anchor_year_range = (analysis.anchor_year, analysis.anchor_year)
    elif analysis.anchor_year_range:
        anchor_year_range = analysis.anchor_year_range

    return search(
        query_vector=eq.pooled_vector,
        top_k=top_k,
        relativity_class=analysis.relativity_class,
        anchor_year_range=anchor_year_range,
    )