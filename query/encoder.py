import numpy as np
from dataclasses import dataclass

from descriptors.extractor import extract_descriptors
from descriptors.postprocessing import postprocess
from descriptors.schema import ChunkDescriptor
from embedding.encoding import embed
from embedding.pooling import pool_query
from query.analyser import QueryAnalysis, analyze
from query.weights import QueryWeights, resolve


@dataclass
class EncodedQuery:
    raw: str
    analysis: QueryAnalysis
    weights: QueryWeights
    descriptor: ChunkDescriptor
    # Per-level embeddings for reranking: {level: [(tag, conf, vector), ...]}
    level_embeddings: dict[str, list]
    pooled_vector: np.ndarray


def encode(query: str) -> EncodedQuery:
    
    analysis = analyze(query)
    weights = resolve(analysis)

    raw_descriptor = extract_descriptors(query)
    descriptor = postprocess(raw_descriptor)

    #embed each descriptor
    level_embeddings: dict[str, list] = {"fine": [], "mid": [], "coarse": []}
    for level in ("fine", "mid", "coarse"):
        tags = getattr(descriptor, level)
        for t in tags:
            vec = embed(t.tag)
            level_embeddings[level].append((t.tag, t.confidence, np.array(vec)))

    #compute pooled query vector
    pooled_vector = pool_query(
        descriptor,
        alpha=weights.alpha,
        beta=weights.beta,
        gamma=weights.gamma,
    )

    return EncodedQuery(
        raw=query,
        analysis=analysis,
        weights=weights,
        descriptor=descriptor,
        level_embeddings=level_embeddings,
        pooled_vector=pooled_vector,
    )