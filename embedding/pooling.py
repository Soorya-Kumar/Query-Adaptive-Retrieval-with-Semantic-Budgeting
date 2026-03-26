from typing import List
import numpy as np
from descriptors.schema import ChunkDescriptor, TagWithConfidence
from embedding.encoding import embed


DEFAULT_WEIGHTS = {"coarse": 0.2, "mid": 0.35, "fine": 0.45}


def _pool_level(tags: List[TagWithConfidence]) -> np.ndarray | None:
    """Confidence-weighted mean of embeddings for one granularity level."""
    if not tags:
        return None

    embeddings = [np.array(embed(t.tag)) for t in tags]
    confidences = np.array([t.confidence for t in tags])
    confidences /= confidences.sum()  # normalize

    stacked = np.stack(embeddings)  # (n_tags, dim)
    return (stacked * confidences[:, None]).sum(axis=0)


def pool(
    descriptor: ChunkDescriptor,
    alpha: float = DEFAULT_WEIGHTS["coarse"],
    beta: float = DEFAULT_WEIGHTS["mid"],
    gamma: float = DEFAULT_WEIGHTS["fine"],
) -> np.ndarray:
    """
    pooled = alpha * coarse_pool + beta * mid_pool + gamma * fine_pool
    Falls back gracefully if a level is empty.
    """
    coarse_vec = _pool_level(descriptor.coarse)
    mid_vec = _pool_level(descriptor.mid)
    fine_vec = _pool_level(descriptor.fine)

    # Determine actual dim from first non-None vector
    dim = next(v.shape[0] for v in [coarse_vec, mid_vec, fine_vec] if v is not None)

    coarse_vec = coarse_vec if coarse_vec is not None else np.zeros(dim)
    mid_vec = mid_vec if mid_vec is not None else np.zeros(dim)
    fine_vec = fine_vec if fine_vec is not None else np.zeros(dim)

    pooled = alpha * coarse_vec + beta * mid_vec + gamma * fine_vec

    # L2 normalize for cosine similarity downstream
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled /= norm

    return pooled


def pool_query(
    descriptor: ChunkDescriptor,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Same as pool() but weights are always query-provided (no defaults)."""
    return pool(descriptor, alpha, beta, gamma)