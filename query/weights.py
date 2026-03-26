from query.analyser import QueryAnalysis
from dataclasses import dataclass


@dataclass
class QueryWeights:
    alpha: float   # coarse weight
    beta: float    # mid weight
    gamma: float   # fine weight
    lam: float     # temporal decay λ


_WEIGHT_TABLE: dict[tuple[str, str], QueryWeights] = {
    # shallow queries — broad strokes
    ("shallow", "any"):        QueryWeights(0.50, 0.35, 0.15, 0.0),
    ("shallow", "recent"):     QueryWeights(0.45, 0.35, 0.20, 0.3),
    ("shallow", "historical"): QueryWeights(0.45, 0.35, 0.20, 0.5),
    ("shallow", "timeless"):   QueryWeights(0.50, 0.35, 0.15, 0.0),

    # medium queries — balanced
    ("medium", "any"):         QueryWeights(0.30, 0.40, 0.30, 0.0),
    ("medium", "recent"):      QueryWeights(0.25, 0.40, 0.35, 0.5),
    ("medium", "historical"):  QueryWeights(0.25, 0.40, 0.35, 0.8),
    ("medium", "timeless"):    QueryWeights(0.30, 0.40, 0.30, 0.0),

    # deep queries — fine grained, high specificity
    ("deep", "any"):           QueryWeights(0.15, 0.35, 0.50, 0.0),
    ("deep", "recent"):        QueryWeights(0.10, 0.30, 0.60, 0.7),
    ("deep", "historical"):    QueryWeights(0.10, 0.30, 0.60, 1.0),
    ("deep", "timeless"):      QueryWeights(0.15, 0.35, 0.50, 0.0),
}

_DEFAULT = QueryWeights(0.30, 0.40, 0.30, 0.0)


def resolve(analysis: QueryAnalysis) -> QueryWeights:
    key = (analysis.semantic_depth, analysis.temporal_intent)
    weights = _WEIGHT_TABLE.get(key, _DEFAULT)

    total = weights.alpha + weights.beta + weights.gamma
    if abs(total - 1.0) > 1e-6:
        weights = QueryWeights(
            alpha=weights.alpha / total,
            beta=weights.beta / total,
            gamma=weights.gamma / total,
            lam=weights.lam,
        )

    return weights