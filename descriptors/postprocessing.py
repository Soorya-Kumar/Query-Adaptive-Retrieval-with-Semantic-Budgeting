from Levenshtein import distance
from descriptors.schema import ChunkDescriptor, TagWithConfidence
from typing import List

CONFIDENCE_THRESHOLD = 0.5
LEVENSHTEIN_THRESHOLD = 3  # tags closer than this are considered duplicates


def _dedup(tags: List[TagWithConfidence]) -> List[TagWithConfidence]:
    kept = []
    for candidate in tags:
        is_dup = any(
            distance(candidate.tag.lower(), k.tag.lower()) < LEVENSHTEIN_THRESHOLD
            for k in kept
        )
        if not is_dup:
            kept.append(candidate)
    return kept

def postprocess(descriptor: ChunkDescriptor) -> ChunkDescriptor:
    def filter_and_dedup(tags: List[TagWithConfidence]) -> List[TagWithConfidence]:
        filtered = [t for t in tags if t.confidence >= CONFIDENCE_THRESHOLD]
        return _dedup(filtered)

    return ChunkDescriptor(
        fine=filter_and_dedup(descriptor.fine),
        mid=filter_and_dedup(descriptor.mid),
        coarse=filter_and_dedup(descriptor.coarse),
        anchor_year=descriptor.anchor_year,
        relativity_class=descriptor.relativity_class,
    )