from dataclasses import dataclass
from typing import List
from ingestion.loader import Document

CHUNK_SIZE = 512      # tokens approx (chars / 4)
CHUNK_OVERLAP = 64   # overlap between consecutive chunks


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    raw_text: str
    chunk_index: int
    metadata: dict


def _split_text(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def chunk_document(doc: Document, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    # Convert approx token size to chars
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    parts = _split_text(doc.raw_text, char_size, char_overlap)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}__chunk{i}",
            doc_id=doc.doc_id,
            raw_text=part,
            chunk_index=i,
            metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(parts)},
        )
        for i, part in enumerate(parts)
    ]