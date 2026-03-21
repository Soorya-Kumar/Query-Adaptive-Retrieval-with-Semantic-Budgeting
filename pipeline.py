import numpy as np
from typing import List, Optional

from utils import color_print

from ingestion.loader import load_txt, load_directory, Document
from ingestion.chunker import chunk_document, Chunk
from descriptors.testing import run as extract_descriptors
from embedding.encoding import embed
from embedding.pooling import pool
from storage.store_vector import init_collection, upsert_chunk
from storage.store_postgres import (
    insert_chunk_metadata,
    insert_descriptors,
)
from query.encoder import encode
from retrieval.testing import retrieve
from reranking.desc_retriver import rerank


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_chunk(chunk: Chunk):

    # 1. Extract + postprocess descriptors
    descriptor = extract_descriptors(chunk.raw_text)
    color_print("Extracted descriptors")

    # 2. Embed each descriptor tag per level
    level_embeddings: dict[str, list] = {"fine": [], "mid": [], "coarse": []}
    for level in ("fine", "mid", "coarse"):
        tags = getattr(descriptor, level)
        for t in tags:
            vec = embed(t.tag)
            level_embeddings[level].append((t.tag, t.confidence, vec))
    color_print("Embedded descriptors")

    # 3. Compute pooled vector (default weights at index time)
    pooled_vector = pool(descriptor)
    color_print("Pooled vector")

    # 4. Store metadata in Postgres
    insert_chunk_metadata(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        raw_text=chunk.raw_text,
        anchor_year=descriptor.anchor_year,
        relativity_class=descriptor.relativity_class.value,
    )
    color_print("Stored chunk metadata in Postgres")

    # 5. Store descriptor tags + embeddings in Postgres
    insert_descriptors(
        chunk_id=chunk.chunk_id,
        descriptor=descriptor,
        embeddings=level_embeddings,
    )
    color_print("Stored descriptors in Postgres")

    # 6. Upsert pooled vector + metadata payload into Qdrant
    upsert_chunk(
        chunk_id=chunk.chunk_id,
        pooled_vector=pooled_vector,
        doc_id=chunk.doc_id,
        anchor_year=descriptor.anchor_year,
        relativity_class=descriptor.relativity_class.value,
    )
    color_print("Upserted chunk vector into Qdrant")


def ingest_file(path: str):
    init_collection()
    doc = load_txt(path)
    chunks = chunk_document(doc)
    print(f"Ingesting {doc.doc_id}: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {chunk.chunk_id}")
        ingest_chunk(chunk)
    print(f"Done: {doc.doc_id}")
    color_print("Ingestion file complete.")

def ingest_directory(dir_path: str):
    init_collection()
    docs = load_directory(dir_path)
    print(f"Found {len(docs)} documents")
    for doc in docs:
        chunks = chunk_document(doc)
        print(f"Ingesting {doc.doc_id}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  [{i+1}/{len(chunks)}] {chunk.chunk_id}")
            ingest_chunk(chunk)
    print("Ingestion complete.")
    color_print("Ingestion directory complete.")



# ── Query ─────────────────────────────────────────────────────────────────────

def query(
    text: str,
    retrieve_top_k: int = 500,
    rerank_top_k: int = 20,
) -> List[dict]:
    # 1. Encode query (analyze + weight + embed)
    eq = encode(text)

    print(f"Query: {text!r}")
    print(f"  depth={eq.analysis.semantic_depth}  intent={eq.analysis.intent}  temporal={eq.analysis.temporal_intent}")
    print(f"  weights: α={eq.weights.alpha:.2f} β={eq.weights.beta:.2f} γ={eq.weights.gamma:.2f} λ={eq.weights.lam:.2f}")

    # 2. Retrieve (dense + sparse + RRF)
    candidates = retrieve(eq, top_k=retrieve_top_k)
    print(f"  candidates after RRF: {len(candidates)}")

    # 3. Rerank
    results = rerank(eq, candidates, top_k=rerank_top_k)
    print(f"  top-{rerank_top_k} after rerank")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python pipeline.py ingest <file_or_dir>")
        print("  python pipeline.py query  <query_text>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "ingest":
        target = sys.argv[2]
        import os
        if os.path.isdir(target):
            ingest_directory(target)
        else:
            ingest_file(target)

    elif mode == "query":
        q = " ".join(sys.argv[2:])
        color_print(f"Running query: {q!r}")
        results = query(q)
        for r in results:
            print(r)