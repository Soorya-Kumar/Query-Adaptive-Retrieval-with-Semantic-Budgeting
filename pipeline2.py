import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from utils import color_print

from ingestion.loader import load_txt, load_directory, Document
from ingestion.chunker import chunk_document, Chunk
from descriptors.extractor import extract_descriptors as extract_raw_descriptors
from descriptors.postprocessing import postprocess
from descriptors.schema import ChunkDescriptor
from descriptors.testing import run as extract_descriptors  # keep ingestion descriptor extractor
from embedding.encoding import embed
from embedding.pooling import pool, pool_query
from storage.store_vector import init_collection, upsert_chunk
from storage.store_postgres import (
    insert_chunk_metadata,
    insert_descriptors,
    fetch_descriptor_embeddings,
    fetch_chunk_metadata,
)
from query.analyser import analyze
from query.weights import QueryWeights, resolve as resolve_weights

from retrieval.dense_retriever import dense_retrieve
from retrieval.sparse_retriever import sparse_retrieve
from retrieval.fusion import rrf_fusion

from reranking.scorer import score_chunk


# EncodedQuery (inlined from query.encoder)
@dataclass
class EncodedQuery:
    raw: str
    analysis: object
    weights: QueryWeights
    descriptor: ChunkDescriptor
    level_embeddings: dict[str, list]
    pooled_vector: np.ndarray


def encode(query: str) -> EncodedQuery:
    analysis = analyze(query)
    weights = resolve_weights(analysis)

    raw_descriptor = extract_raw_descriptors(query)
    descriptor = postprocess(raw_descriptor)

    level_embeddings: dict[str, list] = {"fine": [], "mid": [], "coarse": []}
    for level in ("fine", "mid", "coarse"):
        tags = getattr(descriptor, level)
        for t in tags:
            vec = embed(t.tag)
            level_embeddings[level].append((t.tag, t.confidence, np.array(vec)))

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


# Reranker (inlined from reranking.desc_retriver)
def rerank(eq: EncodedQuery, candidates: List[dict], top_k: int = 20) -> List[dict]:
    scored = []

    for item in candidates:
        chunk_id = item["chunk_id"]

        doc_embeddings = fetch_descriptor_embeddings(chunk_id)
        metadata = fetch_chunk_metadata(chunk_id)
        doc_anchor_year = metadata.get("anchor_year") if metadata else None

        final_score = score_chunk(
            eq=eq,
            doc_level_embeddings=doc_embeddings,
            doc_anchor_year=doc_anchor_year,
        )

        scored.append({
            "chunk_id": chunk_id,
            "rrf_score": item.get("rrf_score", 0.0),
            "final_score": final_score,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k]


# Retrieval pipeline (dense+sparse, rerank dense-only, then fuse)
def retrieve(eq: EncodedQuery, top_k: int = 500, dense_rerank_k: int = 50) -> List[dict]:
    color_print("retrieving candidates...")

    dense = dense_retrieve(eq, top_k=top_k)
    color_print("dense retrieval complete.")

    sparse = sparse_retrieve(eq, top_k=top_k)
    color_print("sparse retrieval complete.")

    # Prepare dense candidates for reranking (preserve order)
    dense_for_rerank = [
        {"chunk_id": d["chunk_id"], "rrf_score": d.get("rrf_score", d.get("score", 0.0))}
        for d in dense
    ]

    # Rerank only dense candidates (smaller K)
    dense_reranked = rerank(eq, dense_for_rerank, top_k=dense_rerank_k)

    # rrf_fusion uses ordering to compute reciprocal-rank; pass dense_reranked (ordered) and sparse
    merged = rrf_fusion(dense_reranked, sparse, top_k=top_k)

    return merged


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
    dense_rerank_k: int = 50,
) -> List[dict]:
    eq = encode(text)

    print(f"Query: {text!r}")
    print(f"  depth={eq.analysis.semantic_depth}  intent={eq.analysis.intent}  temporal={eq.analysis.temporal_intent}")
    print(f"  weights: α={eq.weights.alpha:.2f} β={eq.weights.beta:.2f} γ={eq.weights.gamma:.2f} λ={eq.weights.lam:.2f}")

    # Retrieve: dense + sparse; rerank dense-only inside retrieve
    candidates = retrieve(eq, top_k=retrieve_top_k, dense_rerank_k=dense_rerank_k)
    print(f"  candidates after fusion: {len(candidates)}")

    # Already reranked for dense; just take top-N final results
    results = candidates[:rerank_top_k]
    print(f"  top-{rerank_top_k} returned")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python pipeline.py ingest <file_or_dir>")
        print("  python pipeline.py query  <query_text>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "ingest":
        target = sys.argv[2]
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