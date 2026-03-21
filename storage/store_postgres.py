import psycopg2
import psycopg2.extras
from typing import List, Optional
import numpy as np
from descriptors.schema import ChunkDescriptor
import json

# update credentials
DSN = "postgresql://user:password@localhost:5432/fyp1"

def get_conn():
    return psycopg2.connect(DSN)


# ── Write ──────────────────────────────────────────────────────────────────────

def insert_chunk_metadata(
    chunk_id: str,
    doc_id: str,
    raw_text: str,
    anchor_year: Optional[int],
    relativity_class: str,
):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (chunk_id, doc_id, raw_text, anchor_year, relativity_class)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            (chunk_id, doc_id, raw_text, anchor_year, relativity_class),
        )


def insert_descriptors(
    chunk_id: str,
    descriptor: ChunkDescriptor,
    embeddings: dict[str, List[tuple[str, float, List[float]]]]
):
    """
    embeddings = {
        "fine":   [(tag, confidence, vector), ...],
        "mid":    [...],
        "coarse": [...],
    }
    """
    with get_conn() as conn, conn.cursor() as cur:
        # Insert tags
        for level in ("fine", "mid", "coarse"):
            tags = getattr(descriptor, level)
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO descriptor_tags (chunk_id, level, tag, confidence)
                VALUES %s ON CONFLICT DO NOTHING
                """,
                [(chunk_id, level, t.tag, t.confidence) for t in tags],
            )

        # Insert embeddings
        for level, items in embeddings.items():
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO descriptor_embeddings (chunk_id, level, tag, confidence, embedding)
                VALUES %s
                """,
                [(chunk_id, level, tag, conf, vec) for tag, conf, vec in items],
            )


# ── Read ───────────────────────────────────────────────────────────────────────

def fetch_descriptor_embeddings(
    chunk_id: str,
) -> dict[str, List[tuple[str, float, np.ndarray]]]:
    """Returns {level: [(tag, confidence, vector), ...]} for reranking."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT level, tag, confidence, embedding
            FROM descriptor_embeddings
            WHERE chunk_id = %s
            """,
            (chunk_id,),
        )
        rows = cur.fetchall()
    
    result: dict[str, list] = {"fine": [], "mid": [], "coarse": []}

    cnt = 0
    for level, tag, conf, emb in rows:
        # parse if it comes back as a string
        if isinstance(emb, str):
            emb = json.loads(emb.replace("'", '"'))
        result[level].append((tag, conf, np.array(emb, dtype=float)))
    return result


def bm25_search(query: str, top_k: int = 500) -> List[dict]:
    """Full-text search over descriptor tags using tsvector."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, SUM(confidence * ts_rank(tag_tsv, query)) AS score
            FROM descriptor_tags, to_tsquery('english', %s) query
            WHERE tag_tsv @@ query
            GROUP BY chunk_id
            ORDER BY score DESC
            LIMIT %s
            """,
            (_to_tsquery(query), top_k),
        )
        rows = cur.fetchall()

    return [{"chunk_id": r[0], "score": float(r[1])} for r in rows]


def fetch_chunk_metadata(chunk_id: str) -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, anchor_year, relativity_class FROM chunks WHERE chunk_id = %s",
            (chunk_id,),
        )
        row = cur.fetchone()
    if not row:
        return {}
    return {"doc_id": row[0], "anchor_year": row[1], "relativity_class": row[2]}


def _to_tsquery(text: str) -> str:
    """Convert raw query string to tsquery format (AND of terms)."""
    terms = [t.strip() for t in text.split() if t.strip()]
    return " & ".join(terms)


def fetch_tags_for_document(doc_id: str):
    """
    Fetches all tags for a specific document ID from the descriptor_tags table.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT dt.level, dt.tag, dt.confidence
            FROM descriptor_tags dt
            JOIN chunks c ON dt.chunk_id = c.chunk_id
            WHERE c.doc_id = %s
            ORDER BY dt.level, dt.confidence DESC
            """,
            (doc_id,)
        )
        rows = cur.fetchall()

    if not rows:
        print(f"No tags found for document ID: {doc_id}")
        return

    print(f"Tags for document ID: {doc_id}")
    for level, tag, confidence in rows:
        print(f"  Level: {level}, Tag: {tag}, Confidence: {confidence:.2f}")