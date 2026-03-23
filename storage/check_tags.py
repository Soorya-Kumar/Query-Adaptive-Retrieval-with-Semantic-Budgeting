# scripts/list_docs_tags.py
from collections import defaultdict
from storage.store_postgres import get_conn
import argparse

SQL = """
SELECT c.doc_id, dt.level, dt.tag, AVG(dt.confidence) AS avg_conf, COUNT(*) AS cnt
FROM descriptor_tags dt
JOIN chunks c ON dt.chunk_id = c.chunk_id
GROUP BY c.doc_id, dt.level, dt.tag
ORDER BY c.doc_id, dt.level, avg_conf DESC
"""

def fetch_docs_tags(min_conf: float = 0.0, top_n_per_level: int | None = None):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(SQL)
        rows = cur.fetchall()

    docs = defaultdict(lambda: defaultdict(list))
    for doc_id, level, tag, avg_conf, cnt in rows:
        if avg_conf < min_conf:
            continue
        docs[doc_id][level].append((tag, float(avg_conf), int(cnt)))

    # optionally trim to top N per level
    if top_n_per_level:
        for doc_id in docs:
            for level in docs[doc_id]:
                docs[doc_id][level] = docs[doc_id][level][:top_n_per_level]
    return docs

def print_docs(docs):
    for doc_id in sorted(docs):
        print(f"Document: {doc_id}")
        for level in ("fine", "mid", "coarse"):
            tags = docs[doc_id].get(level, [])
            if not tags:
                continue
            print(f"  {level}:")
            for tag, avg_conf, cnt in tags:
                print(f"    - {tag} (avg_conf={avg_conf:.2f}, occurrences={cnt})")
        print()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="List ingested documents and their descriptor tags (no embeddings).")
    p.add_argument("--min-conf", type=float, default=0.0, help="Minimum average confidence to include")
    p.add_argument("--top", type=int, default=10, help="Top N tags per level")
    args = p.parse_args()

    docs = fetch_docs_tags(min_conf=args.min_conf, top_n_per_level=args.top)
    print_docs(docs)