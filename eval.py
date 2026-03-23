# eval_subset.py
from pathlib import Path
from collections import defaultdict
import pipeline
from storage.store_postgres import fetch_chunk_metadata

CHECKPOINT = Path("ingestion_checkpoint.txt")
QRELS = Path("data/cisi/qrels.tsv")
QUERIES = Path("data/cisi/queries.tsv")

TOP_K = 50 # evaluation cutoff for precision/recall
RERANK_TOP_K = 50  # to match pipeline.query default

def load_checkpoint():
    return {line.strip() for line in CHECKPOINT.read_text().splitlines() if line.strip()}

def load_qrels():
    qrels = defaultdict(set)
    for line in QRELS.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            qid, docid = parts[0], parts[1]
            qrels[qid].add(docid)
    return qrels

def load_queries():
    qs = {}
    for line in QUERIES.read_text().splitlines():
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            qid, text = parts
            qs[qid] = text
    return qs

def eval_on_subset():
    checkpoint_docs = load_checkpoint()
    qrels = load_qrels()
    queries = load_queries()

    # select queries that have at least one relevant doc ingested
    selected_qids = [qid for qid, docs in qrels.items() if docs & checkpoint_docs]
    print(f"Selected {len(selected_qids)} queries (have relevant docs in checkpoint)")

    sum_prec = 0.0
    sum_recall = 0.0
    sum_mrr = 0.0
    count = 0

    for qid in selected_qids:
        qtext = queries.get(qid)
        if not qtext:
            continue

        results = pipeline.query(qtext, retrieve_top_k=500, rerank_top_k=RERANK_TOP_K)
        retrieved_docs = []
        for r in results[:TOP_K]:
            cid = r.get("chunk_id")
            meta = fetch_chunk_metadata(cid)
            docid = meta.get("doc_id")
            if docid:
                retrieved_docs.append(str(docid))

        relevant_ingested = qrels[qid] & checkpoint_docs
        if not relevant_ingested:
            continue

        # precision@K
        num_rel_ret = sum(1 for d in retrieved_docs if d in relevant_ingested)
        prec = num_rel_ret / TOP_K
        # recall@K
        recall = num_rel_ret / len(relevant_ingested)
        # MRR
        mrr = 0.0
        for idx, d in enumerate(retrieved_docs, start=1):
            if d in relevant_ingested:
                mrr = 1.0 / idx
                break

        sum_prec += prec
        sum_recall += recall
        sum_mrr += mrr
        count += 1

        print(f"Q{qid}: prec@{TOP_K}={prec:.3f} recall@{TOP_K}={recall:.3f} mrr={mrr:.3f}")

    if count == 0:
        print("No selected queries evaluated.")
        return

    print("=== Aggregate ===")
    print(f"Queries evaluated: {count}")
    print(f"Mean Precision@{TOP_K}: {sum_prec/count:.4f}")
    print(f"Mean Recall@{TOP_K}: {sum_recall/count:.4f}")
    print(f"Mean MRR: {sum_mrr/count:.4f}")

if __name__ == "__main__":
    eval_on_subset()