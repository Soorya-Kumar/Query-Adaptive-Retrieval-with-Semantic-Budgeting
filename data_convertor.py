# cisi_convert.py
import os
import sys
import re

def parse_cisi_all(path):
    text = open(path, "r", encoding="latin-1", errors="ignore").read()
    # Split on records starting with ".I <id>"
    parts = re.split(r"\n\.I\s+", "\n" + text)
    docs = {}
    for p in parts:
        p = p.strip()
        if not p:
            continue
        lines = p.splitlines()
        id_line = lines[0].strip().split()[0]
        docid = id_line
        # find .W section (document text)
        m = re.search(r"\.W\s*\n(.*)", p, flags=re.S)
        if m:
            body = m.group(1).strip()
        else:
            body = p
        docs[docid] = body
    return docs

def parse_cisi_qry(path):
    text = open(path, "r", encoding="latin-1", errors="ignore").read()
    parts = re.split(r"\n\.I\s+", "\n" + text)
    queries = {}
    for p in parts:
        p = p.strip()
        if not p:
            continue
        lines = p.splitlines()
        qid = lines[0].strip().split()[0]
        m = re.search(r"\.W\s*\n(.*)", p, flags=re.S)
        qtext = m.group(1).strip() if m else ""
        queries[qid] = qtext
    return queries

def parse_cisi_rel(path):
    qrels = {}
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                qid, did = parts[0], parts[1]
                qrels.setdefault(qid, set()).add(did)
    return qrels

def write_txts(docs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for did, text in docs.items():
        fname = os.path.join(out_dir, f"{did}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(text)

def write_queries(queries, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, txt in queries.items():
            f.write("{}\t{}\n".format(qid, txt.replace("\n", " ")))

def write_qrels(qrels, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for did in docs:
                f.write(f"{qid}\t{did}\t1\n")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python cisi_convert.py CISI.ALL CISI.QRY CISI.REL out_dir")
        sys.exit(1)
    all_path, qry_path, rel_path, out_dir = sys.argv[1:5]
    docs = parse_cisi_all(all_path)
    queries = parse_cisi_qry(qry_path)
    qrels = parse_cisi_rel(rel_path)
    write_txts(docs, os.path.join(out_dir, "docs_txt"))
    write_queries(queries, os.path.join(out_dir, "queries.tsv"))
    write_qrels(qrels, os.path.join(out_dir, "qrels.tsv"))
    print("Wrote docs, queries.tsv, qrels.tsv to", out_dir)