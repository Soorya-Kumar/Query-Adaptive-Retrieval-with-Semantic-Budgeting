from storage.store_postgres import bm25_search
print(bm25_search("documents about technology", top_k=10))