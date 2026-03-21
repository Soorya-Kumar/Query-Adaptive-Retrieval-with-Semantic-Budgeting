from storage.store_vector import client, COLLECTION
print(client.get_collection(COLLECTION))

from storage.store_postgres import get_conn
with get_conn() as conn, conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM chunks")
    print("chunks:", cur.fetchone())
    cur.execute("SELECT COUNT(*) FROM descriptor_tags")
    print("tags:", cur.fetchone())
    cur.execute("SELECT COUNT(*) FROM descriptor_embeddings")
    print("embeddings:", cur.fetchone())