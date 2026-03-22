# save as debug_pg.py and run: python debug_pg.py
import psycopg2
DSN = "postgresql://user:password@localhost:5432/fyp1"
conn = psycopg2.connect(DSN)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM chunks")
print('chunks count:', cur.fetchone()[0])
cur.execute("SELECT chunk_id, substring(raw_text,1,200) FROM chunks LIMIT 5")
print(cur.fetchall())
cur.close(); conn.close()