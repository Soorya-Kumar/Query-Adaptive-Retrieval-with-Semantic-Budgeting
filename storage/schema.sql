-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Chunks metadata table
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    raw_text        TEXT NOT NULL,
    anchor_year     INT,
    relativity_class TEXT CHECK (relativity_class IN ('recent', 'historical', 'timeless')),
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id        ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_anchor_year   ON chunks (anchor_year);
CREATE INDEX IF NOT EXISTS idx_chunks_relativity    ON chunks (relativity_class);


-- Descriptor tags table (sparse/BM25 via tsvector)
CREATE TABLE IF NOT EXISTS descriptor_tags (
    id          SERIAL PRIMARY KEY,
    chunk_id    TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    level       TEXT NOT NULL CHECK (level IN ('fine', 'mid', 'coarse')),
    tag         TEXT NOT NULL,
    confidence  FLOAT NOT NULL,
    tag_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', tag)) STORED
);

CREATE INDEX IF NOT EXISTS idx_tags_chunk_id  ON descriptor_tags (chunk_id);
CREATE INDEX IF NOT EXISTS idx_tags_level     ON descriptor_tags (level);
CREATE INDEX IF NOT EXISTS idx_tags_tsv       ON descriptor_tags USING GIN (tag_tsv);
CREATE INDEX IF NOT EXISTS idx_tags_trgm      ON descriptor_tags USING GIN (tag gin_trgm_ops);


-- Descriptor embeddings table (for rerank cosine lookup)
-- dim=768 for nomic-embed-text, change if using a different model
CREATE TABLE IF NOT EXISTS descriptor_embeddings (
    id          SERIAL PRIMARY KEY,
    chunk_id    TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    level       TEXT NOT NULL CHECK (level IN ('fine', 'mid', 'coarse')),
    tag         TEXT NOT NULL,
    confidence  FLOAT NOT NULL,
    embedding   vector(768)
);

CREATE INDEX IF NOT EXISTS idx_emb_chunk_id   ON descriptor_embeddings (chunk_id);
CREATE INDEX IF NOT EXISTS idx_emb_level      ON descriptor_embeddings (chunk_id, level);
CREATE INDEX IF NOT EXISTS idx_emb_fine_ivf   ON descriptor_embeddings USING ivfflat (embedding vector_cosine_ops)
    WHERE level = 'fine';
CREATE INDEX IF NOT EXISTS idx_emb_mid_ivf    ON descriptor_embeddings USING ivfflat (embedding vector_cosine_ops)
    WHERE level = 'mid';
CREATE INDEX IF NOT EXISTS idx_emb_coarse_ivf ON descriptor_embeddings USING ivfflat (embedding vector_cosine_ops)
    WHERE level = 'coarse';