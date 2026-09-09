CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faqs (
    id SERIAL PRIMARY KEY,
    faq_id TEXT UNIQUE NOT NULL,
    categoria TEXT NOT NULL,
    pregunta TEXT NOT NULL,
    respuesta TEXT NOT NULL,
    metadata JSONB,
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS faqs_embedding_idx
    ON faqs USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
