-- Enable pgvector on the application database (ADR-0010).
-- Runs once, against POSTGRES_DB, on first container initialization.
CREATE EXTENSION IF NOT EXISTS vector;
