-- Create a dedicated database for Langfuse in the same Postgres instance.
-- `\gexec` runs the generated CREATE DATABASE only when it does not already exist.
SELECT 'CREATE DATABASE langfuse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
