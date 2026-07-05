#!/usr/bin/env bash
# Bootstrap local prerequisites (Task 0.9) — POSIX shell.
#
#   ./scripts/bootstrap.sh
#
# Pulls the required Ollama models (ADR-0004) and checks that Ollama and the
# docker-compose stack are reachable.
set -euo pipefail

CHAT_MODEL="qwen2.5-coder:7b-instruct"
EMBED_MODEL="nomic-embed-text"
OLLAMA_URL="${OLLAMA__BASE_URL:-http://localhost:11434}"
LANGFUSE_URL="http://localhost:3000"

http_ok() { curl -fsS --max-time 3 "$1" >/dev/null 2>&1; }

echo "==> Checking Ollama at ${OLLAMA_URL}"
if http_ok "${OLLAMA_URL}"; then
  echo "    Ollama is up."
else
  echo "    WARNING: Ollama not reachable. Start it (native install or 'docker compose --profile with-ollama up -d')." >&2
fi

echo "==> Pulling models"
ollama pull "${CHAT_MODEL}"
ollama pull "${EMBED_MODEL}"

echo "==> Checking Postgres (pg_isready via docker compose)"
if docker compose -f infra/docker-compose.yml exec -T postgres pg_isready >/dev/null 2>&1; then
  echo "    Postgres is ready."
else
  echo "    WARNING: Postgres not ready. Run: docker compose -f infra/docker-compose.yml up -d" >&2
fi

echo "==> Checking Langfuse at ${LANGFUSE_URL}"
if http_ok "${LANGFUSE_URL}/api/public/health"; then
  echo "    Langfuse is up."
else
  echo "    WARNING: Langfuse not reachable (optional). Run: docker compose -f infra/docker-compose.yml up -d" >&2
fi

echo "==> Bootstrap complete."
