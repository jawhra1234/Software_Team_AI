# Bootstrap local prerequisites (Task 0.9) — Windows / PowerShell.
#
#   .\scripts\bootstrap.ps1
#
# Pulls the required Ollama models (ADR-0004) and checks that Ollama and the
# docker-compose stack are reachable.

$ErrorActionPreference = "Stop"

$ChatModel  = "qwen2.5-coder:7b-instruct"
$EmbedModel = "nomic-embed-text"
$OllamaUrl  = if ($env:OLLAMA__BASE_URL) { $env:OLLAMA__BASE_URL } else { "http://localhost:11434" }
$LangfuseUrl = "http://localhost:3000"

function Test-Http($url) {
    try { Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing | Out-Null; return $true }
    catch { return $false }
}

Write-Host "==> Checking Ollama at $OllamaUrl"
if (-not (Test-Http $OllamaUrl)) {
    Write-Warning "Ollama is not reachable. Start it (native install or 'docker compose --profile with-ollama up -d')."
} else {
    Write-Host "    Ollama is up."
}

Write-Host "==> Pulling models"
ollama pull $ChatModel
ollama pull $EmbedModel

Write-Host "==> Checking Postgres (pg_isready via docker compose)"
try {
    docker compose -f infra/docker-compose.yml exec -T postgres pg_isready | Out-Null
    Write-Host "    Postgres is ready."
} catch {
    Write-Warning "Postgres not ready. Run: docker compose -f infra/docker-compose.yml up -d"
}

Write-Host "==> Checking Langfuse at $LangfuseUrl"
if (Test-Http "$LangfuseUrl/api/public/health") {
    Write-Host "    Langfuse is up."
} else {
    Write-Warning "Langfuse not reachable (optional). Run: docker compose -f infra/docker-compose.yml up -d"
}

Write-Host "==> Bootstrap complete."
