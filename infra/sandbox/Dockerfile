# Sandbox image for executing agent-generated code (Task 1.2, ADR-0007).
# Run with `--network=none` and resource limits; pytest/ripgrep/git are baked in
# so verification works without network access inside the container.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pytest

# Non-root execution.
RUN useradd --create-home --uid 1000 sandbox
USER sandbox

WORKDIR /work
