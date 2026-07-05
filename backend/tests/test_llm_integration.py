"""Golden LLM test (Tasks 0.4/0.6) — integration; skipped without a live Ollama.

Exercises the real Ollama adapter end to end through structured_call. Marked
`integration` and skipped automatically when no Ollama server is reachable, so
the default `pytest` run stays hermetic.
"""

from __future__ import annotations

import pytest
from app.core.config import Settings
from app.providers.base import ChatMessage
from app.providers.factory import get_provider
from pydantic import BaseModel

from tests.conftest import ollama_available

pytestmark = pytest.mark.integration


class Answer(BaseModel):
    language: str
    is_compiled: bool


@pytest.mark.skipif(not ollama_available(), reason="local Ollama not reachable")
def test_structured_call_against_live_ollama() -> None:
    settings = Settings(_env_file=None)
    provider = get_provider("coder", settings)
    messages = [
        ChatMessage(role="system", content="You answer with structured data only."),
        ChatMessage(role="user", content="Describe the Python programming language."),
    ]
    answer = provider.structured(messages, Answer)
    assert answer.language.strip() != ""


@pytest.mark.skipif(not ollama_available(), reason="local Ollama not reachable")
def test_embed_against_live_ollama() -> None:
    settings = Settings(_env_file=None)
    embedder = get_provider("embed", settings)
    vectors = embedder.embed(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0
