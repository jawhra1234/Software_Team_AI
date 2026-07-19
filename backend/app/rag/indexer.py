"""Repository indexing — full build + incremental reindex (Tasks 3.6/3.7, ADR-0008).

``index_project`` walks a workspace (``.gitignore``-aware via ``git ls-files``,
with an os.walk fallback for non-git dirs), chunks each source file, embeds the
chunks, and upserts them into the namespaced vector store — run once on
workspace attach, off the run's critical path.

``reindex_changed`` diffs each file's current chunk-hash set against what's
stored and re-embeds only files that actually changed (and drops deleted files),
so incremental reindex stays cheap on CPU (Phase-3 risk note). Binary/oversized
files and parse errors are skipped/degraded gracefully — indexing never raises
on a bad file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.rag.chunker import CodeChunk, chunk_source, language_for_path
from app.rag.embeddings import ChunkEmbedder
from app.rag.vector_store import VectorStore
from app.tools.git import Git

log = get_logger("rag.indexer")

_IGNORE_DIRS = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
)
_MAX_FILE_BYTES = 512_000


@dataclass
class IndexStats:
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_indexed: int = 0
    files_deleted: int = 0


class Indexer:
    """Builds and maintains a project's code index in the vector store."""

    def __init__(self, store: VectorStore, embedder: ChunkEmbedder) -> None:
        self._store = store
        self._embedder = embedder

    def index_project(self, project_id: str, root: Path) -> IndexStats:
        """Full (re)index: clear the namespace, then index every source file."""
        self._store.ensure_schema()
        self._store.clear_project(project_id)
        stats = IndexStats()
        all_chunks: list[CodeChunk] = []
        for rel_path in self._source_files(root):
            chunks = self._chunk_file(root, rel_path, stats)
            all_chunks.extend(chunks)
            if chunks:
                stats.files_indexed += 1
        if all_chunks:
            embedded = self._embedder.embed_chunks(all_chunks)
            stats.chunks_indexed = self._store.add_chunks(project_id, embedded)
        log.info("index_project", project_id=project_id, **vars(stats))
        return stats

    def reindex_changed(self, project_id: str, root: Path) -> IndexStats:
        """Re-embed only files whose chunk-hash set changed; drop deleted files."""
        stored = self._store.hashes_by_path(project_id)
        stats = IndexStats()
        current: set[str] = set()
        for rel_path in self._source_files(root):
            current.add(rel_path)
            chunks = self._chunk_file(root, rel_path, stats)
            current_hashes = {c.content_hash for c in chunks}
            if current_hashes == stored.get(rel_path, set()):
                continue  # unchanged — skip re-embed
            embedded = self._embedder.embed_chunks(chunks) if chunks else []
            self._store.replace_file(project_id, rel_path, embedded)
            stats.files_indexed += 1
            stats.chunks_indexed += len(embedded)
        deleted = set(stored) - current
        for rel_path in deleted:
            self._store.delete_paths(project_id, [rel_path])
        stats.files_deleted = len(deleted)
        log.info("reindex_changed", project_id=project_id, **vars(stats))
        return stats

    def _chunk_file(self, root: Path, rel_path: str, stats: IndexStats) -> list[CodeChunk]:
        abs_path = root / rel_path
        try:
            if not abs_path.is_file() or abs_path.stat().st_size > _MAX_FILE_BYTES:
                stats.files_skipped += 1
                return []
            text = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            stats.files_skipped += 1
            return []
        return chunk_source(rel_path.replace(os.sep, "/"), text)

    def _source_files(self, root: Path) -> list[str]:
        """List indexable source files, relative to root, respecting .gitignore."""
        candidates = Git(root).ls_files() if (root / ".git").exists() else _walk_files(root)
        return [p for p in candidates if language_for_path(p) is not None]


def _walk_files(root: Path) -> list[str]:
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            files.append(rel.replace(os.sep, "/"))
    return files
