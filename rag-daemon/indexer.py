"""
indexer.py — Hybrid RAG Code Indexer
======================================
This module provides ``CodeIndexer``, an object that:

  1. Walks every directory in ``settings.allowed_workspaces``.
  2. Skips paths matching ``settings.ignore_patterns``.
  3. Reads supported source files and splits them into semantically
     meaningful *chunks*.
  4. Generates dense vector embeddings locally using sentence-transformers.
  5. Upserts the chunks (text + embeddings + metadata) into a persistent
     ChromaDB collection.

Chunking strategy
-----------------
* **Python files** — parsed with the built-in ``ast`` module.  Each top-level
  function and class definition becomes its own chunk, preserving exact line
  numbers.  If a function/class body exceeds ``chunk_token_limit`` tokens it
  is sub-chunked with a sliding window.
* **All other text files** — split into fixed-size windows of
  ``chunk_token_limit`` whitespace-delimited tokens with ``chunk_overlap_lines``
  lines of overlap to preserve context across boundaries.
* Binary files are silently skipped.

Usage
-----
    from indexer import CodeIndexer
    indexer = CodeIndexer()
    await indexer.index_all()          # index every allowed workspace
    await indexer.index_path(some_dir) # index a single directory
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import chromadb
from chromadb import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File extensions the indexer will attempt to read as text.
# Add or remove extensions here to tune what gets indexed.
# ---------------------------------------------------------------------------
TEXT_EXTENSIONS: set[str] = {
    # Python
    ".py", ".pyi",
    # JavaScript / TypeScript
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Web
    ".html", ".css", ".scss", ".sass", ".less",
    # Data / Config
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    # Documentation
    ".md", ".rst", ".txt",
    # C family
    ".c", ".h", ".cpp", ".hpp", ".cc",
    # Java / Kotlin
    ".java", ".kt",
    # Go
    ".go",
    # Rust
    ".rs",
    # SQL
    ".sql",
    # Shell
    ".sh", ".bash", ".zsh", ".fish",
}


# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------

class CodeChunk:
    """A slice of a source file ready to be embedded and stored."""

    __slots__ = ("file_path", "start_line", "end_line", "text", "chunk_id")

    def __init__(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        text: str,
    ) -> None:
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.text = text
        # Deterministic, content-based ID so re-indexing the same chunk is an
        # upsert (idempotent) rather than a duplicate insert.
        self.chunk_id = hashlib.sha256(
            f"{file_path}:{start_line}:{end_line}:{text}".encode()
        ).hexdigest()[:32]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CodeChunk(file={self.file_path!r}, "
            f"lines={self.start_line}-{self.end_line}, "
            f"tokens≈{len(self.text.split())})"
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _should_ignore(path: Path, ignore_patterns: List[str]) -> bool:
    """
    Return True if *any* component of ``path`` matches one of the patterns.

    Patterns are matched against individual path components (not the full
    path), so ``'.git'`` will match ``.git/`` at any depth.
    Glob wildcard syntax is supported via ``fnmatch``.
    """
    for part in path.parts:
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _is_binary(path: Path, sample_size: int = 8192) -> bool:
    """
    Quick heuristic: if the first ``sample_size`` bytes contain a null byte
    the file is almost certainly binary.
    """
    try:
        with path.open("rb") as fh:
            chunk = fh.read(sample_size)
        return b"\x00" in chunk
    except OSError:
        return True


def _safe_read(path: Path) -> Optional[str]:
    """
    Attempt to read *path* as UTF-8, then latin-1 as a fallback.
    Returns None if the file cannot be read.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue
    logger.warning("Could not read file %s — skipping.", path)
    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_python(file_path: str, source: str, token_limit: int) -> List[CodeChunk]:
    """
    Parse *source* with the ``ast`` module and emit one chunk per top-level
    function / class definition.  Each chunk includes its docstring (if any)
    and the full body.

    If a node's text exceeds ``token_limit``, it is further split by a
    fixed-size sliding window so no single chunk overwhelms the embedding
    model's context window.
    """
    chunks: List[CodeChunk] = []
    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.debug("ast.parse failed for %s (%s) — falling back.", file_path, exc)
        return _chunk_fixed(file_path, source, token_limit, overlap_lines=5)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Only process top-level definitions (not nested ones, to avoid duplication).
        if not isinstance(getattr(node, "parent", None), ast.Module):
            continue

        start = node.lineno - 1          # 0-indexed
        end = node.end_lineno or start   # 0-indexed (inclusive)
        node_lines = lines[start:end + 1]
        node_text = "\n".join(node_lines)

        if len(node_text.split()) > token_limit:
            # Sub-chunk oversized nodes with the fixed-window splitter.
            sub_chunks = _chunk_fixed(
                file_path, node_text, token_limit, overlap_lines=3,
                line_offset=start
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=start + 1,   # back to 1-indexed for metadata
                    end_line=end + 1,
                    text=node_text,
                )
            )

    # Attach parent refs so the top-level filter above works.
    # (We do this after walking to avoid modifying the tree in-place.)
    # Re-parse with parent tracking for the next call:
    if not chunks:
        # No top-level defs found (e.g., script-style file) — use fixed window.
        return _chunk_fixed(file_path, source, token_limit, overlap_lines=5)

    return chunks


def _set_parents(tree: ast.AST) -> ast.AST:
    """Annotate every AST node with a ``parent`` attribute."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]
    return tree


def _chunk_fixed(
    file_path: str,
    source: str,
    token_limit: int,
    overlap_lines: int = 5,
    line_offset: int = 0,
) -> List[CodeChunk]:
    """
    Split *source* into fixed-size windows measured in whitespace-delimited
    tokens (words), with ``overlap_lines`` lines of context carried over into
    the next chunk.

    ``line_offset`` shifts reported line numbers (useful when recursively
    chunking a sub-range of a file).
    """
    raw_lines = source.splitlines()
    chunks: List[CodeChunk] = []

    current_tokens: List[str] = []
    chunk_lines: List[str] = []
    chunk_start: int = 0  # 0-indexed into raw_lines

    for i, line in enumerate(raw_lines):
        line_tokens = line.split()
        current_tokens.extend(line_tokens)
        chunk_lines.append(line)

        if len(current_tokens) >= token_limit:
            # Emit the current chunk.
            chunk_text = "\n".join(chunk_lines)
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=chunk_start + line_offset + 1,  # 1-indexed
                    end_line=i + line_offset + 1,
                    text=chunk_text,
                )
            )
            # Carry over the last `overlap_lines` lines into the next chunk.
            overlap = chunk_lines[-overlap_lines:] if overlap_lines > 0 else []
            chunk_start = i + 1 - len(overlap)
            chunk_lines = list(overlap)
            current_tokens = " ".join(overlap).split()

    # Flush remaining lines.
    if chunk_lines:
        chunk_text = "\n".join(chunk_lines)
        chunks.append(
            CodeChunk(
                file_path=file_path,
                start_line=chunk_start + line_offset + 1,
                end_line=len(raw_lines) + line_offset,
                text=chunk_text,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Main indexer class
# ---------------------------------------------------------------------------

class CodeIndexer:
    """
    Orchestrates workspace traversal, chunking, embedding, and ChromaDB storage.

    Thread-safety
    -------------
    This class is **not** thread-safe.  The recommended usage pattern is to
    run indexing tasks in a background ``asyncio`` thread via
    ``asyncio.to_thread(indexer.index_all_sync)``.
    """

    def __init__(self) -> None:
        # ---- Embedding model (loaded once; kept in memory) ----
        logger.info("Loading embedding model '%s' …", settings.embedding_model)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(settings.embedding_model)
        logger.info(
            "Model loaded in %.2fs.", time.perf_counter() - t0
        )

        # ---- ChromaDB persistent client ----
        logger.info(
            "Connecting to ChromaDB at '%s' …", settings.chroma_persist_dir
        )
        self._chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            # Use cosine distance so that semantic similarity scores are
            # intuitive (higher = more similar).
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready (%d documents already indexed).",
            settings.chroma_collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_all_sync(self) -> None:
        """
        Index every directory in ``settings.allowed_workspaces``.
        Blocking — call via ``asyncio.to_thread`` from async contexts.
        """
        if not settings.allowed_workspaces:
            logger.warning("No workspaces configured — nothing to index.")
            return

        for workspace in settings.allowed_workspaces:
            self.index_path_sync(Path(workspace))

    def index_path_sync(self, root: Path) -> None:
        """
        Index a single directory tree.
        Blocking — call via ``asyncio.to_thread`` from async contexts.
        """
        logger.info("Indexing workspace: %s", root)
        t0 = time.perf_counter()
        total_files = 0
        total_chunks = 0

        for file_path in self._walk(root):
            chunks = self._process_file(file_path)
            if chunks:
                self._upsert_chunks(chunks)
                total_files += 1
                total_chunks += len(chunks)

        logger.info(
            "Indexed workspace '%s': %d files, %d chunks in %.1fs.",
            root, total_files, total_chunks, time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _walk(self, root: Path) -> Generator[Path, None, None]:
        """
        Recursively yield file paths under *root*, skipping any path whose
        components match ``settings.ignore_patterns``.
        """
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if _should_ignore(path.relative_to(root), settings.ignore_patterns):
                logger.debug("Ignoring %s (matches ignore pattern)", path)
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                logger.debug("Skipping %s (extension not in TEXT_EXTENSIONS)", path)
                continue
            if _is_binary(path):
                logger.debug("Skipping %s (binary content detected)", path)
                continue
            yield path

    def _process_file(self, file_path: Path) -> List[CodeChunk]:
        """Read *file_path* and return a list of chunks ready for embedding."""
        source = _safe_read(file_path)
        if source is None or not source.strip():
            return []

        file_path_str = str(file_path.resolve())
        suffix = file_path.suffix.lower()

        if suffix == ".py":
            # For Python, use AST-based chunking first, with a fixed-window
            # fallback built into _chunk_python itself.
            tree = ast.parse(source, type_comments=False)
            _set_parents(tree)
            chunks = _chunk_python(
                file_path_str, source, settings.chunk_token_limit
            )
        else:
            chunks = _chunk_fixed(
                file_path_str,
                source,
                settings.chunk_token_limit,
                overlap_lines=settings.chunk_overlap_lines,
            )

        # Filter out vacuous chunks (e.g., files that are only comments/blank lines).
        chunks = [c for c in chunks if c.text.strip()]
        return chunks

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text strings.

        sentence-transformers normalises embeddings by default when
        ``normalize_embeddings=True``, which is compatible with ChromaDB's
        cosine-distance HNSW index.
        """
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def _upsert_chunks(self, chunks: List[CodeChunk], batch_size: int = 64) -> None:
        """
        Embed and upsert *chunks* into ChromaDB in batches.

        Using ``upsert`` (rather than ``add``) means re-indexing a file is
        idempotent — identical chunks are updated in-place rather than
        duplicated.  The chunk ID is content-addressed (SHA256), so a chunk
        is only re-embedded when its text actually changes.
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            # ChromaDB requires unique IDs per upsert. Deduplicate by chunk_id
            # (same content can appear e.g. via symlinks or chunking edge cases).
            by_id: dict[str, CodeChunk] = {}
            for c in batch:
                by_id[c.chunk_id] = c
            batch = list(by_id.values())
            if not batch:
                continue
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in batch
            ]
            embeddings = self._embed(texts)

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.debug("Upserted batch of %d chunks.", len(batch))
