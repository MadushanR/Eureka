#testing
"""
config.py — Daemon Configuration
=================================
Loads configuration from a `.env` file or environment variables using
pydantic-settings.  Exposes two critical security-relevant settings:

  ALLOWED_WORKSPACES  – Absolute paths the daemon may read/write.
  IGNORE_PATTERNS     – Directory / file name patterns to skip during indexing.

All workspace paths are resolved to their canonical absolute form on load,
so that downstream path-containment checks are reliable.

Usage
-----
    from config import settings
    print(settings.allowed_workspaces)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class DaemonSettings(BaseSettings):
    """
    All runtime configuration for the Hybrid RAG edge daemon.

    Priority order (highest → lowest):
      1. Actual OS environment variables
      2. Variables defined in `.env` (next to this file)
      3. Defaults declared below

    JSON-valued environment variables are supported natively by pydantic-settings.
    For example:
        ALLOWED_WORKSPACES='["/home/user/project", "/home/user/libs"]'
    """
    model_config = SettingsConfigDict(
        # Load a .env file from the project root (same directory as this file).
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore extra keys in the environment so the daemon doesn't crash
        # if unrelated env vars are set.
        extra="ignore",
        # Case-insensitive key matching (useful on Windows).
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Core paths
    # ------------------------------------------------------------------

    allowed_workspaces: List[str] = []
    """
    List of absolute directory paths that the daemon is permitted to index
    and apply patches to.  Any operation targeting a path outside this list
    will be rejected with a 403 error.

    Example .env:
        ALLOWED_WORKSPACES='["/home/user/myproject", "/home/user/libs"]'
    """

    chroma_persist_dir: str = "./chroma_db"
    """
    Directory where ChromaDB persists its data between daemon restarts.
    Relative paths are resolved relative to the working directory.
    """

    chroma_collection_name: str = "code_index"
    """Name of the ChromaDB collection used to store code embeddings."""

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------

    embedding_model: str = "all-MiniLM-L6-v2"
    """
    Sentence-Transformers model used to generate embeddings.
    This model runs fully locally — no network calls at inference time.
    """

    # ------------------------------------------------------------------
    # Indexing behaviour
    # ------------------------------------------------------------------

    ignore_patterns: List[str] = [
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
        "*.egg-info",
    ]
    """
    Directory and file-name patterns to skip during workspace traversal.
    Supports exact names (e.g. '.git') and glob-style wildcards (e.g. '*.egg-info').
    Matched against each path *component*, not the full path, so '.git' will
    prune any `.git/` directory at any depth.
    """

    chunk_token_limit: int = 500
    """
    Maximum number of whitespace-delimited tokens (words) per chunk when the
    indexer falls back to fixed-size chunking (i.e. no AST-level splitting).
    """

    chunk_overlap_lines: int = 5
    """
    Number of lines of overlap to keep between consecutive fixed-size chunks,
    to preserve context at chunk boundaries.
    """

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    host: str = "127.0.0.1"
    """Bind address for the uvicorn server.  Defaults to loopback only."""

    port: int = 8765
    """Port the daemon listens on."""

    log_level: str = "info"
    """Uvicorn / Python logging level: debug | info | warning | error."""

    # ------------------------------------------------------------------
    # Spotify (optional — if set, daemon can control playback via Web API)
    # ------------------------------------------------------------------

    spotify_client_id: Optional[str] = None
    """Spotify app Client ID from the Spotify Developer Dashboard."""

    spotify_client_secret: Optional[str] = None
    """Spotify app Client Secret."""

    spotify_refresh_token: Optional[str] = None
    """Refresh token from OAuth (one-time auth flow). Used to get access tokens."""

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("allowed_workspaces", mode="before")
    @classmethod
    def _parse_workspaces(cls, v: object) -> List[str]:
        """
        Accept either a Python list (already parsed by pydantic-settings from
        a JSON string in the env) or a plain comma-separated string for
        convenience.
        """
        if isinstance(v, str):
            stripped = v.strip()
            # JSON array notation
            if stripped.startswith("["):
                return json.loads(stripped)
            # Comma-separated fallback
            return [p.strip() for p in stripped.split(",") if p.strip()]
        return v  # already a list

    @model_validator(mode="after")
    def _resolve_and_validate_workspaces(self) -> "DaemonSettings":
        """
        Resolve every workspace path to its canonical absolute form and
        verify that each directory actually exists.  This makes containment
        checks in the /apply-patch endpoint unambiguous.
        """
        resolved: List[str] = []
        for raw_path in self.allowed_workspaces:
            p = Path(raw_path).expanduser().resolve()
            if not p.is_dir():
                raise ValueError(
                    f"ALLOWED_WORKSPACES entry '{raw_path}' does not exist or "
                    "is not a directory.  Please check your .env file."
                )
            resolved.append(str(p))
        self.allowed_workspaces = resolved

        if not self.allowed_workspaces:
            logger.warning(
                "ALLOWED_WORKSPACES is empty.  The daemon will not index any code "
                "and will reject all /apply-patch requests.  Set this variable in "
                "your .env file."
            )

        # Resolve chroma_persist_dir to an absolute path as well.
        self.chroma_persist_dir = str(
            Path(self.chroma_persist_dir).expanduser().resolve()
        )

        return self


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere.
# ---------------------------------------------------------------------------
settings = DaemonSettings()

logger.info(
    "Daemon configuration loaded. "
    "allowed_workspaces=%s  chroma_dir=%s",
    settings.allowed_workspaces,
    settings.chroma_persist_dir,
)
