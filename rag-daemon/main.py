"""
main.py — Hybrid RAG Edge Daemon
==================================
Entry-point for the FastAPI application.

Endpoints
---------
  GET  /health          Liveness probe — returns daemon status.
  POST /search          Semantic code search against the ChromaDB index.
  POST /apply-patch     Apply a git patch within a verified workspace path.
  POST /index-now       Trigger a full re-index of all allowed workspaces.

Security model
--------------
The /apply-patch endpoint is potentially dangerous because it shells out to
``git apply``.  The following layers of defence are applied:

  1. ``workspace_path`` is resolved to a canonical absolute path (resolves
     symlinks, removes ``..`` components).
  2. The resolved path is compared against every resolved path in
     ``settings.allowed_workspaces`` using a strict *prefix* check.
  3. Only if the containment check passes does the subprocess run.
  4. ``git apply`` runs with ``--check`` first (a dry-run) to confirm the
     patch is well-formed and applies cleanly before mutating the repository.
  5. stdin is used to deliver the patch string so that it is never written
     to a temp file on disk (reduces the attack surface for temp-file
     injection attacks).

Run the daemon
--------------
    uvicorn main:app --host 127.0.0.1 --port 8765
or, during development with auto-reload:
    uvicorn main:app --reload
"""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import logging
import re
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import chromadb
from chromadb import Settings as ChromaSettings
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

import requests

from config import settings
from indexer import CodeIndexer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hybrid RAG Edge Daemon",
    description=(
        "A local edge daemon that indexes code into ChromaDB and exposes "
        "semantic search + git-patch application endpoints."
    ),
    version="1.0.0",
    # Disable the Swagger UI's 'try-it-out' feature in production by pointing
    # openapi to None.  Remove this line during development for convenience.
    # openapi_url=None,
)

# ---------------------------------------------------------------------------
# Shared state (lazy-initialised on startup)
# ---------------------------------------------------------------------------

# These are set during the ``lifespan`` startup hook and used in route handlers.

_indexer: Optional[CodeIndexer] = None
_chroma_collection: Optional[chromadb.Collection] = None
_embed_model: Optional[SentenceTransformer] = None


# ---------------------------------------------------------------------------
# Application lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager  # noqa: E402  (after imports above)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start-up: load the embedding model, connect to ChromaDB, and kick off an
    initial background index of all allowed workspaces.

    Shut-down: nothing extraordinary required — ChromaDB flushes on its own.
    """
    global _indexer, _chroma_collection, _embed_model

    logger.info("=== Hybrid RAG Edge Daemon starting up ===")
    logger.info("Allowed workspaces: %s", settings.allowed_workspaces)

    # Initialise the indexer (loads the SentenceTransformer model and opens
    # the ChromaDB persistent client).
    _indexer = CodeIndexer()

    # Keep a direct reference to the collection and model for the search
    # endpoint so we avoid going through the indexer's internal interface.
    _chroma_collection = _indexer._collection  # noqa: SLF001
    _embed_model = _indexer._model              # noqa: SLF001

    # Kick off indexing in a background thread so the HTTP server is
    # immediately available (useful for health checks from orchestrators).
    asyncio.create_task(
        asyncio.to_thread(_indexer.index_all_sync),
        name="initial-index",
    )

    logger.info("=== Daemon ready — listening on %s:%s ===", settings.host, settings.port)
    yield  # <-- application runs here

    logger.info("=== Hybrid RAG Edge Daemon shutting down ===")


app.router.lifespan_context = lifespan


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Payload for the /search endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Natural-language or code query string.",
        examples=["function to validate JWT tokens"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of results to return.",
    )


class SearchResultItem(BaseModel):
    """A single result from the semantic code search."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    score: float = Field(description="Cosine similarity score (higher = more relevant).")
    text: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


class ApplyPatchRequest(BaseModel):
    """Payload for the /apply-patch endpoint."""

    patch_string: str = Field(
        ...,
        min_length=1,
        description="A valid unified diff / git patch to apply.",
    )
    workspace_path: str = Field(
        ...,
        description=(
            "Absolute path to the git repository root where the patch should "
            "be applied.  MUST be within ALLOWED_WORKSPACES."
        ),
        examples=["/home/user/myproject"],
    )

    @field_validator("workspace_path")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("workspace_path must not be blank.")
        return v


class ApplyPatchResponse(BaseModel):
    success: bool
    message: str
    stdout: str = ""
    stderr: str = ""


class ComputeRemoveLinePatchRequest(BaseModel):
    """Request body for /edit/compute-remove-line-patch."""

    workspace_path: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1, description="Path relative to workspace_path.")
    line_number: int = Field(..., ge=1, description="1-based line number to remove.")
    line_end: Optional[int] = Field(default=None, ge=1, description="Optional 1-based end line for range (inclusive).")

    @field_validator("workspace_path", "file_path")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Must not be blank.")
        return v


class ComputeRemoveLinePatchResponse(BaseModel):
    success: bool
    message: str = ""
    patch: str = ""


class ComputeRemoveLinesMatchingPatchRequest(BaseModel):
    """Request body for /edit/compute-remove-lines-matching-patch."""

    workspace_path: str = Field(..., min_length=1)
    pattern: str = Field(..., description="Literal substring to find in lines; matching lines are removed.")
    file_path: Optional[str] = Field(default=None, description="If set, only this file (relative to workspace); else all files.")
    path_glob: Optional[str] = Field(default="**/*", description="Glob for relative paths when scanning multiple files (e.g. **/*.py).")

    @field_validator("workspace_path")
    @classmethod
    def _wp_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("workspace_path must not be blank.")
        return v


class ComputeRemoveLinesMatchingPatchResponse(BaseModel):
    success: bool
    message: str = ""
    patch: str = ""
    files_affected: int = 0


class ComputeReplacePatchRequest(BaseModel):
    """Payload for /edit/compute-replace-patch."""

    workspace_path: str = Field(..., min_length=1, description="Absolute path to the git workspace root.")
    file_path: str = Field(..., min_length=1, description="Path to file, relative to workspace root.")
    search_string: str = Field(..., min_length=1, description="Exact substring to find in the file.")
    replace_string: str = Field(..., description="Replacement text (can be empty to delete the match).")

    @field_validator("workspace_path", "file_path")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Must not be blank.")
        return v


class ComputeReplacePatchResponse(BaseModel):
    success: bool
    message: str = ""
    patch: str = ""


class InsertCodeRequest(BaseModel):
    """Payload for /edit/insert-code."""

    workspace_path: str = Field(..., min_length=1, description="Absolute path to the git workspace root.")
    file_path: str = Field(..., min_length=1, description="Path to file, relative to workspace root.")
    after_line: int = Field(..., ge=0, description="1-based line number to insert after. 0 = prepend to start of file.")
    new_code: str = Field(..., min_length=1, description="The new code to insert.")

    @field_validator("workspace_path", "file_path")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Must not be blank.")
        return v


class InsertCodeResponse(BaseModel):
    success: bool
    message: str = ""
    patch: str = ""


class DeleteBlockRequest(BaseModel):
    """Payload for /edit/delete-block."""

    workspace_path: str = Field(..., min_length=1, description="Absolute path to the git workspace root.")
    file_path: str = Field(..., min_length=1, description="Path to file, relative to workspace root.")
    search_term: str = Field(
        ...,
        min_length=1,
        description=(
            "A string that uniquely identifies the function/class/endpoint to delete. "
            "For example: '/testing' or 'def ping' or 'async def health'. "
            "The daemon finds the top-level block (including decorators and docstring) that contains this string."
        ),
    )

    @field_validator("workspace_path", "file_path")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Must not be blank.")
        return v


class DeleteBlockResponse(BaseModel):
    success: bool
    message: str = ""
    patch: str = ""
    deleted_lines: str = ""


class IndexNowResponse(BaseModel):
    message: str


class WorkspaceFolders(BaseModel):
    """Top-level folders for a single allowed workspace."""

    workspace_path: str
    folders: List[str]


class ListFoldersResponse(BaseModel):
    """Response model for /list-folders endpoint."""

    workspaces: List[WorkspaceFolders]


class FolderEntry(BaseModel):
    """Single file or folder entry inside a directory."""

    name: str
    type: Literal["file", "folder"]


class ListFolderContentsResponse(BaseModel):
    """Response model for /list-folder-contents endpoint."""

    folder_path: str
    entries: List[FolderEntry]


class DeletePathRequest(BaseModel):
    """Payload for the /delete-path endpoint."""

    path: str = Field(
        ...,
        min_length=1,
        description="Absolute path to a file or folder to delete. Must be inside an allowed workspace (and not the workspace root).",
    )

    @field_validator("path")
    @classmethod
    def _path_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("path must not be blank.")
        return v


class DeletePathResponse(BaseModel):
    """Response model for /delete-path endpoint."""

    success: bool
    message: str


class ReadFileRequest(BaseModel):
    """Payload for the /read-file endpoint."""

    path: str = Field(
        ...,
        min_length=1,
        description="Absolute path to a file within an allowed workspace.",
    )
    start_line: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional 1-based start line. If set, only lines from start_line are returned.",
    )
    end_line: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional 1-based end line (inclusive). If set, only lines up to end_line are returned.",
    )


class ReadFileResponse(BaseModel):
    """Response model for /read-file endpoint."""

    success: bool
    path: str = ""
    content: str = ""
    total_lines: int = 0
    error: str = ""


# Allowed command lines for /run-command (normalized to single spaces, lowercased for comparison).
# Only these exact commands are permitted to avoid shell injection.
RUN_COMMAND_WHITELIST = frozenset({
    "npm test",
    "npm run test",
    "yarn test",
    "pnpm test",
    "pnpm run test",
    "pytest",
    "python -m pytest",
    "dotnet test",
})


class RunCommandRequest(BaseModel):
    """Payload for the /run-command endpoint."""

    workspace_path: str = Field(
        ...,
        min_length=1,
        description="Absolute path to the workspace (must be in ALLOWED_WORKSPACES).",
    )
    command_line: str = Field(
        ...,
        min_length=1,
        description="Command to run, e.g. 'npm test' or 'pytest'. Must match whitelist.",
    )


class RunCommandResponse(BaseModel):
    """Response model for /run-command endpoint."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    error: str = ""


class SpotifyGenericResponse(BaseModel):
    """Generic response for Spotify control endpoints."""

    success: bool
    message: str


class SpotifyVolumeRequest(BaseModel):
    """Request body for /spotify/volume."""

    volume_percent: int = Field(ge=0, le=100, description="Volume 0-100.")


class GitUncommittedDiffResponse(BaseModel):
    """Response for /git/uncommitted-diff."""

    success: bool
    message: str
    workspace_path: str = ""
    status_short: str = ""  # git status --short
    diff: str = ""  # git diff (unstaged) + git diff --staged
    has_changes: bool = False


class GitRepoInfo(BaseModel):
    """Information about a discovered git repository."""

    name: str
    path: str


class ListGitReposResponse(BaseModel):
    """Response for /git/find-repos."""

    repos: List[GitRepoInfo]


class CommitAndPushRequest(BaseModel):
    """Request body for /git/commit-and-push (legacy combined flow)."""

    workspace_path: str = Field(..., min_length=1, description="Git repo root within ALLOWED_WORKSPACES.")
    commit_message: str = Field(
        default="Update from Eureka",
        description="Commit message for the commit.",
    )
    completion_webhook_url: Optional[str] = Field(
        default=None,
        description="If set, commit runs synchronously but push runs in background; response is 202 and completion is POSTed here.",
    )

    @field_validator("workspace_path")
    @classmethod
    def _path_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("workspace_path must not be blank.")
        return v


class CommitAndPushResponse(BaseModel):
    """Response for /git/commit-and-push."""

    success: bool
    message: str
    stdout: str = ""
    stderr: str = ""
    started: bool = False  # True when push is running in background (202 response)


class CommitAllRequest(BaseModel):
    """Request body for /git/commit-all (commit only, no push)."""

    workspace_path: str = Field(..., min_length=1, description="Git repo root within ALLOWED_WORKSPACES.")
    commit_message: str = Field(
        default="Update from Eureka",
        description="Commit message for the commit.",
    )

    @field_validator("workspace_path")
    @classmethod
    def _commit_path_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("workspace_path must not be blank.")
        return v


class CommitAllResponse(BaseModel):
    """Response for /git/commit-all."""

    success: bool
    message: str
    stdout: str = ""
    stderr: str = ""


class PushRequest(BaseModel):
    """Request body for /git/push-only (push existing commits)."""

    workspace_path: str = Field(..., min_length=1, description="Git repo root within ALLOWED_WORKSPACES.")
    completion_webhook_url: Optional[str] = Field(
        default=None,
        description="If set, push runs in background; response is 202 and completion is POSTed here.",
    )

    @field_validator("workspace_path")
    @classmethod
    def _push_path_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("workspace_path must not be blank.")
        return v


class PushResponse(BaseModel):
    """Response for /git/push-only."""

    success: bool
    message: str
    stdout: str = ""
    stderr: str = ""
    started: bool = False  # True when push is running in background (202 response)


# ---------------------------------------------------------------------------
# Spotify Web API helper (sync — run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _spotify_access_token() -> Optional[str]:
    """Exchange refresh_token for access_token. Returns None if not configured or on error."""
    cid = getattr(settings, "spotify_client_id", None)
    secret = getattr(settings, "spotify_client_secret", None)
    ref = getattr(settings, "spotify_refresh_token", None)
    if not cid or not secret or not ref:
        return None
    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": ref,
                "client_id": settings.spotify_client_id,
                "client_secret": settings.spotify_client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        logger.warning("Spotify token refresh failed: %s", e)
        return None


def _spotify_api(
    method: str,
    path: str,
    *,
    access_token: str,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Call Spotify Web API. Returns (success, message)."""
    url = f"https://api.spotify.com/v1{path}"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        if method == "GET":
            r = requests.get(url, headers=headers, params=params, timeout=10)
        elif method == "PUT":
            r = requests.put(url, headers=headers, json=json_body, params=params, timeout=10)
        elif method == "POST":
            r = requests.post(url, headers=headers, json=json_body, timeout=10)
        else:
            return False, "Unsupported method"
        if r.status_code in (200, 204):
            return True, "OK"
        try:
            err = r.json().get("error", {})
            msg = err.get("message", r.text or str(r.status_code))
        except Exception:
            msg = r.text or str(r.status_code)
        return False, msg
    except Exception as e:
        return False, str(e)


def _spotify_devices_sync(access_token: str) -> tuple[bool, str, List[Dict[str, Any]]]:
    """Get user's available Spotify devices. Returns (success, message, devices list)."""
    url = "https://api.spotify.com/v1/me/player/devices"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
        if r.status_code != 200:
            try:
                err = r.json().get("error", {})
                msg = err.get("message", r.text or str(r.status_code))
            except Exception:
                msg = r.text or str(r.status_code)
            return False, msg, []
        data = r.json()
        devices = data.get("devices") or []
        return True, "OK", devices
    except Exception as e:
        return False, str(e), []


def _pick_best_device(devices: List[Dict[str, Any]]) -> Optional[str]:
    """Prefer PC/Windows/Desktop or type Computer; else first available device."""
    if not devices:
        return None
    # Prefer device whose name suggests a computer (user said "on my PC")
    for d in devices:
        name = (d.get("name") or "").lower()
        dtype = (d.get("type") or "").lower()
        if "computer" in dtype or "pc" in name or "windows" in name or "desktop" in name:
            return d.get("id")
    return devices[0].get("id")


def _spotify_launch_desktop_sync() -> bool:
    """Try to launch the Spotify desktop app on this machine. Returns True if we attempted launch."""
    system = platform.system()
    try:
        if system == "Windows":
            # Prefer installed app under AppData
            appdata = os.environ.get("APPDATA", "")
            spotify_exe = Path(appdata) / "Spotify" / "Spotify.exe"
            if spotify_exe.is_file():
                subprocess.Popen(
                    [str(spotify_exe)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return True
            # Fallback: use the spotify: URI (opens default handler, often the app)
            subprocess.run(
                ["cmd", "/c", "start", "spotify:"],
                timeout=5,
                capture_output=True,
                check=False,
            )
            return True
        if system == "Darwin":
            subprocess.Popen(
                ["open", "-a", "Spotify"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        if system == "Linux":
            subprocess.Popen(
                ["spotify"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
    except Exception as e:
        logger.warning("Could not launch Spotify desktop app: %s", e)
    return False


def _spotify_close_desktop_sync() -> tuple[bool, str]:
    """Close the Spotify desktop app on this machine. Returns (success, message)."""
    system = platform.system()
    try:
        if system == "Windows":
            result = subprocess.run(
                ["taskkill", "/IM", "Spotify.exe", "/F"],
                timeout=10,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return True, "Spotify closed."
            if "not found" in (result.stderr or "").lower() or result.returncode == 128:
                return True, "Spotify was not running."
            return False, result.stderr or result.stdout or "Failed to close Spotify."
        if system == "Darwin":
            result = subprocess.run(
                ["osascript", "-e", 'quit app "Spotify"'],
                timeout=5,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return True, "Spotify closed."
            return True, "Spotify was not running or already closed."
        if system == "Linux":
            result = subprocess.run(
                ["pkill", "-x", "spotify"],
                timeout=5,
                capture_output=True,
                check=False,
            )
            if result.returncode in (0, 1):
                return True, "Spotify closed." if result.returncode == 0 else "Spotify was not running."
            return False, "Failed to close Spotify."
    except Exception as e:
        logger.warning("Could not close Spotify: %s", e)
        return False, str(e)
    return False, "Unsupported OS"


def _spotify_play_sync() -> tuple[bool, str]:
    token = _spotify_access_token()
    if not token:
        return False, "Spotify is not configured (missing client id, secret, or refresh token)."
    ok, msg = _spotify_api("PUT", "/me/player/play", access_token=token)
    if ok:
        return True, msg
    # If no active device, get devices and retry with a chosen device
    if "no active device" in msg.lower() or "device not found" in msg.lower():
        dev_ok, _, devices = _spotify_devices_sync(token)
        if dev_ok and devices:
            device_id = _pick_best_device(devices)
            if device_id:
                ok2, msg2 = _spotify_api(
                    "PUT",
                    "/me/player/play",
                    access_token=token,
                    params={"device_id": device_id},
                )
                if ok2:
                    return True, "Playing on your device."
                return False, msg2
        # No devices: try to launch Spotify on this PC, wait, then retry
        launched = _spotify_launch_desktop_sync()
        if launched:
            time.sleep(5)
            dev_ok2, _, devices2 = _spotify_devices_sync(token)
            if dev_ok2 and devices2:
                device_id = _pick_best_device(devices2)
                if device_id:
                    ok3, msg3 = _spotify_api(
                        "PUT",
                        "/me/player/play",
                        access_token=token,
                        params={"device_id": device_id},
                    )
                    if ok3:
                        return True, "Opened Spotify on your PC and started playing."
                    return False, msg3
            return False, "Opened Spotify. Wait a few seconds and try again."
        return False, "No active device found. Open Spotify on your PC or phone, then try again."
    return False, msg


def _spotify_pause_sync() -> tuple[bool, str]:
    token = _spotify_access_token()
    if not token:
        return False, "Spotify is not configured."
    return _spotify_api("PUT", "/me/player/pause", access_token=token)


def _spotify_next_sync() -> tuple[bool, str]:
    token = _spotify_access_token()
    if not token:
        return False, "Spotify is not configured."
    return _spotify_api("POST", "/me/player/next", access_token=token)


def _spotify_previous_sync() -> tuple[bool, str]:
    token = _spotify_access_token()
    if not token:
        return False, "Spotify is not configured."
    return _spotify_api("POST", "/me/player/previous", access_token=token)


def _spotify_volume_sync(volume_percent: int) -> tuple[bool, str]:
    token = _spotify_access_token()
    if not token:
        return False, "Spotify is not configured."
    return _spotify_api(
        "PUT",
        "/me/player/volume",
        access_token=token,
        params={"volume_percent": volume_percent},
    )


def _spotify_status_sync() -> tuple[bool, str, Optional[Dict[str, Any]]]:
    """Returns (success, message, data). data is current playback info or None."""
    token = _spotify_access_token()
    if not token:
        return False, "Spotify is not configured.", None
    url = "https://api.spotify.com/v1/me/player"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
        if r.status_code == 204:
            return True, "No active device or nothing playing.", None
        if r.status_code != 200:
            try:
                err = r.json().get("error", {})
                msg = err.get("message", r.text or str(r.status_code))
            except Exception:
                msg = r.text or str(r.status_code)
            return False, msg, None
        return True, "OK", r.json()
    except Exception as e:
        return False, str(e), None


# ---------------------------------------------------------------------------
# Security utility
# ---------------------------------------------------------------------------

def _is_path_within_allowed_workspaces(raw_path: str) -> tuple[bool, Path]:
    """
    Resolve ``raw_path`` to its canonical absolute form and check that it is
    strictly contained within one of the allowed workspaces.

    Returns
    -------
    (allowed: bool, resolved_path: Path)

    Defence layers
    --------------
    * ``Path.resolve()`` expands symlinks and removes ``..`` components,
      preventing path-traversal attacks like ``/allowed/../../etc/passwd``.
    * The prefix check uses string comparison on the *parts* of two resolved
      paths, not on raw strings, to avoid false positives from shared prefixes
      (e.g. ``/allowed_dir_extra`` matching against ``/allowed_dir``).
    """
    try:
        resolved = Path(raw_path).resolve()
    except (OSError, ValueError) as exc:
        logger.warning("Could not resolve workspace_path '%s': %s", raw_path, exc)
        return False, Path(raw_path)

    for workspace in settings.allowed_workspaces:
        workspace_path = Path(workspace)  # already resolved in config.py
        try:
            # Path.is_relative_to() was added in Python 3.9.
            # It returns True only when resolved is workspace_path itself
            # OR a descendant of it.
            resolved.relative_to(workspace_path)
            return True, resolved
        except ValueError:
            continue  # not relative to this workspace, try the next one

    return False, resolved


# ---------------------------------------------------------------------------
# Git helpers (sync — run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _git_uncommitted_diff_sync(workspace_path: str) -> tuple[bool, str, str, str, bool]:
    """Run git status and git diff in repo. Returns (success, message, status_short, full_diff, has_changes)."""
    allowed, resolved = _is_path_within_allowed_workspaces(workspace_path)
    if not allowed:
        return False, "Workspace path is not in ALLOWED_WORKSPACES.", "", "", False
    if not (resolved / ".git").exists():
        return False, "Not a git repository.", "", "", False
    try:
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status_short = (status_result.stdout or "").strip()
        diff_result = subprocess.run(
            ["git", "diff"],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=30,
        )
        staged_result = subprocess.run(
            ["git", "diff", "--staged"],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=30,
        )
        full_diff = (diff_result.stdout or "").strip()
        staged_diff = (staged_result.stdout or "").strip()
        if staged_diff:
            full_diff = "=== Staged changes ===\n" + staged_diff + "\n\n=== Unstaged changes ===\n" + full_diff
        has_changes = bool(status_short)
        if not full_diff and status_short:
            full_diff = (
                "Untracked/unstaged files (will be added with `git add -A`):\n\n"
                + status_short
            )
        return True, "OK", status_short, full_diff or "(no diff)", has_changes
    except subprocess.TimeoutExpired:
        return False, "Git command timed out.", "", "", False
    except Exception as e:
        return False, str(e), "", "", False


def _git_commit_and_push_sync(workspace_path: str, commit_message: str) -> tuple[bool, str, str, str]:
    """Run git add -A, git commit, git push. Returns (success, message, stdout, stderr)."""
    allowed, resolved = _is_path_within_allowed_workspaces(workspace_path)
    if not allowed:
        return False, "Workspace path is not in ALLOWED_WORKSPACES.", "", ""
    if not (resolved / ".git").exists():
        return False, "Not a git repository.", "", ""
    try:
        add_result = subprocess.run(
            ["git", "add", "-A"],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if add_result.returncode != 0:
            return False, "git add failed.", add_result.stdout or "", add_result.stderr or ""
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if commit_result.returncode != 0:
            if "nothing to commit" in (commit_result.stdout or "") + (commit_result.stderr or ""):
                return False, "Nothing to commit (working tree clean).", commit_result.stdout or "", commit_result.stderr or ""
            return False, "git commit failed.", commit_result.stdout or "", commit_result.stderr or ""
        push_result = subprocess.run(
            ["git", "push"],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if push_result.returncode != 0:
            return False, "git push failed.", push_result.stdout or "", push_result.stderr or ""
        return True, "Committed and pushed successfully.", push_result.stdout or "", push_result.stderr or ""
    except subprocess.TimeoutExpired:
        return False, "Git command timed out.", "", ""
    except Exception as e:
        return False, str(e), "", ""


def _git_commit_all_sync(workspace_path: str, commit_message: str) -> tuple[bool, str, str, str]:
    """Run git add -A and git commit. Returns (success, message, stdout, stderr)."""
    allowed, resolved = _is_path_within_allowed_workspaces(workspace_path)
    if not allowed:
        return False, "Workspace path is not in ALLOWED_WORKSPACES.", "", ""
    if not (resolved / ".git").exists():
        return False, "Not a git repository.", "", ""
    try:
        add_result = subprocess.run(
            ["git", "add", "-A"],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if add_result.returncode != 0:
            return False, "git add failed.", add_result.stdout or "", add_result.stderr or ""
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if commit_result.returncode != 0:
            if "nothing to commit" in (commit_result.stdout or "") + (commit_result.stderr or ""):
                return False, "Nothing to commit (working tree clean).", commit_result.stdout or "", commit_result.stderr or ""
            return False, "git commit failed.", commit_result.stdout or "", commit_result.stderr or ""
        return True, "Committed successfully.", commit_result.stdout or "", commit_result.stderr or ""
    except subprocess.TimeoutExpired:
        return False, "Git command timed out.", "", ""
    except Exception as e:
        return False, str(e), "", ""


def _git_push_only_sync(workspace_path: str, push_timeout: int = 60) -> tuple[bool, str, str, str]:
    """Run git push only (no add/commit). Returns (success, message, stdout, stderr)."""
    allowed, resolved = _is_path_within_allowed_workspaces(workspace_path)
    if not allowed:
        return False, "Workspace path is not in ALLOWED_WORKSPACES.", "", ""
    if not (resolved / ".git").exists():
        return False, "Not a git repository.", "", ""
    try:
        push_result = subprocess.run(
            ["git", "push"],
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=push_timeout,
        )
        if push_result.returncode != 0:
            return False, "git push failed.", push_result.stdout or "", push_result.stderr or ""
        return True, "Pushed successfully.", push_result.stdout or "", push_result.stderr or ""
    except subprocess.TimeoutExpired:
        return False, "Git command timed out.", "", ""
    except Exception as e:
        return False, str(e), "", ""


# Background push timeout (no request blocking, so we can allow long pushes)
_BACKGROUND_PUSH_TIMEOUT = 600


def _run_push_and_notify_sync(workspace_path: str, completion_webhook_url: str) -> None:
    """Run git push (long timeout) then POST result to completion_webhook_url. Runs in thread."""
    ok, msg, stdout, stderr = _git_push_only_sync(workspace_path, push_timeout=_BACKGROUND_PUSH_TIMEOUT)
    payload: Dict[str, Any] = {
        "success": ok,
        "message": msg,
        "stdout": stdout or "",
        "stderr": stderr or "",
    }
    try:
        requests.post(
            completion_webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except Exception as e:
        logger.warning("Failed to call completion webhook %s: %s", completion_webhook_url, e)


# ---------------------------------------------------------------------------
# Edit helpers (compute patch only; no disk write)
# ---------------------------------------------------------------------------

def _edit_should_ignore(rel_path: Path) -> bool:
    """True if any component of rel_path matches settings.ignore_patterns."""
    for part in rel_path.parts:
        for pattern in settings.ignore_patterns:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _edit_read_text(path: Path) -> Optional[str]:
    """Read file as text; return None if binary or unreadable."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue
    return None


def _build_unified_diff(rel_path_str: str, old_text: str, new_text: str) -> str:
    """Build a git-style unified diff for one file.

    ``rel_path_str`` is the path *relative to the git repo root* using forward
    slashes (e.g. ``rag-daemon/config.py``). ``old_text`` and ``new_text`` are
    the full file contents *before* and *after* the edit.
    """
    fromfile = "a/" + rel_path_str.replace("\\", "/")
    tofile = "b/" + rel_path_str.replace("\\", "/")
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=fromfile,
            tofile=tofile,
            fromfiledate="",
            tofiledate="",
            lineterm="\n",
        )
    )
    if not diff_lines:
        return ""
    return "".join(diff_lines)


def _compute_remove_line_patch_sync(
    workspace_path: str,
    file_path: str,
    line_number: int,
    line_end: Optional[int],
) -> tuple[bool, str, str]:
    """Compute a patch that removes the given line(s). Returns (success, message, patch)."""
    allowed, resolved_workspace = _is_path_within_allowed_workspaces(workspace_path)
    if not allowed:
        return False, "Workspace path is not in ALLOWED_WORKSPACES.", ""
    # Resolve file_path relative to workspace (no .. escape).
    try:
        full_path = (resolved_workspace / file_path.replace("\\", "/").strip("/")).resolve()
        full_path.relative_to(resolved_workspace)
    except (ValueError, OSError):
        return False, "file_path is outside the workspace.", ""
    if not full_path.is_file():
        return False, f"Not a file or does not exist: {full_path}", ""
    content = _edit_read_text(full_path)
    if content is None:
        return False, "Could not read file as text.", ""
    lines = content.splitlines(keepends=True)
    n = len(lines)
    start = line_number - 1
    end = (line_end if line_end is not None else line_number) - 1
    if start < 0 or end < start or end >= n:
        return False, f"Line number(s) out of range (file has {n} lines).", ""
    new_lines = lines[:start] + lines[end + 1 :]
    rel_path_str = str(full_path.relative_to(resolved_workspace)).replace("\\", "/")
    new_text = "".join(new_lines)
    patch = _build_unified_diff(rel_path_str, content, new_text)
    return True, "", patch


def _compute_remove_lines_matching_patch_sync(
    workspace_path: str,
    pattern: str,
    file_path: Optional[str],
    path_glob: str,
) -> tuple[bool, str, str, int]:
    """Compute a patch that removes all lines containing pattern. Returns (success, message, patch, files_affected)."""
    allowed, resolved_workspace = _is_path_within_allowed_workspaces(workspace_path)
    if not allowed:
        return False, "Workspace path is not in ALLOWED_WORKSPACES.", "", 0
    patches: List[str] = []
    files_affected = 0

    if file_path:
        full_path = (resolved_workspace / file_path.replace("\\", "/").strip("/")).resolve()
        try:
            full_path.relative_to(resolved_workspace)
        except ValueError:
            return False, "file_path is outside the workspace.", "", 0
        if not full_path.is_file():
            return False, f"Not a file or does not exist: {full_path}", "", 0
        content = _edit_read_text(full_path)
        if content is None:
            return False, "Could not read file as text.", "", 0
        lines = content.splitlines(keepends=True)
        new_lines = [line for line in lines if pattern not in line]
        if len(new_lines) == len(lines):
            return True, "", "", 0
        rel_path_str = str(full_path.relative_to(resolved_workspace)).replace("\\", "/")
        new_text = "".join(new_lines)
        patch = _build_unified_diff(rel_path_str, content, new_text)
        return True, "", patch, 1

    for path in resolved_workspace.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(resolved_workspace)
        except ValueError:
            continue
        if _edit_should_ignore(rel):
            continue
        if not fnmatch.fnmatch(str(rel).replace("\\", "/"), path_glob):
            continue
        content = _edit_read_text(path)
        if content is None:
            continue
        lines = content.splitlines(keepends=True)
        new_lines = [line for line in lines if pattern not in line]
        if len(new_lines) == len(lines):
            continue
        rel_path_str = str(rel).replace("\\", "/")
        new_text = "".join(new_lines)
        patch = _build_unified_diff(rel_path_str, content, new_text)
        patches.append(patch)
        files_affected += 1

    return True, "", "\n".join(patches), files_affected


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> Dict[str, Any]:
    """
    Liveness probe.

    Returns basic daemon status information.  Orchestrators should poll this
    endpoint before sending indexing or search requests.
    """
    indexed_count = _chroma_collection.count() if _chroma_collection else 0
    return {
        "status": "ok",
        "allowed_workspaces": settings.allowed_workspaces,
        "indexed_chunks": indexed_count,
        "embedding_model": settings.embedding_model,
    }


@app.get("/ping", tags=["System"])
async def ping() -> Dict[str, str]:
    """Simple liveness ping. Returns {"status": "ok"}."""
    return {"status": "ok"}


@app.get(
    "/list-folders",
    response_model=ListFoldersResponse,
    tags=["System"],
    summary="List top-level folders in each allowed workspace",
)
async def list_folders() -> ListFoldersResponse:
    """
    Return the immediate subdirectories of each path in ALLOWED_WORKSPACES.

    This gives the orchestrator (and the LLM) a safe, read-only view of the
    workspace structure without exposing arbitrary filesystem access.
    """
    workspaces: List[WorkspaceFolders] = []

    for raw_path in settings.allowed_workspaces:
        try:
            root = Path(raw_path).resolve()
        except (OSError, ValueError) as exc:
            logger.warning("Could not resolve allowed workspace '%s': %s", raw_path, exc)
            continue

        if not root.exists() or not root.is_dir():
            logger.warning(
                "Allowed workspace '%s' does not exist or is not a directory (resolved: '%s').",
                raw_path,
                root,
            )
            continue

        try:
            folders = sorted(
                entry.name
                for entry in root.iterdir()
                if entry.is_dir()
            )
        except OSError as exc:
            logger.warning(
                "Failed to list folders for workspace '%s': %s",
                root,
                exc,
            )
            continue

        workspaces.append(
            WorkspaceFolders(
                workspace_path=str(root),
                folders=folders,
            )
        )

    return ListFoldersResponse(workspaces=workspaces)


@app.get(
    "/list-folder-contents",
    response_model=ListFolderContentsResponse,
    tags=["System"],
    summary="List files and folders inside a directory within allowed workspaces",
)
async def list_folder_contents(folder_path: str) -> ListFolderContentsResponse:
    """
    Return the immediate files and folders inside the given directory, as long
    as it is contained within one of ALLOWED_WORKSPACES.
    """
    allowed, resolved = _is_path_within_allowed_workspaces(folder_path)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requested folder_path is not within any allowed workspace.",
        )

    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested folder does not exist or is not a directory.",
        )

    try:
        entries: List[FolderEntry] = []
        for entry in sorted(resolved.iterdir(), key=lambda p: p.name.lower()):
            entry_type: Literal["file", "folder"] = "folder" if entry.is_dir() else "file"
            entries.append(
                FolderEntry(
                    name=entry.name,
                    type=entry_type,
                )
            )
    except OSError as exc:
        logger.warning(
            "Failed to list contents for folder '%s': %s",
            resolved,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list folder contents.",
        ) from exc

    return ListFolderContentsResponse(
        folder_path=str(resolved),
        entries=entries,
    )


def _is_allowed_delete_path(resolved: Path) -> bool:
    """True if resolved is inside an allowed workspace and not the workspace root itself."""
    for workspace in settings.allowed_workspaces:
        workspace_path = Path(workspace).resolve()
        try:
            resolved.relative_to(workspace_path)
        except ValueError:
            continue
        if resolved == workspace_path:
            return False
        return True
    return False


@app.post(
    "/delete-path",
    response_model=DeletePathResponse,
    tags=["System"],
    summary="Delete a file or folder within an allowed workspace",
)
async def delete_path(request: DeletePathRequest) -> DeletePathResponse:
    """
    Delete the file or folder at the given path. The path must be inside one of
    ALLOWED_WORKSPACES and must not be the workspace root (to avoid deleting
    the entire workspace).
    """
    allowed, resolved = _is_path_within_allowed_workspaces(request.path)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requested path is not within any allowed workspace.",
        )
    if not _is_allowed_delete_path(resolved):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete an allowed workspace root. Only files or folders inside it.",
        )
    if not resolved.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Path does not exist.",
        )

    try:
        if resolved.is_dir():
            await asyncio.to_thread(shutil.rmtree, resolved)
            logger.info("Deleted folder: %s", resolved)
            return DeletePathResponse(success=True, message=f"Deleted folder: {resolved}")
        resolved.unlink()
        logger.info("Deleted file: %s", resolved)
        return DeletePathResponse(success=True, message=f"Deleted file: {resolved}")
    except OSError as exc:
        logger.warning("Failed to delete '%s': %s", resolved, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete: {exc!s}",
        ) from exc


@app.post(
    "/read-file",
    response_model=ReadFileResponse,
    tags=["System"],
    summary="Read a file's contents within an allowed workspace",
)
async def read_file(request: ReadFileRequest) -> ReadFileResponse:
    """
    Return the text contents of a file with line numbers.
    Supports optional start_line/end_line for reading a range.
    The path must be inside one of ALLOWED_WORKSPACES.
    """
    allowed, resolved = _is_path_within_allowed_workspaces(request.path.strip())
    if not allowed:
        return ReadFileResponse(
            success=False,
            path=request.path,
            error="Requested path is not within any allowed workspace.",
        )
    if not resolved.exists():
        return ReadFileResponse(success=False, path=str(resolved), error="Path does not exist.")
    if not resolved.is_file():
        return ReadFileResponse(success=False, path=str(resolved), error="Path is not a file.")
    raw_content = _edit_read_text(resolved)
    if raw_content is None:
        return ReadFileResponse(
            success=False,
            path=str(resolved),
            error="File could not be read as text (binary or unreadable).",
        )
    all_lines = raw_content.splitlines()
    total = len(all_lines)
    start = (request.start_line or 1) - 1
    end = request.end_line or total
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    selected = all_lines[start:end]
    # Prefix each line with its 1-based line number
    numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(selected)]
    content = "\n".join(numbered)
    return ReadFileResponse(success=True, path=str(resolved), content=content, total_lines=total)


def _run_command_sync(workspace_path: Path, command_line: str) -> tuple[bool, str, str, int]:
    """Run a whitelisted command in the given directory. Returns (success, stdout, stderr, exit_code)."""
    normalized = " ".join(command_line.strip().lower().split())
    if normalized not in RUN_COMMAND_WHITELIST:
        return False, "", f"Command not in whitelist. Allowed: {sorted(RUN_COMMAND_WHITELIST)}", -1
    try:
        result = subprocess.run(
            command_line.strip().split(),
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return True, result.stdout or "", result.stderr or "", result.returncode
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out (120s).", -1
    except Exception as e:
        return False, "", str(e), -1


@app.post(
    "/run-command",
    response_model=RunCommandResponse,
    tags=["System"],
    summary="Run a whitelisted command (e.g. tests) in a workspace",
)
async def run_command(request: RunCommandRequest) -> RunCommandResponse:
    """
    Run a whitelisted command in the given workspace. Only test-related commands
    are allowed (e.g. npm test, pytest). Used by the dev agent to run tests.
    """
    allowed, resolved = _is_path_within_allowed_workspaces(request.workspace_path.strip())
    if not allowed:
        return RunCommandResponse(
            success=False,
            error="Workspace path is not within any allowed workspace.",
        )
    if not resolved.exists() or not resolved.is_dir():
        return RunCommandResponse(success=False, error="Workspace path does not exist or is not a directory.")
    ok, stdout, stderr, exit_code = await asyncio.to_thread(
        _run_command_sync,
        resolved,
        request.command_line.strip(),
    )
    return RunCommandResponse(
        success=ok,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        error="" if ok else (stderr or "Command failed."),
    )


# ---------------------------------------------------------------------------
# System control (shutdown, sleep, restart) — runs on the machine where daemon runs
# ---------------------------------------------------------------------------

def _run_system_command_sync(cmd: List[str], delay_seconds: float = 0) -> tuple[bool, str]:
    """Run a system command (optionally after a delay). Returns (success, message)."""
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    try:
        subprocess.run(
            cmd,
            timeout=10,
            capture_output=True,
            check=False,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _system_shutdown_sync() -> tuple[bool, str]:
    """Shut down the machine. Returns after triggering (OS may delay a few seconds)."""
    system = platform.system()
    if system == "Windows":
        # /s shutdown, /t 3 = 3 sec delay so HTTP response can be sent first
        return _run_system_command_sync(["shutdown", "/s", "/t", "3"])
    if system == "Darwin":
        return _run_system_command_sync(["osascript", "-e", 'tell application "System Events" to shut down'])
    if system == "Linux":
        return _run_system_command_sync(["systemctl", "poweroff"])
    return False, f"Unsupported OS: {system}"


def _system_restart_sync() -> tuple[bool, str]:
    """Restart the machine."""
    system = platform.system()
    if system == "Windows":
        return _run_system_command_sync(["shutdown", "/r", "/t", "3"])
    if system == "Darwin":
        return _run_system_command_sync(["osascript", "-e", 'tell application "System Events" to restart'])
    if system == "Linux":
        return _run_system_command_sync(["systemctl", "reboot"])
    return False, f"Unsupported OS: {system}"


def _system_sleep_sync() -> tuple[bool, str]:
    """Put the machine to sleep (suspend)."""
    system = platform.system()
    if system == "Windows":
        # SetSuspendState(0, 1, 0) = Hibernate=False, Suspend=True, Force=False
        return _run_system_command_sync(
            ["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"]
        )
    if system == "Darwin":
        return _run_system_command_sync(["pmset", "sleepnow"])
    if system == "Linux":
        return _run_system_command_sync(["systemctl", "suspend"])
    return False, f"Unsupported OS: {system}"


def _system_lock_sync() -> tuple[bool, str]:
    """Lock the screen (like Win+L on Windows)."""
    system = platform.system()
    if system == "Windows":
        return _run_system_command_sync(["rundll32.exe", "user32.dll,LockWorkStation"])
    if system == "Darwin":
        # Control-Command-Q (key code 12 = q); CGSession was removed in Big Sur+
        ok, _ = _run_system_command_sync(
            ["osascript", "-e", 'tell application "System Events" to key code 12 using {control down, command down}']
        )
        if ok:
            return True, "OK"
        return _run_system_command_sync(["pmset", "displaysleepnow"])
    if system == "Linux":
        # Prefer loginctl (systemd); fallback to xdg-screensaver
        ok, _ = _run_system_command_sync(["loginctl", "lock-session"])
        if ok:
            return True, "OK"
        return _run_system_command_sync(["xdg-screensaver", "lock"])
    return False, f"Unsupported OS: {system}"


@app.post(
    "/system/shutdown",
    response_model=SpotifyGenericResponse,
    tags=["System"],
    summary="Shut down the PC",
)
async def system_shutdown() -> SpotifyGenericResponse:
    """Trigger a system shutdown. The machine will shut down in a few seconds."""
    ok, msg = await asyncio.to_thread(_system_shutdown_sync)
    return SpotifyGenericResponse(
        success=ok,
        message="PC will shut down in a few seconds." if ok else msg,
    )


@app.post(
    "/system/restart",
    response_model=SpotifyGenericResponse,
    tags=["System"],
    summary="Restart the PC",
)
async def system_restart() -> SpotifyGenericResponse:
    """Trigger a system restart. The machine will restart in a few seconds."""
    ok, msg = await asyncio.to_thread(_system_restart_sync)
    return SpotifyGenericResponse(
        success=ok,
        message="PC will restart in a few seconds." if ok else msg,
    )


@app.post(
    "/system/sleep",
    response_model=SpotifyGenericResponse,
    tags=["System"],
    summary="Put the PC to sleep",
)
async def system_sleep() -> SpotifyGenericResponse:
    """Put the machine to sleep (suspend)."""
    ok, msg = await asyncio.to_thread(_system_sleep_sync)
    return SpotifyGenericResponse(
        success=ok,
        message="PC is going to sleep." if ok else msg,
    )


@app.post(
    "/system/lock",
    response_model=SpotifyGenericResponse,
    tags=["System"],
    summary="Lock the PC",
)
async def system_lock() -> SpotifyGenericResponse:
    """Lock the screen (same as Win+L on Windows)."""
    ok, msg = await asyncio.to_thread(_system_lock_sync)
    return SpotifyGenericResponse(
        success=ok,
        message="PC locked." if ok else msg,
    )


# ---------------------------------------------------------------------------
# Git (uncommitted diff, commit and push)
# ---------------------------------------------------------------------------

@app.get(
    "/git/uncommitted-diff",
    response_model=GitUncommittedDiffResponse,
    tags=["Git"],
    summary="Get uncommitted changes (diff) for a repo",
)
async def git_uncommitted_diff(workspace_path: str) -> GitUncommittedDiffResponse:
    """Return git status --short and full diff (staged + unstaged) for the given repo."""
    ok, msg, status_short, full_diff, has_changes = await asyncio.to_thread(
        _git_uncommitted_diff_sync, workspace_path
    )
    allowed, resolved = _is_path_within_allowed_workspaces(workspace_path)
    path_str = str(resolved) if allowed else workspace_path
    return GitUncommittedDiffResponse(
        success=ok,
        message=msg,
        workspace_path=path_str,
        status_short=status_short,
        diff=full_diff,
        has_changes=has_changes,
    )


@app.get(
    "/git/find-repos",
    response_model=ListGitReposResponse,
    tags=["Git"],
    summary="Find git repositories under allowed workspaces",
)
async def git_find_repos() -> ListGitReposResponse:
    """
    Scan every directory under each path in ALLOWED_WORKSPACES and return a
    list of git repositories (directories containing a `.git` subdirectory).

    This gives the orchestrator a reliable way to resolve human-friendly
    project names like \"Eureka\" to absolute repo paths, without guessing.
    """
    seen: set[str] = set()
    repos: List[GitRepoInfo] = []

    for raw_path in settings.allowed_workspaces:
        try:
            root = Path(raw_path).resolve()
        except (OSError, ValueError) as exc:
            logger.warning("Could not resolve allowed workspace '%s': %s", raw_path, exc)
            continue

        if not root.exists() or not root.is_dir():
            logger.warning(
                "Allowed workspace '%s' does not exist or is not a directory (resolved: '%s').",
                raw_path,
                root,
            )
            continue

        try:
            for git_dir in root.rglob(".git"):
                if not git_dir.is_dir():
                    continue
                repo_root = git_dir.parent.resolve()
                repo_path_str = str(repo_root)
                if repo_path_str in seen:
                    continue
                seen.add(repo_path_str)
                repos.append(GitRepoInfo(name=repo_root.name, path=repo_path_str))
        except OSError as exc:
            logger.warning("Failed to scan for git repos under '%s': %s", root, exc)
            continue

    return ListGitReposResponse(repos=repos)


@app.post(
    "/git/commit-and-push",
    response_model=CommitAndPushResponse,
    tags=["Git"],
    summary="Commit all changes and push",
)
async def git_commit_and_push(request: CommitAndPushRequest):
    """Run git add -A, git commit -m <message>, git push in the given repo. If completion_webhook_url is set, push runs in background and result is POSTed there (202)."""
    if request.completion_webhook_url:
        # Run add+commit synchronously; push in background and notify via webhook.
        ok, msg, stdout, stderr = await asyncio.to_thread(
            _git_commit_all_sync,
            request.workspace_path,
            request.commit_message,
        )
        if not ok:
            return CommitAndPushResponse(success=False, message=msg, stdout=stdout, stderr=stderr)
        asyncio.create_task(
            asyncio.to_thread(
                _run_push_and_notify_sync,
                request.workspace_path,
                request.completion_webhook_url,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=CommitAndPushResponse(
                success=True,
                message="Commit done. Push started in background; you will be notified when it completes.",
                started=True,
            ).model_dump(),
        )
    ok, msg, stdout, stderr = await asyncio.to_thread(
        _git_commit_and_push_sync,
        request.workspace_path,
        request.commit_message,
    )
    return CommitAndPushResponse(success=ok, message=msg, stdout=stdout, stderr=stderr)


@app.post(
    "/git/commit-all",
    response_model=CommitAllResponse,
    tags=["Git"],
    summary="Commit all changes (no push)",
)
async def git_commit_all(request: CommitAllRequest) -> CommitAllResponse:
    """Run git add -A and git commit -m <message> in the given repo."""
    ok, msg, stdout, stderr = await asyncio.to_thread(
        _git_commit_all_sync,
        request.workspace_path,
        request.commit_message,
    )
    return CommitAllResponse(success=ok, message=msg, stdout=stdout, stderr=stderr)


@app.post(
    "/git/push-only",
    response_model=PushResponse,
    tags=["Git"],
    summary="Push current branch without committing",
)
async def git_push_only(request: PushRequest):
    """Run git push in the given repo without staging or committing. If completion_webhook_url is set, push runs in background and result is POSTed there (202)."""
    if request.completion_webhook_url:
        asyncio.create_task(
            asyncio.to_thread(
                _run_push_and_notify_sync,
                request.workspace_path,
                request.completion_webhook_url,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=PushResponse(
                success=True,
                message="Push started in background; you will be notified when it completes.",
                started=True,
            ).model_dump(),
        )
    ok, msg, stdout, stderr = await asyncio.to_thread(
        _git_push_only_sync,
        request.workspace_path,
    )
    return PushResponse(success=ok, message=msg, stdout=stdout, stderr=stderr)


@app.post(
    "/edit/compute-remove-line-patch",
    response_model=ComputeRemoveLinePatchResponse,
    tags=["Edit"],
    summary="Compute a patch that removes a line (or range) from a file",
)
async def compute_remove_line_patch(request: ComputeRemoveLinePatchRequest) -> ComputeRemoveLinePatchResponse:
    """Compute a unified diff that removes the given line(s). Does not apply the patch."""
    success, message, patch = await asyncio.to_thread(
        _compute_remove_line_patch_sync,
        request.workspace_path,
        request.file_path,
        request.line_number,
        request.line_end,
    )
    return ComputeRemoveLinePatchResponse(success=success, message=message, patch=patch)


@app.post(
    "/edit/compute-remove-lines-matching-patch",
    response_model=ComputeRemoveLinesMatchingPatchResponse,
    tags=["Edit"],
    summary="Compute a patch that removes all lines containing a pattern",
)
async def compute_remove_lines_matching_patch(
    request: ComputeRemoveLinesMatchingPatchRequest,
) -> ComputeRemoveLinesMatchingPatchResponse:
    """Compute a unified diff that removes every line containing pattern (in one file or all). Does not apply."""
    success, message, patch, files_affected = await asyncio.to_thread(
        _compute_remove_lines_matching_patch_sync,
        request.workspace_path,
        request.pattern,
        request.file_path,
        request.path_glob or "**/*",
    )
    return ComputeRemoveLinesMatchingPatchResponse(
        success=success,
        message=message,
        patch=patch,
        files_affected=files_affected,
    )


def _compute_replace_patch_sync(
    workspace_path: str,
    file_path: str,
    search_string: str,
    replace_string: str,
) -> tuple[bool, str, str]:
    """Exact search/replace → unified diff. Returns (success, message, patch)."""
    allowed, resolved_workspace = _is_path_within_allowed_workspaces(workspace_path)
    if not allowed:
        return False, "Workspace path is not in ALLOWED_WORKSPACES.", ""
    try:
        full_path = (resolved_workspace / file_path.replace("\\", "/").strip("/")).resolve()
        full_path.relative_to(resolved_workspace)
    except (ValueError, OSError):
        return False, "file_path is not inside the given workspace.", ""
    if not full_path.is_file():
        return False, f"File does not exist: {file_path}", ""
    original = _edit_read_text(full_path)
    if original is None:
        return False, "File could not be read as text.", ""
    if search_string not in original:
        # Retry with trailing whitespace stripped from each line (LLMs often
        # drop trailing spaces). Build a mapping so we can do the replacement
        # on the original content.
        def _strip_trailing(text: str) -> str:
            return "\n".join(line.rstrip() for line in text.split("\n"))
        stripped_original = _strip_trailing(original)
        stripped_search = _strip_trailing(search_string)
        if stripped_search not in stripped_original:
            return False, "search_string not found in file. Make sure it matches exactly (including whitespace and newlines).", ""
        # Found with stripped matching; do the replacement on the stripped version
        # and rebuild the file.
        stripped_replace = _strip_trailing(replace_string)
        new_text_stripped = stripped_original.replace(stripped_search, stripped_replace, 1)
        # Re-expand: since we only stripped trailing whitespace, using the stripped
        # version as the new content is safe (trailing whitespace removal is benign).
        original = stripped_original
        new_text = new_text_stripped
    else:
        new_text = original.replace(search_string, replace_string, 1)
    if new_text == original:
        return False, "Replacement produced no change.", ""
    rel_path_str = str(full_path.relative_to(resolved_workspace)).replace("\\", "/")
    patch = _build_unified_diff(rel_path_str, original, new_text)
    if not patch:
        return False, "Diff was empty after replacement.", ""
    return True, "Patch computed successfully.", patch


@app.post(
    "/edit/compute-replace-patch",
    response_model=ComputeReplacePatchResponse,
    tags=["Edit"],
    summary="Compute a unified diff from an exact search/replace in a file",
)
async def compute_replace_patch(
    request: ComputeReplacePatchRequest,
) -> ComputeReplacePatchResponse:
    """
    Read the target file, find search_string, replace with replace_string,
    and return a unified diff suitable for git apply. Does NOT write to disk.
    """
    success, message, patch = await asyncio.to_thread(
        _compute_replace_patch_sync,
        request.workspace_path,
        request.file_path,
        request.search_string,
        request.replace_string,
    )
    return ComputeReplacePatchResponse(success=success, message=message, patch=patch)


def _insert_code_sync(
    workspace_path: str,
    file_path: str,
    after_line: int,
    new_code: str,
) -> tuple[bool, str, str]:
    """Insert new_code after after_line (1-based, 0=prepend). Returns (success, message, patch)."""
    allowed, resolved_workspace = _is_path_within_allowed_workspaces(workspace_path)
    if not allowed:
        return False, "Workspace path is not in ALLOWED_WORKSPACES.", ""
    try:
        full_path = (resolved_workspace / file_path.replace("\\", "/").strip("/")).resolve()
        full_path.relative_to(resolved_workspace)
    except (ValueError, OSError):
        return False, "file_path is not inside the given workspace.", ""
    if not full_path.is_file():
        return False, f"File does not exist: {file_path}", ""
    original = _edit_read_text(full_path)
    if original is None:
        return False, "File could not be read as text.", ""
    lines = original.splitlines(keepends=True)
    if after_line < 0 or after_line > len(lines):
        return False, f"after_line {after_line} is out of range (file has {len(lines)} lines).", ""
    # Ensure new_code ends with a newline so it becomes a proper line
    code_to_insert = new_code if new_code.endswith("\n") else new_code + "\n"
    new_lines = lines[:after_line] + [code_to_insert] + lines[after_line:]
    new_text = "".join(new_lines)
    rel_path_str = str(full_path.relative_to(resolved_workspace)).replace("\\", "/")
    patch = _build_unified_diff(rel_path_str, original, new_text)
    if not patch:
        return False, "Diff was empty after insertion.", ""
    return True, "Patch computed successfully.", patch


@app.post(
    "/edit/insert-code",
    response_model=InsertCodeResponse,
    tags=["Edit"],
    summary="Compute a unified diff that inserts new code after a given line number",
)
async def insert_code(request: InsertCodeRequest) -> InsertCodeResponse:
    """
    Insert new_code after the given line number (1-based; 0 = prepend).
    Returns a unified diff suitable for git apply. Does NOT write to disk.
    """
    success, message, patch = await asyncio.to_thread(
        _insert_code_sync,
        request.workspace_path,
        request.file_path,
        request.after_line,
        request.new_code,
    )
    return InsertCodeResponse(success=success, message=message, patch=patch)


def _find_python_block(lines: List[str], search_term: str) -> Optional[tuple[int, int]]:
    """
    Find the top-level Python block (function, class, or decorated endpoint) that
    contains search_term. Returns (start_idx, end_idx) as 0-based line indices
    where start_idx is the first line of the block (including leading decorators)
    and end_idx is the last line of the block (exclusive — i.e. the line where
    the next top-level block starts, or len(lines)).

    A "top-level block" is a contiguous run of lines starting with a decorator (@)
    or def/async def/class at column 0, followed by indented lines (and blank lines
    between them).
    """
    # Find all lines that match the search_term
    candidate_lines = [i for i, line in enumerate(lines) if search_term in line]
    if not candidate_lines:
        return None

    def _is_toplevel_start(line: str) -> bool:
        stripped = line.lstrip()
        if not stripped:
            return False
        if line[0] == " " or line[0] == "\t":
            return False
        return stripped.startswith(("def ", "async def ", "class ", "@"))

    def _is_decorator(line: str) -> bool:
        return line.lstrip().startswith("@") and (not line[0].isspace())

    # For each candidate, walk backwards to find the block start (including decorators),
    # then walk forward to find the block end.
    for candidate_idx in candidate_lines:
        # Walk back to find the start of this block
        block_body_start = candidate_idx
        # First, find the def/class line at or above the candidate
        def_line = candidate_idx
        while def_line >= 0:
            stripped = lines[def_line].lstrip()
            if stripped.startswith(("def ", "async def ", "class ")):
                break
            if _is_decorator(lines[def_line]):
                break
            def_line -= 1
        if def_line < 0:
            continue

        # Walk further back to include decorators
        block_start = def_line
        while block_start > 0 and _is_decorator(lines[block_start - 1]):
            block_start -= 1

        # Walk forward from def_line to find end of block.
        # The body consists of indented lines. Blank lines between indented
        # lines are part of the body, but blank lines followed by a non-indented
        # line are NOT (they're separators between blocks).
        block_end = def_line + 1
        while block_end < len(lines):
            line = lines[block_end]
            # Indented line → part of the body
            if line.strip() and (line[0] == " " or line[0] == "\t"):
                block_end += 1
                continue
            # Blank line → only include if followed by more indented body
            if line.strip() == "":
                lookahead = block_end + 1
                while lookahead < len(lines) and lines[lookahead].strip() == "":
                    lookahead += 1
                if lookahead < len(lines) and lines[lookahead].strip() and (lines[lookahead][0] == " " or lines[lookahead][0] == "\t"):
                    block_end = lookahead
                    continue
                break
            # Non-indented, non-blank → next top-level thing, stop
            break

        # Trim trailing blank lines from the block
        while block_end > block_start and lines[block_end - 1].strip() == "":
            block_end -= 1

        return (block_start, block_end)

    return None


def _delete_block_sync(
    workspace_path: str,
    file_path: str,
    search_term: str,
) -> tuple[bool, str, str, str]:
    """Find and delete a top-level block containing search_term. Returns (success, message, patch, deleted_lines_preview)."""
    allowed, resolved_workspace = _is_path_within_allowed_workspaces(workspace_path)
    if not allowed:
        return False, "Workspace path is not in ALLOWED_WORKSPACES.", "", ""
    try:
        full_path = (resolved_workspace / file_path.replace("\\", "/").strip("/")).resolve()
        full_path.relative_to(resolved_workspace)
    except (ValueError, OSError):
        return False, "file_path is not inside the given workspace.", "", ""
    if not full_path.is_file():
        return False, f"File does not exist: {file_path}", "", ""
    original = _edit_read_text(full_path)
    if original is None:
        return False, "File could not be read as text.", "", ""

    lines = original.splitlines(keepends=True)
    lines_stripped = [l.rstrip("\n").rstrip("\r") for l in lines]
    result = _find_python_block(lines_stripped, search_term)
    if result is None:
        return False, f"No top-level block containing '{search_term}' was found in {file_path}.", "", ""

    block_start, block_end = result
    deleted_preview = "".join(lines[block_start:block_end]).strip()
    # Build the new file without the block. Remove at most 1 blank line after
    # to avoid leaving excessive gaps, but keep other blank lines intact.
    remove_trailing = 0
    if block_end < len(lines) and lines[block_end].strip() == "":
        remove_trailing = 1
    new_lines = lines[:block_start] + lines[block_end + remove_trailing:]
    new_text = "".join(new_lines)

    rel_path_str = str(full_path.relative_to(resolved_workspace)).replace("\\", "/")
    patch = _build_unified_diff(rel_path_str, original, new_text)
    if not patch:
        return False, "Diff was empty after deletion.", "", ""
    return True, f"Block deleted (lines {block_start + 1}-{block_end}).", patch, deleted_preview[:500]


@app.post(
    "/edit/delete-block",
    response_model=DeleteBlockResponse,
    tags=["Edit"],
    summary="Compute a patch that deletes a top-level function/class/endpoint block",
)
async def delete_block(request: DeleteBlockRequest) -> DeleteBlockResponse:
    """
    Find the top-level Python block (function, class, decorated endpoint) that
    contains search_term, and return a unified diff that removes it.
    Does NOT write to disk.
    """
    success, message, patch, deleted = await asyncio.to_thread(
        _delete_block_sync,
        request.workspace_path,
        request.file_path,
        request.search_term,
    )
    return DeleteBlockResponse(success=success, message=message, patch=patch, deleted_lines=deleted)


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["RAG"],
    summary="Semantic code search",
)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Embed *query* using the local model and perform a nearest-neighbour search
    against the ChromaDB collection.

    Returns the *top_k* most semantically similar code chunks together with
    their source file paths and line numbers.
    """
    if _embed_model is None or _chroma_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Daemon is still initialising.  Please retry shortly.",
        )

    # Embed the query in a background thread to avoid blocking the event loop.
    query_embedding: List[float] = await asyncio.to_thread(
        lambda: _embed_model.encode(
            request.query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
    )

    # Query ChromaDB.  ``include`` controls which fields are returned.
    raw = _chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k,
        include=["distances", "documents", "metadatas"],
    )

    # ChromaDB wraps results in a list-of-lists (one inner list per query).
    ids: List[str] = raw.get("ids", [[]])[0]
    distances: List[float] = raw.get("distances", [[]])[0]
    documents: List[str] = raw.get("documents", [[]])[0]
    metadatas: List[Dict] = raw.get("metadatas", [[]])[0]

    results: List[SearchResultItem] = []
    for chunk_id, distance, doc, meta in zip(ids, distances, documents, metadatas):
        # ChromaDB returns *cosine distance* (0 = identical, 2 = opposite).
        # Convert to similarity score in [−1, 1] for interpretability.
        similarity = 1.0 - distance
        results.append(
            SearchResultItem(
                chunk_id=chunk_id,
                file_path=meta.get("file_path", ""),
                start_line=int(meta.get("start_line", 0)),
                end_line=int(meta.get("end_line", 0)),
                score=round(similarity, 4),
                text=doc,
            )
        )

    return SearchResponse(query=request.query, results=results)


@app.post(
    "/apply-patch",
    response_model=ApplyPatchResponse,
    tags=["Patch"],
    summary="Apply a git patch to a verified workspace",
)
async def apply_patch(request: ApplyPatchRequest) -> ApplyPatchResponse:
    """
    Apply *patch_string* to the git repository at *workspace_path*.

    **Security contract**: this endpoint will reject any request where
    ``workspace_path`` is not strictly contained within ``ALLOWED_WORKSPACES``.
    The check is performed on the *resolved* (canonical) path so that symlinks
    and path-traversal sequences (``..``) cannot be used to escape the sandbox.

    The patch is applied in two stages:
      1. ``git apply --check`` — dry-run to verify the patch is valid.
      2. ``git apply``         — actual application only if the dry-run passes.

    Both commands receive the patch via stdin so it is never written to disk.
    """
    # ------------------------------------------------------------------ #
    # SECURITY GATE A: Workspace containment check                        #
    # ------------------------------------------------------------------ #
    allowed, resolved_workspace = _is_path_within_allowed_workspaces(
        request.workspace_path
    )
    if not allowed:
        # Log the attempted escape so it can be audited.
        logger.warning(
            "SECURITY: /apply-patch rejected for path '%s' (resolved: '%s'). "
            "Not within ALLOWED_WORKSPACES.",
            request.workspace_path,
            resolved_workspace,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"workspace_path '{request.workspace_path}' is not within any "
                "allowed workspace.  This request has been logged."
            ),
        )

    # ------------------------------------------------------------------ #
    # SECURITY GATE B: Verify the workspace is a git repository          #
    # ------------------------------------------------------------------ #
    # Refuse to run 'git apply' in a directory that has no .git directory,
    # since it would have no meaning and could indicate path manipulation.
    if not (resolved_workspace / ".git").exists():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"'{resolved_workspace}' does not appear to be the root of a "
                "git repository (no .git directory found)."
            ),
        )

    # Normalize and validate patch (strip markdown, require unified diff, trailing newline)
    normalized, norm_error = _normalize_patch_string(request.patch_string)
    if norm_error:
        logger.warning("/apply-patch validation failed: %s", norm_error)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=norm_error,
        )

    patch_bytes = normalized.encode("utf-8")

    # ------------------------------------------------------------------ #
    # Stage 1: Dry-run (--check)                                          #
    # ------------------------------------------------------------------ #
    logger.info(
        "Dry-running patch in '%s' (patch size: %d bytes).",
        resolved_workspace, len(patch_bytes),
    )
    try:
        check_result = await asyncio.to_thread(
            _run_git_apply,
            resolved_workspace,
            patch_bytes,
            dry_run=True,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="'git' executable not found on PATH.  Please install git.",
        )

    if check_result.returncode != 0:
        logger.warning(
            "Patch dry-run FAILED in '%s': %s", resolved_workspace, check_result.stderr
        )
        return ApplyPatchResponse(
            success=False,
            message="Patch dry-run (--check) failed.  Patch was NOT applied.",
            stdout=check_result.stdout,
            stderr=check_result.stderr,
        )

    # ------------------------------------------------------------------ #
    # Stage 2: Actually apply the patch                                   #
    # ------------------------------------------------------------------ #
    logger.info("Patch dry-run passed.  Applying patch in '%s'.", resolved_workspace)
    apply_result = await asyncio.to_thread(
        _run_git_apply,
        resolved_workspace,
        patch_bytes,
        dry_run=False,
    )

    if apply_result.returncode != 0:
        logger.error(
            "Patch application FAILED in '%s': %s",
            resolved_workspace, apply_result.stderr,
        )
        return ApplyPatchResponse(
            success=False,
            message="Patch application failed after dry-run passed.",
            stdout=apply_result.stdout,
            stderr=apply_result.stderr,
        )

    logger.info("Patch applied successfully in '%s'.", resolved_workspace)

    # ------------------------------------------------------------------ #
    # Post-patch: trigger re-index of the affected workspace              #
    # ------------------------------------------------------------------ #
    if _indexer is not None:
        asyncio.create_task(
            asyncio.to_thread(_indexer.index_path_sync, resolved_workspace),
            name=f"reindex-{resolved_workspace}",
        )
        logger.info("Scheduled re-index of '%s' after patch.", resolved_workspace)

    return ApplyPatchResponse(
        success=True,
        message=f"Patch applied successfully to '{resolved_workspace}'.",
        stdout=apply_result.stdout,
        stderr=apply_result.stderr,
    )


@app.post(
    "/index-now",
    response_model=IndexNowResponse,
    tags=["System"],
    summary="Trigger a full re-index of all allowed workspaces",
)
async def index_now() -> IndexNowResponse:
    """
    Immediately trigger a background re-index of all allowed workspaces.

    Useful after bulk file changes that aren't driven through /apply-patch.
    Returns immediately; indexing runs in a background thread.
    """
    if _indexer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Indexer not yet initialised.",
        )
    asyncio.create_task(
        asyncio.to_thread(_indexer.index_all_sync),
        name="manual-reindex",
    )
    return IndexNowResponse(message="Re-indexing started in the background.")


# ---------------------------------------------------------------------------
# Spotify control (optional — requires SPOTIFY_* env vars)
# ---------------------------------------------------------------------------

def _spotify_unavailable() -> bool:
    return not all([
        getattr(settings, "spotify_client_id", None),
        getattr(settings, "spotify_client_secret", None),
        getattr(settings, "spotify_refresh_token", None),
    ])


@app.post(
    "/spotify/play",
    response_model=SpotifyGenericResponse,
    tags=["Spotify"],
    summary="Start or resume playback",
)
async def spotify_play() -> SpotifyGenericResponse:
    if _spotify_unavailable():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify is not configured. Set SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REFRESH_TOKEN.",
        )
    ok, msg = await asyncio.to_thread(_spotify_play_sync)
    return SpotifyGenericResponse(success=ok, message=msg)


@app.post(
    "/spotify/pause",
    response_model=SpotifyGenericResponse,
    tags=["Spotify"],
    summary="Pause playback",
)
async def spotify_pause() -> SpotifyGenericResponse:
    if _spotify_unavailable():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify is not configured.",
        )
    ok, msg = await asyncio.to_thread(_spotify_pause_sync)
    return SpotifyGenericResponse(success=ok, message=msg)


@app.post(
    "/spotify/next",
    response_model=SpotifyGenericResponse,
    tags=["Spotify"],
    summary="Skip to next track",
)
async def spotify_next() -> SpotifyGenericResponse:
    if _spotify_unavailable():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify is not configured.",
        )
    ok, msg = await asyncio.to_thread(_spotify_next_sync)
    return SpotifyGenericResponse(success=ok, message=msg)


@app.post(
    "/spotify/previous",
    response_model=SpotifyGenericResponse,
    tags=["Spotify"],
    summary="Skip to previous track",
)
async def spotify_previous() -> SpotifyGenericResponse:
    if _spotify_unavailable():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify is not configured.",
        )
    ok, msg = await asyncio.to_thread(_spotify_previous_sync)
    return SpotifyGenericResponse(success=ok, message=msg)


@app.post(
    "/spotify/volume",
    response_model=SpotifyGenericResponse,
    tags=["Spotify"],
    summary="Set volume (0–100)",
)
async def spotify_volume(request: SpotifyVolumeRequest) -> SpotifyGenericResponse:
    if _spotify_unavailable():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify is not configured.",
        )
    ok, msg = await asyncio.to_thread(_spotify_volume_sync, request.volume_percent)
    return SpotifyGenericResponse(success=ok, message=msg)


@app.get(
    "/spotify/status",
    tags=["Spotify"],
    summary="Get current playback status",
)
async def spotify_status() -> Dict[str, Any]:
    if _spotify_unavailable():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Spotify is not configured.",
        )
    ok, msg, data = await asyncio.to_thread(_spotify_status_sync)
    if not ok:
        return {"success": False, "message": msg}
    if data is None:
        return {"success": True, "message": msg}
    item = data.get("item") or {}
    return {
        "success": True,
        "message": msg,
        "is_playing": data.get("is_playing", False),
        "device": (data.get("device") or {}).get("name"),
        "track": item.get("name"),
        "artist": ", ".join(a.get("name", "") for a in (item.get("artists") or [])),
        "album": (item.get("album") or {}).get("name"),
    }


@app.post(
    "/spotify/close",
    response_model=SpotifyGenericResponse,
    tags=["Spotify"],
    summary="Close the Spotify desktop app",
)
async def spotify_close() -> SpotifyGenericResponse:
    """Quit the Spotify application on this machine (does not require API credentials)."""
    ok, msg = await asyncio.to_thread(_spotify_close_desktop_sync)
    return SpotifyGenericResponse(success=ok, message=msg)


# ---------------------------------------------------------------------------
# Patch string normalization (LLM often wraps in markdown or sends invalid input)
# ---------------------------------------------------------------------------

def _normalize_patch_string(raw: str) -> tuple[str, Optional[str]]:
    """
    Normalize and validate a patch string before passing to git apply.

    - Strips leading/trailing whitespace.
    - If wrapped in ```diff ... ``` or ``` ... ```, extracts the inner content.
    - Ensures the result starts with "--- " (unified diff header) and ends with newline.
    Returns (normalized_string, None) on success, or ("", error_message) on failure.
    """
    s = raw.strip()
    if not s:
        return "", "Patch string is empty after trimming."
    # Unwrap markdown code blocks if present
    code_block = re.compile(r"^```(?:diff)?\s*\n(.*?)\n```\s*$", re.DOTALL)
    m = code_block.match(s)
    if m:
        s = m.group(1).strip()
    if not s:
        return "", "Patch string is empty after removing markdown fence."
    # Require unified diff format (starts with --- )
    if not s.startswith("--- "):
        preview = s[:80].replace("\n", " ")
        return "", (
            "Patch does not look like a unified diff (must start with '--- a/path'). "
            f"Got: {preview!r}…"
        )
    if not s.endswith("\n"):
        s = s + "\n"
    return s, None


# ---------------------------------------------------------------------------
# Subprocess helper  (kept synchronous — called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _run_git_apply(
    cwd: Path,
    patch_bytes: bytes,
    dry_run: bool,
) -> subprocess.CompletedProcess:
    """
    Run ``git apply [--check]`` in *cwd*, feeding *patch_bytes* via stdin.

    Parameters
    ----------
    cwd        : working directory (must be a valid git root)
    patch_bytes: the raw unified diff, UTF-8 encoded
    dry_run    : if True, passes ``--check`` to validate without modifying files

    Returns
    -------
    subprocess.CompletedProcess with returncode, stdout, and stderr.

    Security notes
    --------------
    * ``input=patch_bytes`` delivers the patch through a pipe to git's stdin,
      avoiding temp files.
    * ``shell=False`` prevents shell injection (cmd is a list, not a string).
    * ``timeout=60`` prevents the subprocess from hanging indefinitely.
    """
    cmd: List[str] = ["git", "apply"]
    if dry_run:
        cmd.append("--check")
    # Ignore whitespace/line-ending differences so LF patches apply on CRLF files (Windows).
    cmd.append("--ignore-whitespace")
    # Fix trailing whitespace etc. in applied hunks.
    cmd.append("--whitespace=fix")
    # Read patch from stdin.
    cmd.append("-")

    return subprocess.run(
        cmd,
        input=patch_bytes,
        capture_output=True,
        cwd=str(cwd),
        timeout=60,
        shell=False,  # IMPORTANT: never True — prevents shell injection.
    )


# ---------------------------------------------------------------------------
# Global exception handler — prevent leaking internal tracebacks to clients
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler that logs the full traceback server-side but returns
    only a generic error message to the client, preventing information leakage.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,  # Set to True during development.
    )
