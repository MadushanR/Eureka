"""
host_worker.py — Eureka Bare-Metal Host Worker
===============================================
A standalone, headless Python script that replaces the FastAPI daemon.

Instead of exposing an HTTP server, it connects directly to Upstash Redis
and uses BLPOP to receive tasks pushed by the Next.js orchestrator on Vercel.

Two task patterns are supported:
  1. Fire-and-forget (apply_patch, git_push, etc.): task includes `sender_id`
     and `bot_token` — the worker sends the Telegram result notification itself.
  2. Request/response (search, read_file, etc.): task includes `reply_to` key.
     The worker sets the result at that Redis key; Next.js polls for it.

Environment variables (all from .env in this directory):
  UPSTASH_REDIS_URL          — Raw TCP Redis URL: rediss://:{token}@host.upstash.io:6379
  TELEGRAM_BOT_TOKEN         — Bot token (for async Telegram notifications)
  ALLOWED_WORKSPACES         — JSON array of allowed absolute paths
  OPENAI_API_KEY             — Injected into aider subprocesses
  GITHUB_TOKEN               — Optional, for GitHub repo creation
  SPOTIFY_CLIENT_ID          — Optional, for Spotify control
  SPOTIFY_CLIENT_SECRET      — Optional, for Spotify control
  SPOTIFY_REFRESH_TOKEN      — Optional, for Spotify control

Usage:
  python host_worker.py
"""

from __future__ import annotations

import ctypes
import difflib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import psutil
import redis as redis_lib
import requests
import yt_dlp
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("host_worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("UPSTASH_REDIS_URL", "")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ALLOWED_WORKSPACES: list[str] = json.loads(os.environ.get("ALLOWED_WORKSPACES", "[]"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REFRESH_TOKEN = os.environ.get("SPOTIFY_REFRESH_TOKEN", "")
TASK_QUEUE = "eureka:host_commands"
# How often (seconds) to flush the output buffer back to Telegram.
_CLAUDE_STREAM_INTERVAL = 5.0

if not REDIS_URL:
    log.error("UPSTASH_REDIS_URL is not set. Set it in .env and restart.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Path validation (mirrors config.py logic)
# ---------------------------------------------------------------------------

def _is_allowed(path: str) -> bool:
    """Return True if `path` is strictly within one of ALLOWED_WORKSPACES."""
    try:
        p = Path(path).resolve()
        for ws in ALLOWED_WORKSPACES:
            ws_resolved = Path(ws).resolve()
            if p == ws_resolved or str(p).startswith(str(ws_resolved) + os.sep):
                return True
        return False
    except Exception:
        return False


def _require_allowed(path: str) -> str | None:
    """Return canonical path string if allowed, else None."""
    if not _is_allowed(path):
        return None
    return str(Path(path).resolve())

# ---------------------------------------------------------------------------
# Telegram helper
# ---------------------------------------------------------------------------

def _tg(chat_id: str, text: str, bot_token: str | None = None) -> None:
    token = bot_token or BOT_TOKEN
    if not token or not chat_id:
        return
    for attempt in range(3):
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text[:4096]},
                timeout=30,
            )
            return
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                log.warning("Telegram send failed: %s", e)

# ---------------------------------------------------------------------------
# Streaming callback helper
# ---------------------------------------------------------------------------

def _post_worker_callback(text: str, sender_id: str, bot_token: str) -> None:
    """Send a streaming chunk directly to Telegram."""
    _tg(sender_id, text, bot_token)


# ---------------------------------------------------------------------------
# Action handlers — strict dispatch (no arbitrary shell execution)
# ---------------------------------------------------------------------------

# --- Build / Execution -------------------------------------------------------

def _handle_execute_build_step(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    terminal_commands: list[str] = payload.get("terminal_commands", [])
    step_name: str = payload.get("step_name", "")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "error": f"Path not in ALLOWED_WORKSPACES: {workspace_path}",
                "stdout": "", "stderr": "", "exit_code": -1}

    os.makedirs(resolved, exist_ok=True)
    env = {**os.environ, "OPENAI_API_KEY": OPENAI_API_KEY}

    all_stdout: list[str] = []
    all_stderr: list[str] = []

    # Safety: reject obviously destructive patterns
    _DANGEROUS = ["rm -rf /", "mkfs", "format c:", ":(){:|:&};:"]
    for cmd in terminal_commands:
        if any(d in cmd.lower() for d in _DANGEROUS):
            return {
                "success": False,
                "error": f"Command blocked (matches dangerous pattern): {cmd}",
                "stdout": "", "stderr": "", "exit_code": -1,
            }

    for cmd in terminal_commands:
        log.info("[build] [%s] Running: %s", step_name, cmd[:120])
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=resolved,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
            all_stdout.append(result.stdout)
            all_stderr.append(result.stderr)
            if result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "\n".join(all_stdout)[:4000],
                    "stderr": "\n".join(all_stderr)[:2000],
                    "exit_code": result.returncode,
                    "error": f"Command failed (exit {result.returncode}): {cmd}",
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "\n".join(all_stdout)[:4000],
                "stderr": "\n".join(all_stderr)[:2000],
                "exit_code": -1,
                "error": f"Command timed out after 600 s: {cmd}",
            }

    return {
        "success": True,
        "stdout": "\n".join(all_stdout)[:4000],
        "stderr": "\n".join(all_stderr)[:2000],
        "exit_code": 0,
        "error": "",
    }


# --- Patch / Git helpers -----------------------------------------------------

def _git(args: list[str], cwd: str, stdin: str | None = None, timeout: int = 120) -> tuple[bool, str, str]:
    """Run a git command. Returns (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"git {args[0]} timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def _handle_apply_patch(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    patch_string: str = payload.get("patch_string", "")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": f"Path not in ALLOWED_WORKSPACES: {workspace_path}"}

    # Dry-run first
    ok, _, err = _git(["apply", "--check"], resolved, stdin=patch_string)
    if not ok:
        return {"success": False, "message": f"Patch check failed: {err.strip()[:400]}"}

    ok, out, err = _git(["apply"], resolved, stdin=patch_string)
    if not ok:
        return {"success": False, "message": f"Patch apply failed: {err.strip()[:400]}"}

    return {"success": True, "message": "Patch applied successfully.", "stdout": out, "stderr": err}


def _handle_git_commit_and_push(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    commit_message: str = payload.get("commit_message", "Update from Eureka")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": f"Path not in ALLOWED_WORKSPACES: {workspace_path}"}

    _git(["add", "-A"], resolved)
    ok, out, err = _git(["commit", "-m", commit_message], resolved)
    if not ok and "nothing to commit" not in err.lower() and "nothing to commit" not in out.lower():
        return {"success": False, "message": f"Commit failed: {err.strip()[:400]}"}

    ok, out, err = _git(["push"], resolved, timeout=120)
    if not ok:
        return {"success": False, "message": f"Push failed: {err.strip()[:400]}", "stdout": out, "stderr": err}

    return {"success": True, "message": f"Committed and pushed. {out.strip()[:200]}"}


def _handle_git_push_only(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": f"Path not in ALLOWED_WORKSPACES: {workspace_path}"}

    ok, out, err = _git(["push"], resolved, timeout=120)
    if not ok:
        return {"success": False, "message": f"Push failed: {err.strip()[:400]}", "stdout": out, "stderr": err}

    return {"success": True, "message": f"Push completed. {out.strip()[:200]}"}


def _handle_git_push(payload: dict) -> dict:
    """Final git push for build pipeline (may set a new remote first)."""
    workspace_path: str = payload.get("workspace_path", "")
    commit_message: str = payload.get("commit_message", "feat: built by Eureka")
    clone_url: str | None = payload.get("clone_url")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": f"Path not in ALLOWED_WORKSPACES: {workspace_path}",
                "stdout": "", "stderr": ""}

    _git(["add", "-A"], resolved)
    _git(["commit", "-m", commit_message], resolved)

    if clone_url:
        _git(["remote", "remove", "origin"], resolved)
        _git(["remote", "add", "origin", clone_url], resolved)
        _git(["branch", "-M", "main"], resolved)
        ok, out, err = _git(["push", "-u", "origin", "main"], resolved, timeout=180)
    else:
        ok, out, err = _git(["push"], resolved, timeout=180)

    if not ok:
        return {"success": False, "message": f"Push failed: {err.strip()[:400]}", "stdout": out, "stderr": err}

    return {"success": True, "message": out.strip()[:300] or "Push completed.", "stdout": out, "stderr": err}


def _handle_git_commit_all(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    commit_message: str = payload.get("commit_message", "Update from Eureka")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": f"Path not in ALLOWED_WORKSPACES: {workspace_path}"}

    _git(["add", "-A"], resolved)
    ok, out, err = _git(["commit", "-m", commit_message], resolved)
    if not ok and "nothing to commit" not in err.lower() and "nothing to commit" not in out.lower():
        return {"success": False, "message": f"Commit failed: {err.strip()[:400]}"}

    return {"success": True, "message": "Committed. " + out.strip()[:200]}


def _handle_git_clone(payload: dict) -> dict:
    clone_url: str = payload.get("clone_url", "")
    parent_directory: str = payload.get("parent_directory", "")
    folder_name: str | None = payload.get("folder_name")

    resolved_parent = _require_allowed(parent_directory)
    if resolved_parent is None:
        return {"success": False, "message": f"parent_directory not in ALLOWED_WORKSPACES: {parent_directory}"}

    if not clone_url:
        return {"success": False, "message": "clone_url is required."}

    cmd = ["git", "clone", clone_url]
    if folder_name:
        cmd.append(folder_name)
    local_path = Path(resolved_parent) / (folder_name or clone_url.rstrip("/").split("/")[-1].removesuffix(".git"))

    try:
        result = subprocess.run(cmd, cwd=resolved_parent, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return {"success": False, "message": result.stderr.strip()[:400]}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "git clone timed out after 180s"}

    return {"success": True, "message": "Cloned successfully.", "local_path": str(local_path)}


# --- Uncommitted diff --------------------------------------------------------

def _handle_uncommitted_diff(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": f"Path not in ALLOWED_WORKSPACES: {workspace_path}"}

    ok_status, status_out, _ = _git(["status", "--short"], resolved)
    if not ok_status:
        return {"success": False, "message": "git status failed."}

    has_changes = bool(status_out.strip())
    if not has_changes:
        return {"success": True, "has_changes": False, "diff": "", "status_short": ""}

    _, diff_out, _ = _git(["diff", "HEAD"], resolved)
    if not diff_out.strip():
        # Staged but not committed
        _, diff_out, _ = _git(["diff", "--cached"], resolved)

    return {
        "success": True,
        "has_changes": True,
        "diff": diff_out[:6000],
        "status_short": status_out.strip()[:500],
    }


# --- Git repo discovery -------------------------------------------------------

def _handle_list_git_repos(_payload: dict) -> dict:
    repos: list[dict] = []
    for ws in ALLOWED_WORKSPACES:
        ws_path = Path(ws)
        if not ws_path.exists():
            continue
        if (ws_path / ".git").exists():
            repos.append({"name": ws_path.name, "path": str(ws_path)})
        else:
            for child in ws_path.iterdir():
                if child.is_dir() and (child / ".git").exists():
                    repos.append({"name": child.name, "path": str(child)})
    return {"repos": repos}


# --- Folder operations -------------------------------------------------------

def _handle_list_folders(_payload: dict) -> dict:
    workspaces: list[dict] = []
    for ws in ALLOWED_WORKSPACES:
        ws_path = Path(ws)
        if not ws_path.exists():
            continue
        try:
            folders = [
                entry.name for entry in ws_path.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            ]
            workspaces.append({"workspace_path": str(ws_path), "folders": sorted(folders)})
        except PermissionError:
            pass
    return {"workspaces": workspaces}


def _handle_list_folder_contents(payload: dict) -> dict:
    folder_path: str = payload.get("folder_path", "")

    resolved = _require_allowed(folder_path)
    if resolved is None:
        return {"success": False, "error": f"Path not in ALLOWED_WORKSPACES: {folder_path}"}

    try:
        entries = []
        for entry in Path(resolved).iterdir():
            entries.append({
                "name": entry.name,
                "type": "folder" if entry.is_dir() else "file",
                "path": str(entry),
            })
        entries.sort(key=lambda e: (e["type"] == "file", e["name"].lower()))
        return {"success": True, "path": resolved, "entries": entries}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- File operations ---------------------------------------------------------

def _handle_read_file(payload: dict) -> dict:
    path: str = payload.get("path", "")
    start_line: int | None = payload.get("start_line")
    end_line: int | None = payload.get("end_line")

    resolved = _require_allowed(path)
    if resolved is None:
        return {"success": False, "error": f"Path not in ALLOWED_WORKSPACES: {path}"}

    try:
        text = Path(resolved).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        if start_line is not None or end_line is not None:
            s = (start_line or 1) - 1
            e = end_line or total_lines
            lines = lines[s:e]
        numbered = "".join(f"{i + (start_line or 1)}| {line}" for i, line in enumerate(lines))
        return {"success": True, "path": resolved, "content": numbered, "total_lines": total_lines}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _handle_create_file(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    file_path: str = payload.get("file_path", "")
    content: str = payload.get("content", "")

    resolved_ws = _require_allowed(workspace_path)
    if resolved_ws is None:
        return {"success": False, "message": f"workspace_path not in ALLOWED_WORKSPACES: {workspace_path}"}

    abs_path = Path(resolved_ws) / file_path
    # Prevent directory traversal escaping the workspace
    if not str(abs_path.resolve()).startswith(resolved_ws):
        return {"success": False, "message": "file_path attempts to escape workspace."}

    try:
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        return {"success": True, "message": f"Created {file_path}", "path": str(abs_path)}
    except Exception as e:
        return {"success": False, "message": str(e)}


def _handle_create_files(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    files: list[dict] = payload.get("files", [])

    results = []
    created = failed = 0
    for f in files:
        r = _handle_create_file({
            "workspace_path": workspace_path,
            "file_path": f.get("file_path", ""),
            "content": f.get("content", ""),
        })
        if r.get("success"):
            created += 1
        else:
            failed += 1
        results.append({"file_path": f.get("file_path", ""), "success": r.get("success", False), "message": r.get("message", "")})

    return {"success": failed == 0, "created": created, "failed": failed, "results": results}


def _handle_delete_path(payload: dict) -> dict:
    path: str = payload.get("path", "")

    resolved = _require_allowed(path)
    if resolved is None:
        return {"success": False, "detail": f"Path not in ALLOWED_WORKSPACES: {path}"}

    p = Path(resolved)
    # Prevent deleting a workspace root itself
    if any(Path(resolved) == Path(ws).resolve() for ws in ALLOWED_WORKSPACES):
        return {"success": False, "detail": "Cannot delete an entire workspace root."}

    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return {"success": True, "message": f"Deleted: {resolved}"}
    except Exception as e:
        return {"success": False, "detail": str(e)}


# --- Command execution (whitelisted) -----------------------------------------

_ALLOWED_COMMANDS = [
    "npm test", "npm run test", "yarn test", "pnpm test", "pnpm run test",
    "pytest", "python -m pytest", "dotnet test",
    "cargo test", "go test", "mvn test",
]
_ALLOWED_SCAFFOLD = [
    "npx create-next-app", "npx create-react-app", "npx create-vite",
    "npx create-remix", "npx create-astro",
    "django-admin startproject", "flask init",
    "npm init", "yarn init", "pnpm init", "pnpm create",
    "python -m venv", "pip install", "pip3 install",
    "git init", "cargo init", "go mod init",
]


def _run_command(cmd: str, cwd: str, timeout: int = 120) -> dict:
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:2000],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Timed out after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "exit_code": -1}


def _handle_run_command(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    command_line: str = payload.get("command_line", "").strip()

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "error": f"workspace_path not in ALLOWED_WORKSPACES: {workspace_path}"}

    if not any(command_line.lower().startswith(allowed) for allowed in _ALLOWED_COMMANDS):
        return {"success": False, "error": f"Command not on whitelist: {command_line}"}

    return _run_command(command_line, resolved)


def _handle_run_scaffold(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    command_line: str = payload.get("command_line", "").strip()

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "error": f"workspace_path not in ALLOWED_WORKSPACES: {workspace_path}"}

    if not any(command_line.lower().startswith(allowed) for allowed in _ALLOWED_SCAFFOLD):
        return {"success": False, "error": f"Scaffold command not on whitelist: {command_line}"}

    return _run_command(command_line, resolved, timeout=300)


# --- Patch computation (edit tools) ------------------------------------------

def _compute_patch(_workspace_path: str, file_path: str, original: str, modified: str) -> dict:
    """Generate a unified diff patch from original → modified."""
    norm = file_path.replace("\\", "/")
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{norm}",
        tofile=f"b/{norm}",
    ))
    if not diff_lines:
        return {"success": False, "message": "No differences found — search_string not in file?"}
    return {"success": True, "patch": "".join(diff_lines)}


def _read_abs(workspace_path: str, file_path: str) -> str | None:
    abs_path = Path(workspace_path) / file_path
    if not abs_path.exists():
        return None
    return abs_path.read_text(encoding="utf-8", errors="replace")


def _handle_edit_replace(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    file_path: str = payload.get("file_path", "")
    search_string: str = payload.get("search_string", "")
    replace_string: str = payload.get("replace_string", "")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": f"workspace_path not in ALLOWED_WORKSPACES"}

    original = _read_abs(resolved, file_path)
    if original is None:
        return {"success": False, "message": f"File not found: {file_path}"}

    if search_string not in original:
        return {"success": False, "message": f"search_string not found in {file_path}"}

    modified = original.replace(search_string, replace_string, 1)
    return _compute_patch(resolved, file_path, original, modified)


def _handle_edit_insert_code(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    file_path: str = payload.get("file_path", "")
    after_line: int = payload.get("after_line", 0)
    new_code: str = payload.get("new_code", "")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": "workspace_path not in ALLOWED_WORKSPACES"}

    original = _read_abs(resolved, file_path)
    if original is None:
        return {"success": False, "message": f"File not found: {file_path}"}

    lines = original.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    # Ensure new_code ends with newline
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    insert_at = min(after_line, len(lines))
    modified_lines = lines[:insert_at] + new_lines + lines[insert_at:]
    modified = "".join(modified_lines)
    return _compute_patch(resolved, file_path, original, modified)


def _handle_edit_remove_line(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    file_path: str = payload.get("file_path", "")
    line_number: int = payload.get("line_number", 0)
    line_end: int | None = payload.get("line_end")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": "workspace_path not in ALLOWED_WORKSPACES"}

    original = _read_abs(resolved, file_path)
    if original is None:
        return {"success": False, "message": f"File not found: {file_path}"}

    lines = original.splitlines(keepends=True)
    total = len(lines)
    s = line_number - 1
    e = (line_end or line_number) - 1

    if s < 0 or s >= total:
        return {"success": False, "message": f"line_number {line_number} out of range (1–{total})"}

    modified = "".join(lines[:s] + lines[e + 1:])
    return _compute_patch(resolved, file_path, original, modified)


def _handle_edit_remove_lines_matching(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    file_path: str | None = payload.get("file_path")
    pattern: str = payload.get("pattern", "")
    path_glob: str = payload.get("path_glob", "**/*")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": "workspace_path not in ALLOWED_WORKSPACES"}

    ws_path = Path(resolved)
    target_files = [ws_path / file_path] if file_path else list(ws_path.rglob(path_glob or "**/*"))
    target_files = [f for f in target_files if f.is_file()]

    all_patches: list[str] = []
    files_affected = 0
    for tf in target_files:
        try:
            original = tf.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = original.splitlines(keepends=True)
        modified_lines = [l for l in lines if pattern not in l]
        if len(modified_lines) == len(lines):
            continue
        modified = "".join(modified_lines)
        rel = str(tf.relative_to(ws_path)).replace("\\", "/")
        result = _compute_patch(resolved, rel, original, modified)
        if result.get("success"):
            all_patches.append(result["patch"])
            files_affected += 1

    if not all_patches:
        return {"success": True, "patch": "", "files_affected": 0}

    return {"success": True, "patch": "\n".join(all_patches), "files_affected": files_affected}


def _handle_edit_delete_block(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    file_path: str = payload.get("file_path", "")
    search_term: str = payload.get("search_term", "")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": "workspace_path not in ALLOWED_WORKSPACES"}

    original = _read_abs(resolved, file_path)
    if original is None:
        return {"success": False, "message": f"File not found: {file_path}"}

    lines = original.splitlines(keepends=True)
    # Find the start of the block containing search_term
    start_idx = None
    for i, line in enumerate(lines):
        if search_term in line:
            start_idx = i
            break

    if start_idx is None:
        return {"success": False, "message": f"search_term '{search_term}' not found in {file_path}"}

    # Walk backwards to include decorators (@...) and blank lines before def/class
    block_start = start_idx
    while block_start > 0 and lines[block_start - 1].startswith(("@", "    @")):
        block_start -= 1

    # Walk forward to find the end of the block (next top-level definition or EOF)
    base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    block_end = start_idx + 1
    while block_end < len(lines):
        line = lines[block_end]
        stripped = line.strip()
        if not stripped:
            block_end += 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent and stripped:
            break
        block_end += 1

    deleted_lines = "".join(lines[block_start:block_end])
    modified = "".join(lines[:block_start] + lines[block_end:])
    result = _compute_patch(resolved, file_path, original, modified)
    if result.get("success"):
        result["deleted_lines"] = deleted_lines
    return result


# --- GitHub ------------------------------------------------------------------

def _handle_github_create_repo(payload: dict) -> dict:
    name: str = payload.get("name", "")
    description: str = payload.get("description", "")
    private: bool = payload.get("private", False)

    if not GITHUB_TOKEN:
        return {"success": False, "message": "GITHUB_TOKEN is not set."}

    try:
        res = requests.post(
            "https://api.github.com/user/repos",
            headers={
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={"name": name, "description": description, "private": private},
            timeout=15,
        )
        data = res.json()
        if not res.ok:
            return {"success": False, "message": data.get("message", f"GitHub API error {res.status_code}")}
        return {
            "success": True,
            "message": f"Repository '{name}' created.",
            "html_url": data.get("html_url", ""),
            "clone_url": data.get("clone_url", ""),
            "name": data.get("name", name),
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


# --- PDF saving (markdown → PDF) ---------------------------------------------

def _handle_save_markdown_as_pdf(payload: dict) -> dict:
    workspace_path: str = payload.get("workspace_path", "")
    filename: str = payload.get("filename", "research.pdf")
    content: str = payload.get("content", "")

    resolved = _require_allowed(workspace_path)
    if resolved is None:
        return {"success": False, "message": f"workspace_path not in ALLOWED_WORKSPACES: {workspace_path}"}

    out_path = Path(resolved) / filename
    try:
        import markdown as md_lib  # type: ignore
        from xhtml2pdf import pisa  # type: ignore

        html_body = md_lib.markdown(content, extensions=["tables", "fenced_code"])
        full_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>body{{font-family:Arial,sans-serif;margin:40px;font-size:11pt;}}
pre{{background:#f4f4f4;padding:10px;border-radius:4px;overflow-x:auto;}}
code{{font-family:monospace;}}
h1,h2,h3{{color:#222;}}
table{{border-collapse:collapse;width:100%;}}
th,td{{border:1px solid #ccc;padding:6px;}}
</style></head><body>{html_body}</body></html>"""
        with open(out_path, "wb") as f:
            pisa.CreatePDF(full_html, dest=f)
        return {"success": True, "path": str(out_path), "message": f"Saved to {out_path}"}
    except Exception as e:
        return {"success": False, "message": f"PDF generation failed: {e}"}


# --- Codebase search ---------------------------------------------------------

def _handle_search(payload: dict) -> dict:
    query: str = payload.get("query", "")
    top_k: int = payload.get("top_k", 5)

    try:
        # Lazy import so the worker starts even if chromadb isn't installed
        from indexer import CodeIndexer  # type: ignore
        from config import DaemonSettings  # type: ignore

        settings = DaemonSettings()
        indexer = CodeIndexer(settings)
        results = indexer.search(query, top_k=top_k)
        return {"query": query, "results": results}
    except ImportError:
        return {"error": "Search unavailable: indexer.py not found or chromadb not installed."}
    except Exception as e:
        return {"error": str(e)}


# --- Spotify -----------------------------------------------------------------

def _get_spotify_access_token() -> str | None:
    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET and SPOTIFY_REFRESH_TOKEN):
        return None
    try:
        import base64
        creds = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
        res = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={"Authorization": f"Basic {creds}"},
            data={"grant_type": "refresh_token", "refresh_token": SPOTIFY_REFRESH_TOKEN},
            timeout=10,
        )
        return res.json().get("access_token")
    except Exception:
        return None


def _spotify_request(method: str, path: str, body: dict | None = None) -> dict:
    token = _get_spotify_access_token()
    if not token:
        return {"success": False, "error": "Spotify not configured or token refresh failed."}
    url = f"https://api.spotify.com/v1{path}"
    try:
        kwargs: dict = {"headers": {"Authorization": f"Bearer {token}"}, "timeout": 10}
        if method == "PUT" and body is not None:
            kwargs["json"] = body
        elif method == "PUT":
            kwargs["json"] = {}
        res = getattr(requests, method.lower())(url, **kwargs)
        if res.status_code in (200, 201, 204):
            try:
                data = res.json()
            except Exception:
                data = {}
            return {"success": True, **data}
        return {"success": False, "error": f"Spotify API {res.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _handle_spotify_play(_payload: dict) -> dict:
    return _spotify_request("PUT", "/me/player/play")


def _handle_spotify_pause(_payload: dict) -> dict:
    return _spotify_request("PUT", "/me/player/pause")


def _handle_spotify_next(_payload: dict) -> dict:
    return _spotify_request("POST", "/me/player/next")


def _handle_spotify_previous(_payload: dict) -> dict:
    return _spotify_request("POST", "/me/player/previous")


def _handle_spotify_volume(payload: dict) -> dict:
    volume_percent: int = int(payload.get("volume_percent", 50))
    return _spotify_request("PUT", f"/me/player/volume?volume_percent={volume_percent}")


def _handle_spotify_status(_payload: dict) -> dict:
    token = _get_spotify_access_token()
    if not token:
        return {"success": False, "error": "Spotify not configured."}
    try:
        res = requests.get(
            "https://api.spotify.com/v1/me/player",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if res.status_code == 204:
            return {"success": True, "is_playing": False, "message": "No active player."}
        data = res.json()
        item = data.get("item") or {}
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        return {
            "success": True,
            "is_playing": data.get("is_playing", False),
            "track": item.get("name", "Unknown"),
            "artists": artists,
            "device": (data.get("device") or {}).get("name", "Unknown"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _handle_spotify_close(_payload: dict) -> dict:
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/IM", "Spotify.exe", "/F"], capture_output=True)
        else:
            subprocess.run(["pkill", "-x", "Spotify"], capture_output=True)
        return {"success": True, "message": "Spotify closed."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- System control ----------------------------------------------------------

def _handle_system_status(_payload: dict) -> dict:
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "success": True,
        "cpu_percent": cpu,
        "memory_used_gb": round(mem.used / 1e9, 2),
        "memory_total_gb": round(mem.total / 1e9, 2),
        "memory_percent": mem.percent,
        "disk_used_gb": round(disk.used / 1e9, 2),
        "disk_total_gb": round(disk.total / 1e9, 2),
        "disk_percent": disk.percent,
    }


def _handle_open_url(payload: dict) -> dict:
    url: str = payload.get("url", "")
    if not url:
        return {"success": False, "message": "url is required."}
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url])
        return {"success": True, "message": f"Opened: {url}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# PowerShell template for open_app.
# Searches both Start Menu trees for a matching .lnk via a case-insensitive
# glob (-like "*$name*"), then falls back to Start-Process which resolves
# executables registered in the Windows App Paths registry (e.g. 'chrome',
# 'code', 'spotify', 'excel').
#
# Note on escaping:
#   - This is a raw Python string: every \ is a literal backslash in the PS source.
#   - {{ and }} are Python .format() escapes that become { and } in the PS script.
#   - {name} is the only Python format placeholder; $name and $(...) are PS-only.
_PS_OPEN_APP = r"""
$name = '{name}'
$dirs = @(
    "$env:ProgramData\Microsoft\Windows\Start Menu\Programs",
    "$env:APPDATA\Microsoft\Windows\Start Menu\Programs"
)
$shortcut = $dirs | ForEach-Object {{
    if (Test-Path $_) {{
        Get-ChildItem -Path $_ -Filter '*.lnk' -Recurse -ErrorAction SilentlyContinue
    }}
}} | Where-Object {{ $_.BaseName -like "*$name*" }} | Select-Object -First 1
if ($shortcut) {{
    Invoke-Item -Path $shortcut.FullName
    Write-Output "Launched via shortcut: $($shortcut.BaseName)"
}} else {{
    try {{
        Start-Process -FilePath '{name}' -ErrorAction Stop
        Write-Output "Launched via App Paths: {name}"
    }} catch {{
        Write-Error "App not found: {name}"
        exit 1
    }}
}}
"""


def _handle_open_app(payload: dict) -> dict:
    """
    Launch a Windows application by plain-English name.

    Execution order:
      1. Search both Start Menu trees ($env:ProgramData and $env:APPDATA) for a
         .lnk file whose BaseName contains app_name (case-insensitive glob).
         Launches the first match via Invoke-Item, which respects the shortcut's
         target, working directory, and run-as settings.
      2. If no shortcut is found, fall back to Start-Process <app_name>, which
         resolves executables registered under HKLM\\SOFTWARE\\Microsoft\\Windows\\
         CurrentVersion\\App Paths (e.g. 'chrome', 'code', 'msedge', 'excel').

    The PowerShell host process runs with -WindowStyle Hidden and -NonInteractive
    so no console window flashes on screen.  The launched application itself
    opens normally in its own window.

    payload keys:
      app_name (str, required) — partial or full name, e.g. "spotify", "VS Code".
    """
    if platform.system() != "Windows":
        return {"success": False, "message": "open_app is Windows-only."}

    app_name: str = payload.get("app_name", "").strip()
    if not app_name:
        return {"success": False, "message": "app_name is required."}

    # Escape single quotes: the only injection vector inside a PS single-quoted
    # string is an embedded single quote — doubled it becomes a literal quote.
    safe = app_name.replace("'", "''")
    ps_script = _PS_OPEN_APP.format(name=safe)

    try:
        result = subprocess.run(
            [
                "powershell",
                "-WindowStyle", "Hidden",
                "-NonInteractive",
                "-NoProfile",
                "-Command", ps_script,
            ],
            capture_output=True,
            text=True,
            timeout=15,   # file-system search is fast; 15 s is generous
        )
        if result.returncode == 0:
            msg = result.stdout.strip() or f"Launched: {app_name}"
            return {"success": True, "message": msg}
        err = result.stderr.strip() or result.stdout.strip() or f"PowerShell exit {result.returncode}"
        return {"success": False, "message": err[:400]}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": f"Timed out searching for: {app_name}"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def _handle_system_lock(_payload: dict) -> dict:
    try:
        if platform.system() == "Windows":
            ctypes.windll.user32.LockWorkStation()
            return {"success": True, "message": "PC locked."}
        else:
            subprocess.run(["loginctl", "lock-session"], capture_output=True)
            return {"success": True, "message": "Session locked."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _handle_system_shutdown(_payload: dict) -> dict:
    try:
        if platform.system() == "Windows":
            subprocess.run(["shutdown", "/s", "/t", "5"], capture_output=True)
        else:
            subprocess.run(["shutdown", "-h", "now"], capture_output=True)
        return {"success": True, "message": "Shutdown initiated."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _handle_system_restart(_payload: dict) -> dict:
    try:
        if platform.system() == "Windows":
            subprocess.run(["shutdown", "/r", "/t", "5"], capture_output=True)
        else:
            subprocess.run(["shutdown", "-r", "now"], capture_output=True)
        return {"success": True, "message": "Restart initiated."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _handle_system_sleep(_payload: dict) -> dict:
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["powershell", "-Command",
                 "Add-Type -Assembly System.Windows.Forms; [System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"],
                capture_output=True,
            )
        else:
            subprocess.run(["systemctl", "suspend"], capture_output=True)
        return {"success": True, "message": "Sleep initiated."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _handle_lockdown_pc(_payload: dict) -> dict:
    """
    Hardware lockdown: mute audio, turn off monitors, then lock the workstation.
    All three operations are executed silently via Win32 API / WASAPI COM — no
    visible windows or console output.

    Execution order:
      1. Mute system audio (WASAPI SetMute — idempotent, always mutes).
      2. Lock the workstation (LockWorkStation — async, posts to message queue).
      3. Brief sleep to let the lock-screen render before the display goes dark.
      4. Kill monitors (SendMessageW SC_MONITORPOWER 0xF170 / param 2 = off).
    """
    if platform.system() != "Windows":
        return {"success": False, "error": "lockdown_pc is Windows-only."}

    errors: list[str] = []

    # 1. Mute system master volume via WASAPI (pycaw → comtypes COM).
    #    SetMute(1, None) is idempotent — safe to call even when already muted.
    try:
        from ctypes import cast as ct_cast, POINTER
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        import comtypes
        speakers = AudioUtilities.GetSpeakers()
        interface = speakers.Activate(
            IAudioEndpointVolume._iid_,
            comtypes.CLSCTX_ALL,
            None,
        )
        volume_ctrl = ct_cast(interface, POINTER(IAudioEndpointVolume))
        volume_ctrl.SetMute(1, None)
    except Exception as exc:
        errors.append(f"mute failed: {exc}")

    # 2. Lock the workstation.  Returns immediately; the actual lock is async.
    try:
        ctypes.windll.user32.LockWorkStation()
    except Exception as exc:
        errors.append(f"lock failed: {exc}")

    # 3. Wait for the lock-screen to appear before cutting the display signal.
    time.sleep(0.75)

    # 4. Cut power to all connected monitors (value 2 = off; 1 = low-power/standby).
    #    HWND_BROADCAST (0xFFFF) ensures every top-level window receives the message,
    #    which in practice causes the driver to power-off all attached displays.
    try:
        HWND_BROADCAST  = 0xFFFF
        WM_SYSCOMMAND   = 0x0112
        SC_MONITORPOWER = 0xF170
        MONITOR_OFF     = 2
        ctypes.windll.user32.SendMessageW(
            HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, MONITOR_OFF
        )
    except Exception as exc:
        errors.append(f"monitor off failed: {exc}")

    if errors:
        return {
            "success": False,
            "message": "Lockdown partially applied.",
            "errors": errors,
        }
    return {"success": True, "message": "PC locked, audio muted, monitors off."}


# --- Remote task manager -----------------------------------------------------

# Process names that must never be killed — doing so can crash or blue-screen
# the OS.  psutil.AccessDenied will catch most of these at runtime anyway, but
# an explicit blocklist gives a clean, user-readable rejection message.
_PROTECTED_PROCESSES: frozenset[str] = frozenset({
    "system", "registry", "idle",
    "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe",
    "lsass.exe", "services.exe", "svchost.exe",
    "dwm.exe", "fontdrvhost.exe", "sihost.exe", "taskhostw.exe",
    "spoolsv.exe", "audiodg.exe", "ntoskrnl.exe",
})


def _get_windowed_pids() -> set[int]:
    """
    Return PIDs of processes that own at least one visible, titled desktop
    window (i.e. the kind that shows up on the taskbar).

    Filtering rules:
    - Window must be visible (IsWindowVisible).
    - Window must have a non-empty title.
    - Window must NOT be a tool-window (WS_EX_TOOLWINDOW), which covers
      system trays, floating helpers, and other off-taskbar chrome.
    """
    pids: set[int] = set()
    try:
        import win32gui
        import win32process
        import win32con

        def _callback(hwnd: int, _: object) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if not win32gui.GetWindowTextLength(hwnd):
                return True
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            if ex_style & win32con.WS_EX_TOOLWINDOW:
                return True
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                pids.add(pid)
            except Exception:
                pass
            return True

        win32gui.EnumWindows(_callback, None)
    except ImportError:
        pass  # pywin32 not installed; callers fall back to resource-only list
    return pids


def _handle_list_apps(_payload: dict) -> dict:
    """
    Return a human-readable list of user-facing processes: those with a
    visible desktop window (tag [W]) OR the top resource hogs by RAM / CPU
    (tag [R]).  Background noise (hundreds of svchost.exe, etc.) is excluded.

    Output is formatted as one line per process so it renders cleanly in
    a Telegram message.
    """
    try:
        windowed_pids = _get_windowed_pids()

        # Snapshot every process once to avoid repeated per-process syscalls.
        # cpu_percent on first call always returns 0.0 — that's acceptable here
        # since we primarily sort by RAM and use CPU as a secondary signal.
        all_procs: list[dict] = []
        for proc in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent", "status"]):
            try:
                info = proc.info
                if info["status"] == psutil.STATUS_ZOMBIE:
                    continue
                all_procs.append({
                    "pid":      info["pid"],
                    "name":     info["name"] or "?",
                    "ram":      info["memory_info"].rss if info["memory_info"] else 0,
                    "cpu":      info["cpu_percent"] or 0.0,
                    "windowed": info["pid"] in windowed_pids,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Collect PIDs from top-10 by RAM and top-10 by CPU, plus all windowed.
        top_ram_pids = {
            p["pid"]
            for p in sorted(all_procs, key=lambda x: x["ram"], reverse=True)[:10]
        }
        top_cpu_pids = {
            p["pid"]
            for p in sorted(all_procs, key=lambda x: x["cpu"], reverse=True)[:10]
        }
        interesting = windowed_pids | top_ram_pids | top_cpu_pids

        selected = [p for p in all_procs if p["pid"] in interesting]
        # Sort: windowed apps first, then by RAM descending within each group.
        selected.sort(key=lambda x: (not x["windowed"], -x["ram"]))

        if not selected:
            return {"success": True, "message": "No foreground applications found.", "apps": []}

        lines: list[str] = []
        for p in selected:
            tag    = "[W]" if p["windowed"] else "[R]"
            ram_gb = p["ram"] / 1_073_741_824
            lines.append(
                f"{tag} PID: {p['pid']:<6} | {p['name']:<28} | "
                f"RAM: {ram_gb:.2f} GB | CPU: {p['cpu']:.1f}%"
            )

        return {
            "success": True,
            "message": "\n".join(lines),
            "apps": [{"pid": p["pid"], "name": p["name"]} for p in selected],
        }

    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _handle_kill_app(payload: dict) -> dict:
    """
    Forcefully terminate a process by PID or by name.

    payload keys:
      pid          (int, optional)  — target a single process by PID
      process_name (str, optional)  — kill all processes with this exact name

    Protection layers:
      1. Explicit blocklist (_PROTECTED_PROCESSES) — returns a clear rejection.
      2. PID 0 (Idle) and PID 4 (System) guard — kernel pseudo-processes.
      3. psutil.AccessDenied — elevated system processes refuse the kill call.
      4. psutil.NoSuchProcess — process already exited between listing and kill.
    All errors are caught individually so one failure never crashes the worker.
    """
    pid: int | None = payload.get("pid")
    name: str | None = payload.get("process_name")

    if pid is None and not name:
        return {"success": False, "error": "Provide 'pid' (int) or 'process_name' (str)."}

    # --- Resolve the target process list ------------------------------------
    targets: list[psutil.Process] = []
    try:
        if pid is not None:
            targets = [psutil.Process(int(pid))]
        else:
            assert name is not None
            name_lower = name.lower()
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if (proc.info["name"] or "").lower() == name_lower:
                        targets.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    except psutil.NoSuchProcess:
        return {"success": False, "error": f"No process found with PID {pid}."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    if not targets:
        return {"success": False, "error": f"No running process named '{name}'."}

    # --- Kill each target, collecting per-process results -------------------
    lines: list[str] = []
    killed = failed = 0

    for proc in targets:
        proc_pid = proc.pid
        try:
            proc_name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_name = "?"

        # Guard: kernel pseudo-processes
        if proc_pid in (0, 4):
            lines.append(f"BLOCKED  PID {proc_pid} ({proc_name}): kernel process.")
            failed += 1
            continue

        # Guard: explicit name blocklist
        if proc_name.lower() in _PROTECTED_PROCESSES:
            lines.append(f"BLOCKED  PID {proc_pid} ({proc_name}): protected system process.")
            failed += 1
            continue

        try:
            proc.kill()  # TerminateProcess on Windows — immediate, no cleanup window
            lines.append(f"KILLED   PID {proc_pid} ({proc_name})")
            killed += 1
        except psutil.NoSuchProcess:
            lines.append(f"SKIPPED  PID {proc_pid} ({proc_name}): already exited.")
            failed += 1
        except psutil.AccessDenied:
            lines.append(f"DENIED   PID {proc_pid} ({proc_name}): insufficient privileges.")
            failed += 1
        except Exception as exc:
            lines.append(f"ERROR    PID {proc_pid} ({proc_name}): {exc}")
            failed += 1

    summary = f"Killed: {killed}  |  Blocked/Failed: {failed}"
    return {
        "success": killed > 0 and failed == 0,
        "message": summary + "\n\n" + "\n".join(lines),
        "killed": killed,
        "failed": failed,
    }


# --- Media & display control -------------------------------------------------

def _handle_adjust_volume(payload: dict) -> dict:
    """
    Set or nudge the system master volume.

    payload keys (mutually exclusive — absolute takes priority):
      absolute_level (int, 0–100) — set volume to an exact percentage.
      step           (str)        — relative adjustment, e.g. "+10" or "-20".

    Strategy:
    - absolute_level  → pycaw IAudioEndpointVolume.SetMasterVolumeLevelScalar()
      Precise, idempotent, no simulated keypresses.
    - step            → pycaw read current level, clamp, then SetMasterVolumeLevelScalar().
      Avoids the VK_VOLUME_UP/DOWN toggle approach, which sends fixed OS-defined
      increments (~2% per key) and cannot target an exact level.
    """
    if platform.system() != "Windows":
        return {"success": False, "error": "adjust_volume is Windows-only."}

    absolute: int | None = payload.get("absolute_level")
    step_raw: str | None = payload.get("step")

    if absolute is None and not step_raw:
        return {"success": False, "error": "Provide 'absolute_level' (0–100) or 'step' (e.g. '+10')."}

    try:
        from ctypes import cast as ct_cast, POINTER
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        import comtypes

        speakers  = AudioUtilities.GetSpeakers()
        interface = speakers.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
        vol_ctrl  = ct_cast(interface, POINTER(IAudioEndpointVolume))

        if absolute is not None:
            target = max(0, min(100, int(absolute)))
        else:
            # Parse step string: "+10", "-20", "10", etc.
            step_str = str(step_raw).strip().replace(" ", "")
            try:
                step_val = int(step_str)
            except ValueError:
                return {"success": False, "error": f"Cannot parse step value: '{step_raw}'"}
            current_scalar = vol_ctrl.GetMasterVolumeLevelScalar()
            current_pct    = round(current_scalar * 100)
            target         = max(0, min(100, current_pct + step_val))

        vol_ctrl.SetMasterVolumeLevelScalar(target / 100.0, None)
        # Unmute if we're setting a non-zero level (muted volume stays muted otherwise).
        if target > 0:
            vol_ctrl.SetMute(0, None)

        return {"success": True, "message": f"Volume set to {target}%."}

    except ImportError:
        return {"success": False, "error": "pycaw is not installed. Run: pip install pycaw"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _handle_adjust_brightness(payload: dict) -> dict:
    """
    Set or nudge display brightness on all monitors.

    payload keys (mutually exclusive — absolute takes priority):
      absolute_level (int, 0–100) — set to an exact percentage.
      step           (str)        — relative adjustment, e.g. "+10" or "-10".

    Uses screen-brightness-control (sbc), which automatically selects the
    right backend per monitor:
      - Laptop / integrated panels → WMI (always works on Windows)
      - External monitors          → DDC/CI via the monitor's I²C bus

    DDC/CI failures (external monitors that don't support software control)
    are caught per-monitor so one unsupported display never blocks the others.
    """
    absolute: int | None = payload.get("absolute_level")
    step_raw: str | None = payload.get("step")

    if absolute is None and not step_raw:
        return {"success": False, "error": "Provide 'absolute_level' (0–100) or 'step' (e.g. '+10')."}

    try:
        import screen_brightness_control as sbc  # type: ignore
    except ImportError:
        return {
            "success": False,
            "error": "screen-brightness-control is not installed. Run: pip install screen-brightness-control",
        }

    # Parse step once (used in the per-monitor relative path).
    step_val: int | None = None
    if absolute is None:
        try:
            step_val = int(str(step_raw).strip().replace(" ", ""))
        except ValueError:
            return {"success": False, "error": f"Cannot parse step value: '{step_raw}'"}

    try:
        monitors = sbc.list_monitors()
    except Exception as exc:
        return {"success": False, "error": f"Could not enumerate monitors: {exc}"}

    if not monitors:
        return {"success": False, "error": "No controllable monitors found."}

    results: list[str] = []
    any_ok = False

    for mon in monitors:
        try:
            if absolute is not None:
                target = max(0, min(100, int(absolute)))
            else:
                assert step_val is not None
                current_list = sbc.get_brightness(display=mon)
                current      = current_list[0] if current_list else 50
                target       = max(0, min(100, current + step_val))

            sbc.set_brightness(target, display=mon)
            results.append(f"OK       {mon}: brightness → {target}%")
            any_ok = True

        except sbc.exceptions.ScreenBrightnessError as exc:
            # DDC/CI not supported on this monitor — surface a friendly message.
            results.append(f"SKIPPED  {mon}: monitor does not support software brightness control ({exc})")
        except Exception as exc:
            results.append(f"ERROR    {mon}: {exc}")

    return {
        "success": any_ok,
        "message": "\n".join(results),
    }


# --- Claude Code -------------------------------------------------------------

# Fallback path if `claude` is not on PATH (Windows installation default).
_CLAUDE_FALLBACK_PATH = r"C:\Users\madus\.local\bin\claude.exe"

# Standard coding tools that cover the full build/edit/read workflow.
# The worker caller may override this via the `allowed_tools` payload key.
_CLAUDE_DEFAULT_TOOLS = "Bash,Write,Edit,Read,Glob,Grep,LS,MultiEdit"


def _handle_run_claude_code(payload: dict) -> dict:
    """
    Invoke the Claude Code CLI in non-interactive (-p) mode.

    payload keys:
      prompt             (str, required)  — task instructions for Claude Code.
      working_directory  (str, optional)  — CWD for the subprocess; must be
                                            within ALLOWED_WORKSPACES. Defaults
                                            to the first configured workspace.
      timeout            (int, optional)  — max wall-clock seconds (default 600).
      allowed_tools      (str, optional)  — comma-separated override for the
                                            --allowedTools flag.

    Claude Code is run as:
      claude -p <prompt>
             --allowedTools <tools>
             --no-session-persistence

    The last 30 lines of stdout are returned as the Telegram message so the
    user gets a concise status report without being flooded by full build logs.
    """
    prompt: str = payload.get("prompt", "").strip()
    if not prompt:
        return {"success": False, "error": "payload.prompt is required."}

    working_directory: str = payload.get("working_directory", "")
    timeout: int = int(payload.get("timeout", 600))
    allowed_tools: str = payload.get("allowed_tools", _CLAUDE_DEFAULT_TOOLS)

    # Resolve the working directory — must be within ALLOWED_WORKSPACES.
    if working_directory:
        resolved = _require_allowed(working_directory)
        if resolved is None:
            return {
                "success": False,
                "error": f"working_directory not in ALLOWED_WORKSPACES: {working_directory}",
            }
    else:
        # No directory specified — fall back to the first configured workspace.
        if not ALLOWED_WORKSPACES:
            return {"success": False, "error": "No ALLOWED_WORKSPACES configured."}
        resolved = str(Path(ALLOWED_WORKSPACES[0]).resolve())

    os.makedirs(resolved, exist_ok=True)

    # Locate the claude executable; fall back to the known installation path.
    claude_exe = shutil.which("claude") or _CLAUDE_FALLBACK_PATH
    if not Path(claude_exe).exists():
        return {
            "success": False,
            "error": f"Claude Code executable not found at '{claude_exe}'. "
                     "Ensure it is installed and on PATH.",
        }

    cmd = [
        claude_exe,
        "-p", prompt,
        "--allowedTools", allowed_tools,
        "--no-session-persistence",   # keep worker invocations isolated
    ]

    log.info(
        "[claude] Starting: cwd=%s tools=%s timeout=%ds prompt=%.120s",
        resolved, allowed_tools, timeout, prompt,
    )

    try:
        result = subprocess.run(
            cmd,
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},  # inherit PATH, ANTHROPIC_API_KEY, etc.
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Claude Code timed out after {timeout}s.",
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "stdout": "", "stderr": "", "exit_code": -1}

    # Trim to last 30 lines for the Telegram message — build logs are verbose.
    stdout_lines = result.stdout.splitlines()
    tail = "\n".join(stdout_lines[-30:]) if stdout_lines else "(no output)"

    log.info("[claude] Finished: exit=%d lines=%d", result.returncode, len(stdout_lines))

    return {
        "success": result.returncode == 0,
        "message": tail,
        "stdout": result.stdout[:6000],
        "stderr": result.stderr[:2000],
        "exit_code": result.returncode,
    }


# ---------------------------------------------------------------------------
# Streaming Claude Code handler (called directly from dispatch)
# ---------------------------------------------------------------------------

def _handle_run_claude_code_streaming(task: dict) -> None:
    """
    Long-running variant of run_claude_code that uses subprocess.Popen so
    Claude Code's stdout can be streamed back to the user in real-time without
    waiting up to 20 minutes for the process to finish.

    Output is buffered for _CLAUDE_STREAM_INTERVAL seconds and then flushed
    as a Markdown code block directly to Telegram.

    This function manages its own result routing; dispatch() must return
    immediately after calling it.
    """
    payload:      dict       = task.get("payload", {})
    sender_id:    str | None = task.get("sender_id")
    bot_token:    str        = task.get("bot_token", BOT_TOKEN) or BOT_TOKEN
    if not sender_id:
        log.warning("[claude-stream] task has no sender_id — cannot stream output.")
        return

    def _send(text: str) -> None:
        _post_worker_callback(text, sender_id, bot_token)

    # ------------------------------------------------------------------
    # 1. Validate prompt
    # ------------------------------------------------------------------
    prompt: str = payload.get("prompt", "").strip()
    if not prompt:
        _send("❌ run_claude_code: payload.prompt is required.")
        return

    # ------------------------------------------------------------------
    # 2. Resolve working directory (must be within ALLOWED_WORKSPACES)
    # ------------------------------------------------------------------
    working_directory: str = payload.get("working_directory", "")
    timeout: int           = int(payload.get("timeout", 1200))  # 20-min default
    allowed_tools: str     = payload.get("allowed_tools", _CLAUDE_DEFAULT_TOOLS)

    if working_directory:
        resolved = _require_allowed(working_directory)
        if resolved is None:
            _send(f"❌ working_directory not in ALLOWED_WORKSPACES: {working_directory}")
            return
    else:
        if not ALLOWED_WORKSPACES:
            _send("❌ No ALLOWED_WORKSPACES configured.")
            return
        resolved = str(Path(ALLOWED_WORKSPACES[0]).resolve())

    os.makedirs(resolved, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Locate the Claude CLI executable
    # ------------------------------------------------------------------
    claude_exe = shutil.which("claude") or _CLAUDE_FALLBACK_PATH
    if not Path(claude_exe).exists():
        _send(
            f"❌ Claude Code executable not found at `{claude_exe}`.\n"
            "Ensure it is installed and available on PATH."
        )
        return

    cmd = [
        claude_exe,
        "-p", prompt,
        "--allowedTools", allowed_tools,
        "--no-session-persistence",
    ]

    log.info(
        "[claude-stream] Starting: cwd=%s tools=%s timeout=%ds prompt=%.120s",
        resolved, allowed_tools, timeout, prompt,
    )
    _send(f"🤖 *Claude Code started* (task `{task_id}`)\n`cwd: {resolved}`")

    # ------------------------------------------------------------------
    # 4. Launch subprocess with piped stdout
    # ------------------------------------------------------------------
    try:
        process = subprocess.Popen(
            cmd,
            cwd=resolved,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr so nothing is silently lost
            text=True,
            encoding="utf-8",
            errors="replace",           # never crash on garbled terminal escapes
            env={**os.environ},         # inherit PATH, ANTHROPIC_API_KEY, etc.
        )
    except Exception as exc:
        _send(f"❌ Failed to launch Claude Code: {exc}")
        return

    # ------------------------------------------------------------------
    # 5. Stream output with 5-second batching
    # ------------------------------------------------------------------
    deadline: float = time.time() + timeout
    # Use a dict so the nested helper can mutate state without nonlocal.
    # Telegram message limit is 4096 chars; code-block delimiters take ~7.
    _CHUNK_LIMIT = 3800
    state = {"buffer": "", "last_flush": time.time()}

    def _flush_buffer() -> None:
        content = state["buffer"].rstrip()
        state["buffer"] = ""
        state["last_flush"] = time.time()
        if not content:
            return
        # Split oversized output across multiple messages.
        for offset in range(0, max(1, len(content)), _CHUNK_LIMIT):
            piece = content[offset : offset + _CHUNK_LIMIT]
            _send(f"```\n{piece}\n```")

    try:
        while True:
            # Hard deadline — kill the process rather than block indefinitely.
            if time.time() > deadline:
                log.warning("[claude-stream] Timeout (%ds) exceeded — killing process.", timeout)
                process.kill()
                _flush_buffer()
                _send(f"⏱ *Claude Code timed out* after {timeout}s and was terminated.")
                return

            line: str = process.stdout.readline()  # type: ignore[union-attr]

            # readline() returns '' only at EOF after the process has exited.
            if line == "" and process.poll() is not None:
                break

            if line:
                state["buffer"] += line

            # Flush every _CLAUDE_STREAM_INTERVAL seconds.
            if state["buffer"] and (time.time() - state["last_flush"]) >= _CLAUDE_STREAM_INTERVAL:
                _flush_buffer()

    except Exception as exc:
        log.exception("[claude-stream] Error reading subprocess stdout.")
        state["buffer"] += f"\n[stream read error: {exc}]"

    # ------------------------------------------------------------------
    # 6. Final flush and completion notice
    # ------------------------------------------------------------------
    _flush_buffer()

    exit_code: int = process.returncode if process.returncode is not None else -1
    log.info("[claude-stream] Finished: exit=%d", exit_code)

    if exit_code == 0:
        _send("✅ *Claude Code — Task Complete* (exit 0)")
    else:
        _send(f"❌ *Claude Code — Task Failed* (exit {exit_code})")


# ---------------------------------------------------------------------------
# File discovery + Telegram file delivery
# ---------------------------------------------------------------------------

def _handle_find_file(payload: dict) -> dict:
    """Search for files by name pattern and/or recency across ALLOWED_WORKSPACES."""
    name_pattern: str = payload.get("name_pattern", "").lower()
    folder_path: str = payload.get("folder_path", "")
    modified_within_days: float = payload.get("modified_within_days", 0)
    max_results: int = min(payload.get("max_results", 20), 100)

    if folder_path:
        resolved = _require_allowed(folder_path)
        if resolved is None:
            return {"success": False, "error": f"Path not in ALLOWED_WORKSPACES: {folder_path}"}
        search_roots = [Path(resolved)]
    else:
        search_roots = [Path(ws) for ws in ALLOWED_WORKSPACES if Path(ws).exists()]

    cutoff: float = 0.0
    if modified_within_days and modified_within_days > 0:
        cutoff = time.time() - modified_within_days * 86400

    results = []
    for root in search_roots:
        try:
            for f in root.rglob("*"):
                if not f.is_file():
                    continue
                if name_pattern and name_pattern not in f.name.lower():
                    continue
                stat = f.stat()
                if cutoff and stat.st_mtime < cutoff:
                    continue
                results.append({
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": stat.st_size,
                    "modified_ts": stat.st_mtime,
                })
        except PermissionError:
            pass

    results.sort(key=lambda x: x["modified_ts"], reverse=True)
    results = results[:max_results]
    return {"success": True, "count": len(results), "files": results}


def _handle_send_file_to_telegram(payload: dict) -> dict:
    """Read a local file and upload it to a Telegram chat via sendDocument / sendPhoto."""
    file_path: str = payload.get("file_path", "")
    chat_id: str = str(payload.get("chat_id", ""))
    bot_token: str = payload.get("bot_token", "")
    caption: str = payload.get("caption", "")

    if not file_path or not chat_id or not bot_token:
        return {"success": False, "error": "file_path, chat_id, and bot_token are required"}

    resolved = _require_allowed(file_path)
    if resolved is None:
        return {"success": False, "error": f"Path not in ALLOWED_WORKSPACES: {file_path}"}

    path = Path(resolved)
    if not path.is_file():
        return {"success": False, "error": f"File not found: {resolved}"}

    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    is_image = path.suffix.lower() in image_exts
    method = "sendPhoto" if is_image else "sendDocument"
    field = "photo" if is_image else "document"
    url = f"https://api.telegram.org/bot{bot_token}/{method}"

    try:
        with open(resolved, "rb") as fh:
            data = {"chat_id": chat_id}
            if caption:
                data["caption"] = caption
            resp = requests.post(url, files={field: (path.name, fh)}, data=data, timeout=60)
        result = resp.json()
        if result.get("ok"):
            return {"success": True, "message": f"Sent: {path.name}"}
        return {"success": False, "error": result.get("description", "Telegram API error")}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Remote Sentinel — constants
# ---------------------------------------------------------------------------

MAX_TG_BYTES = 49 * 1024 * 1024  # 49 MB hard limit (Telegram cap is 50 MB)

MEDIA_DOMAINS = {
    "youtube.com", "youtu.be", "twitter.com", "x.com",
    "vimeo.com", "twitch.tv", "tiktok.com", "instagram.com",
}
DOWNLOAD_DIR = Path(r"D:\Downloads\Eureka_Media")


# ---------------------------------------------------------------------------
# capture_webcam
# ---------------------------------------------------------------------------

def _handle_capture_webcam(payload: dict) -> dict:
    """Snap a single frame from the default webcam and send it to Telegram."""
    chat_id = str(payload.get("chat_id", ""))
    bot_token = payload.get("bot_token", BOT_TOKEN) or BOT_TOKEN

    if not chat_id or not bot_token:
        return {"success": False, "error": "chat_id and bot_token are required."}

    tmp_path = Path(__file__).parent / "sentinel_capture.jpg"

    cap = cv2.VideoCapture(0)
    try:
        # Discard the first few frames — cameras need a moment to auto-expose.
        ok, frame = False, None
        for _ in range(5):
            ok, frame = cap.read()
    finally:
        cap.release()  # always free the hardware lock so the green LED turns off

    if not ok or frame is None:
        return {"success": False, "error": "Webcam capture failed — no frame returned."}

    written = cv2.imwrite(str(tmp_path), frame)
    if not written or not tmp_path.exists() or tmp_path.stat().st_size == 0:
        return {"success": False, "error": "Webcam capture failed — could not write image file."}

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(tmp_path, "rb") as fh:
            resp = requests.post(
                url,
                files={"photo": ("sentinel.jpg", fh)},
                data={"chat_id": chat_id},
                timeout=30,
            )
        result = resp.json()
        if not result.get("ok"):
            return {"success": False, "error": result.get("description", "Telegram API error")}
        return {"success": True, "message": "Webcam snapshot sent."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# rescue_file
# ---------------------------------------------------------------------------

def _handle_rescue_file(payload: dict) -> dict:
    """Find a file by name under %USERPROFILE% and upload it to Telegram."""
    file_name = payload.get("file_name", "").strip()
    chat_id = str(payload.get("chat_id", ""))
    bot_token = payload.get("bot_token", BOT_TOKEN) or BOT_TOKEN

    if not file_name or not chat_id:
        return {"success": False, "error": "file_name and chat_id are required."}

    userprofile = Path(os.environ.get("USERPROFILE", str(Path.home())))
    matches = list(userprofile.rglob(file_name))

    if not matches:
        return {"success": False, "error": f"No file named '{file_name}' found under {userprofile}."}

    # Multiple matches → pick most recently modified; notify user of all candidates
    if len(matches) > 1:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        paths_list = "\n".join(str(p) for p in matches[:10])
        _tg(
            chat_id,
            f"Found {len(matches)} matches. Sending the most recent one:\n{matches[0]}\n\nAll candidates:\n{paths_list}",
            bot_token,
        )

    target = matches[0]
    size = os.path.getsize(target)
    if size > MAX_TG_BYTES:
        return {
            "success": False,
            "error": f"File too large ({size / 1024 / 1024:.1f} MB). Telegram's limit is 50 MB.",
        }

    tg_url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    try:
        with open(target, "rb") as fh:
            resp = requests.post(
                tg_url,
                files={"document": (target.name, fh)},
                data={"chat_id": chat_id, "caption": str(target)},
                timeout=60,
            )
        result = resp.json()
        if not result.get("ok"):
            return {"success": False, "error": result.get("description", "Telegram API error")}
        return {"success": True, "message": f"Sent: {target}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# remote_download (background thread — does not block the BLPOP loop)
# ---------------------------------------------------------------------------

def _download_worker(url: str, chat_id: str, bot_token: str) -> None:
    """Background thread: route URL to yt-dlp, requests-stream, or aria2c."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if url.startswith("magnet:"):
            subprocess.run(
                ["aria2c", "--dir", str(DOWNLOAD_DIR), url],
                check=True,
                timeout=3600,
            )
            _tg(chat_id, f"Download complete (aria2c): {url[:80]}", bot_token)
            return

        host = urlparse(url).netloc.lower().removeprefix("www.")
        if any(d in host for d in MEDIA_DOMAINS):
            ydl_opts = {
                "format": "bestvideo+bestaudio/best",
                "outtmpl": str(DOWNLOAD_DIR / "%(title)s.%(ext)s"),
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", url) if info else url
            _tg(chat_id, f"Download complete: {title}", bot_token)
            return

        # Standard file URL — streaming chunk download
        fname = Path(url.split("?")[0]).name or "download"
        dest = DOWNLOAD_DIR / fname
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        _tg(chat_id, f"Download complete: {dest}", bot_token)

    except Exception as exc:
        _tg(chat_id, f"Download failed: {exc}", bot_token)


def _handle_remote_download(payload: dict) -> dict:
    """Kick off a background download and return immediately."""
    url = payload.get("url", "").strip()
    chat_id = str(payload.get("chat_id", ""))
    bot_token = payload.get("bot_token", BOT_TOKEN) or BOT_TOKEN

    if not url or not chat_id:
        return {"success": False, "error": "url and chat_id are required."}

    threading.Thread(
        target=_download_worker,
        args=(url, chat_id, bot_token),
        daemon=True,
    ).start()
    return {"success": True, "message": f"Download started: {url[:80]}"}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Any] = {
    # Build
    "execute_build_step": _handle_execute_build_step,
    # Git / patch
    "apply_patch": _handle_apply_patch,
    "git_commit_and_push": _handle_git_commit_and_push,
    "git_push_only": _handle_git_push_only,
    "git_push": _handle_git_push,
    "git_commit_all": _handle_git_commit_all,
    "git_clone": _handle_git_clone,
    "uncommitted_diff": _handle_uncommitted_diff,
    "list_git_repos": _handle_list_git_repos,
    # GitHub
    "github_create_repo": _handle_github_create_repo,
    # File operations
    "list_folders": _handle_list_folders,
    "list_folder_contents": _handle_list_folder_contents,
    "find_file": _handle_find_file,
    "send_file_to_telegram": _handle_send_file_to_telegram,
    "read_file": _handle_read_file,
    "create_file": _handle_create_file,
    "create_files": _handle_create_files,
    "delete_path": _handle_delete_path,
    # Commands
    "run_command": _handle_run_command,
    "run_scaffold": _handle_run_scaffold,
    # Patch computation
    "edit_replace": _handle_edit_replace,
    "edit_insert_code": _handle_edit_insert_code,
    "edit_remove_line": _handle_edit_remove_line,
    "edit_remove_lines_matching": _handle_edit_remove_lines_matching,
    "edit_delete_block": _handle_edit_delete_block,
    # PDF
    "save_markdown_as_pdf": _handle_save_markdown_as_pdf,
    # Search (RAG)
    "search": _handle_search,
    # Spotify
    "spotify_play": _handle_spotify_play,
    "spotify_pause": _handle_spotify_pause,
    "spotify_next": _handle_spotify_next,
    "spotify_previous": _handle_spotify_previous,
    "spotify_volume": _handle_spotify_volume,
    "spotify_status": _handle_spotify_status,
    "spotify_close": _handle_spotify_close,
    # System
    "open_url": _handle_open_url,
    "open_app": _handle_open_app,
    "system_status": _handle_system_status,
    "system_lock": _handle_system_lock,
    "system_shutdown": _handle_system_shutdown,
    "system_restart": _handle_system_restart,
    "system_sleep": _handle_system_sleep,
    "lockdown_pc": _handle_lockdown_pc,
    # Task manager
    "list_apps": _handle_list_apps,
    "kill_app": _handle_kill_app,
    # Media & display
    "adjust_volume": _handle_adjust_volume,
    "adjust_brightness": _handle_adjust_brightness,
    # Claude Code
    "run_claude_code": _handle_run_claude_code,
    # Remote Sentinel
    "capture_webcam": _handle_capture_webcam,
    "rescue_file": _handle_rescue_file,
    "remote_download": _handle_remote_download,
}

# ---------------------------------------------------------------------------
# Task dispatcher
# ---------------------------------------------------------------------------

def dispatch(task: dict, r: redis_lib.Redis) -> None:  # type: ignore[type-arg]
    """Execute the action in `task` and route the result."""
    action: str = task.get("action", "")
    payload: dict = task.get("payload", {})
    reply_to: str | None = task.get("reply_to")       # sync: Next.js polls this
    sender_id: str | None = task.get("sender_id")     # async: worker sends Telegram
    bot_token: str | None = task.get("bot_token", BOT_TOKEN) or BOT_TOKEN
    task_id: str = task.get("task_id", "?")

    log.info("[task] action=%s task_id=%s", action, task_id)

    handler = HANDLERS.get(action)
    if handler is None:
        result: dict = {"error": f"Unknown action: {action}"}
    elif action == "run_claude_code":
        # Streaming variant: launches Popen, buffers output, POSTs to the
        # orchestrator every 5 s, and sends a final status message.
        # It manages its own result routing, so we return immediately.
        _handle_run_claude_code_streaming(task)
        return
    else:
        try:
            result = handler(payload)
        except Exception as exc:
            log.exception("[task] Handler raised for action=%s", action)
            result = {"error": str(exc)}

    # Route result
    if reply_to:
        # Sync task — publish result for Next.js to pick up
        try:
            r.set(reply_to, json.dumps(result), ex=120)
        except Exception as e:
            log.error("[task] Failed to publish result to %s: %s", reply_to, e)
    elif sender_id:
        # Async task — notify user directly via Telegram
        if result.get("success") is False or result.get("error"):
            text = result.get("message") or result.get("error") or "Task failed."
        else:
            text = result.get("message") or f"Task '{action}' completed."
        _tg(sender_id, text, bot_token)
    else:
        log.warning("[task] Task has no reply_to or sender_id — result dropped (action=%s)", action)


# ---------------------------------------------------------------------------
# Main loop with reconnection
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Eureka host worker starting (PID %d).", os.getpid())
    log.info("Allowed workspaces: %s", ALLOWED_WORKSPACES)
    log.info("Listening on queue: %s", TASK_QUEUE)

    while True:
        try:
            r: redis_lib.Redis = redis_lib.Redis.from_url(  # type: ignore[type-arg]
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=35,
            )
            r.ping()
            log.info("Connected to Upstash Redis. Waiting for tasks...")

            while True:
                item = r.blpop(TASK_QUEUE, timeout=30)
                if item is None:
                    # BLPOP timeout — just loop again (keeps connection alive)
                    continue
                _, raw = item
                try:
                    task = json.loads(raw)
                except json.JSONDecodeError as e:
                    log.error("Invalid JSON in queue item: %s — %s", raw[:200], e)
                    continue
                dispatch(task, r)

        except redis_lib.exceptions.ConnectionError as e:
            log.warning("Redis connection error: %s. Reconnecting in 5 s...", e)
            time.sleep(5)
        except redis_lib.exceptions.AuthenticationError as e:
            log.error("Redis authentication failed: %s. Check UPSTASH_REDIS_URL.", e)
            time.sleep(30)
        except KeyboardInterrupt:
            log.info("Shutting down.")
            break
        except Exception as e:
            log.exception("Unexpected error in main loop: %s. Restarting in 5 s...", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
