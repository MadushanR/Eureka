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
import time
from pathlib import Path
from typing import Any

import psutil
import redis as redis_lib
import requests
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
    if not ok and "nothing to commit" not in err.lower():
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
                "type": "directory" if entry.is_dir() else "file",
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

def _compute_patch(workspace_path: str, file_path: str, original: str, modified: str) -> dict:
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
    "system_status": _handle_system_status,
    "system_lock": _handle_system_lock,
    "system_shutdown": _handle_system_shutdown,
    "system_restart": _handle_system_restart,
    "system_sleep": _handle_system_sleep,
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
