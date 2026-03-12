# Eureka Host Worker

A standalone, headless Python worker that runs locally on your machine. It connects to Upstash Redis, uses `BLPOP` to pull tasks pushed by the Vercel orchestrator, executes them locally (file ops, git, code search, Spotify, GUI automation, etc.), and publishes results back to Redis (or sends them directly via Telegram for fire-and-forget tasks).

No HTTP server or open port is required.

## Stack

| Layer | Library |
|---|---|
| Redis client | `redis` (raw TCP, `rediss://`) |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| GUI automation | `pyautogui`, `pygetwindow` |
| Browser | `playwright` (Chromium, lazy-init) |
| Screen capture | `opencv-python` (`cv2`) |
| Config | `python-dotenv` + `.env` |

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — at minimum set UPSTASH_REDIS_URL, TELEGRAM_BOT_TOKEN, ALLOWED_WORKSPACES

# 4. Start the worker
python host_worker.py
```

On Windows you can run it invisibly at startup using the included launcher:

```
start_invisible.vbs   → runs start_eureka.bat hidden, logs to logs\worker.log
```

## Configuration Reference (`rag-daemon/.env`)

| Variable | Required | Description |
|---|---|---|
| `UPSTASH_REDIS_URL` | ✓ | Raw TCP URL: `rediss://:{token}@host.upstash.io:6379` |
| `TELEGRAM_BOT_TOKEN` | ✓ | Same token as orchestrator — used for async Telegram notifications |
| `ALLOWED_WORKSPACES` | ✓ | JSON array of absolute paths the worker may read/write |
| `OPENAI_API_KEY` | — | Injected into `aider` subprocesses for DevMode build steps |
| `GITHUB_TOKEN` | — | Personal Access Token with `repo` scope — enables GitHub repo creation |
| `SPOTIFY_CLIENT_ID` | — | Spotify app client ID |
| `SPOTIFY_CLIENT_SECRET` | — | Spotify app client secret |
| `SPOTIFY_REFRESH_TOKEN` | — | Obtained via `python get_spotify_refresh_token.py` |
| `UNLOCK_PIN` | — | Windows PIN/password for `system_unlock` |

## Task Queue Protocol

The orchestrator pushes JSON tasks to the `eureka:host_commands` Redis list. The worker `BLPOP`s them and routes by the `action` field.

**Request/response** tasks include a `reply_to` key. The worker publishes its JSON result to `eureka:result:{taskId}`; the orchestrator polls for it.

**Fire-and-forget** tasks include `sender_id` and `bot_token`. The worker sends the result directly to the user via Telegram.

## Supported Actions

| Action | Description |
|---|---|
| `search` | Semantic code search (ChromaDB + sentence-transformers) |
| `index_now` | Re-index all allowed workspaces |
| `read_file` | Read file with optional line range |
| `list_folders` | Top-level folders in allowed workspaces |
| `list_folder_contents` | Files and folders inside a given directory |
| `list_git_repos` | Discover git repos under allowed workspaces |
| `uncommitted_diff` | `git diff` + `git status` for a repo |
| `create_file` | Create or overwrite a single file |
| `create_files` | Create multiple files atomically |
| `edit_delete_block` | Delete lines matching a search term |
| `edit_insert_code` | Insert code after a given line number |
| `edit_replace` | Replace a string in a file |
| `edit_remove_line` | Remove a specific line |
| `edit_remove_lines_matching` | Remove all lines matching a pattern |
| `delete_path` | Delete a file or directory |
| `find_file` | Search for files by name pattern |
| `run_command` | Run a shell command in a workspace |
| `run_scaffold` | Long-running scaffold command (e.g. `create-react-app`) |
| `apply_patch` | Validate and apply a unified diff (`git apply --check` first) |
| `git_push` | Stage, commit, and push to remote |
| `github_create_repo` | Create a GitHub repo via API |
| `git_clone` | Clone a repo into a workspace |
| `send_file_to_telegram` | Upload a local file to Telegram |
| `rescue_file` | Read and send an oversized file in chunks |
| `capture_webcam` | Take a webcam photo and send via Telegram |
| `take_screenshot` | Screenshot the desktop |
| `remote_download` | Download a URL to a local path |
| `browser` | Playwright browser automation (navigate, click, type, extract, etc.) |
| `gui_action` | PyAutoGUI mouse/keyboard actions |
| `get_windows` | List open windows |
| `focus_window` | Focus a window by title |
| `spotify_*` | Spotify playback control (play, pause, next, prev, volume, etc.) |
| `system_unlock` | Auto-type PIN/password on Windows lock screen |
| `run_claude_code` | Run Claude Code CLI in a workspace |

## Project Structure

```
rag-daemon/
├── .env.example          ← Copy to .env and fill in your values
├── requirements.txt      ← pip dependencies
├── host_worker.py        ← Main worker (Redis queue consumer)
├── config.py             ← pydantic-settings config (ALLOWED_WORKSPACES validation)
├── indexer.py            ← Workspace traversal, chunking, ChromaDB storage
├── main.py               ← FastAPI app (legacy HTTP server, not used by default)
├── get_spotify_refresh_token.py ← OAuth helper for Spotify setup
├── start_eureka.bat      ← Windows batch launcher
├── start_invisible.vbs   ← Runs start_eureka.bat with no console window
└── logs/                 ← Worker logs (gitignored)
```

## Security Notes

- The worker only reads/writes paths within `ALLOWED_WORKSPACES` — enforced via resolved canonical paths on every file/git operation.
- `apply_patch` runs `git apply --check` (dry-run) before every real apply.
- The Redis connection uses TLS (`rediss://`).
- No HTTP port is exposed; the worker is not reachable from the network.
