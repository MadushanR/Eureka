# Hybrid RAG Edge Daemon

A local edge daemon that indexes your code into a vector database and exposes semantic search and git-patch application endpoints.

## Stack

| Layer | Library |
|---|---|
| HTTP server | FastAPI + Uvicorn |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Config | `pydantic-settings` + `.env` |

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
# Edit .env and set ALLOWED_WORKSPACES to your project paths

# 4. Start the daemon
uvicorn main:app --host 127.0.0.1 --port 8765
```

On first start the daemon will:
1. Load the `all-MiniLM-L6-v2` model (downloaded once, then cached).
2. Index all files in your `ALLOWED_WORKSPACES` in a background thread.
3. Begin accepting API requests immediately (even while indexing).

## Endpoints

### `GET /health`
Liveness probe — returns daemon status and indexed chunk count.

### `POST /search`
Semantic search over indexed code.

```json
{ "query": "JWT token validation", "top_k": 5 }
```

**Response** — ranked list of code snippets with file path + line numbers.

### `POST /apply-patch`
Apply a unified diff to a workspace.

```json
{
  "patch_string": "--- a/foo.py\n+++ b/foo.py\n@@...",
  "workspace_path": "/absolute/path/to/project"
}
```

**Security**: `workspace_path` **must** be within `ALLOWED_WORKSPACES`.  
The patch is validated with `git apply --check` before being applied.

### `POST /index-now`
Trigger a manual re-index of all workspaces in the background.

## Project Structure

```
rag-daemon/
├── .env.example      ← Copy to .env and fill in your values
├── requirements.txt  ← pip dependencies
├── config.py         ← pydantic-settings configuration
├── indexer.py        ← Workspace traversal, chunking, embedding, ChromaDB storage
└── main.py           ← FastAPI application (routes, security, lifespan)
```

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `ALLOWED_WORKSPACES` | `[]` | JSON array of absolute workspace paths |
| `IGNORE_PATTERNS` | see `.env.example` | Dirs/filenames to skip |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB data directory |
| `CHUNK_TOKEN_LIMIT` | `500` | Max tokens per chunk |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8765` | Listen port |

## Security Notes

- The daemon binds to `127.0.0.1` by default — **do not expose to the network**.
- `/apply-patch` enforces a strict workspace containment check using resolved (canonical) paths, preventing symlink and path-traversal attacks.
- Patches are delivered via stdin — never written to temp files.
- `git apply --check` (dry-run) runs before every real apply.
- All unhandled exceptions return a generic 500 message to prevent information leakage.
