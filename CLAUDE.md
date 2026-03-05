# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Eureka** is a personal AI coding assistant delivered via a Telegram bot. It has two runtime components that must both be running:

1. **`orchestrator/`** — Next.js 16 app (the Telegram webhook receiver and LLM orchestrator).
2. **`rag-daemon/`** — Local Python FastAPI daemon (code indexer, patch applier, git operator, Spotify controller).

## Commands

### Orchestrator (Next.js)

```bash
cd orchestrator
npm install          # install dependencies
npm run dev          # start dev server (http://localhost:3000)
npm run build        # production build
npm run lint         # run ESLint
```

### RAG Daemon (Python)

```bash
cd rag-daemon
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then edit ALLOWED_WORKSPACES
uvicorn main:app --host 127.0.0.1 --port 8765
```

No tests exist in either component.

## Environment Variables

### `orchestrator/.env.local` (required)
Copy from `orchestrator/.env.local.example`. Key variables:
- `TELEGRAM_BOT_TOKEN` — from @BotFather
- `ALLOWED_TELEGRAM_CHAT_IDS` — comma-separated numeric Telegram chat IDs (all others silently dropped)
- `TELEGRAM_WEBHOOK_SECRET` — optional, validates the `X-Telegram-Bot-Api-Secret-Token` header
- `LOCAL_DAEMON_URL` — URL of the running rag-daemon (e.g. `http://127.0.0.1:8765`)
- `LOCAL_DAEMON_WORKSPACE_PATH` — default absolute path for git/patch operations
- `OPENAI_API_KEY` — used by the Vercel AI SDK
- `LLM_MODEL_NAME` — defaults to `gpt-4.1-mini`; `LLM_MODEL_NAME_DEV` overrides for dev/planner agents
- `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` — Upstash Redis for chat history and job state

### `rag-daemon/.env`
Copy from `rag-daemon/.env.example`. Key variables:
- `ALLOWED_WORKSPACES` — JSON array of absolute paths the daemon may read/write (validated on startup)
- `GITHUB_TOKEN` — optional, enables `/github/create-repo`
- `OPENAI_API_KEY` — injected into `aider` subprocesses for DevMode build steps
- `SPOTIFY_*` — optional, enables Spotify control tools

## Architecture

### Request Flow

```
Telegram → POST /api/webhooks/telegram → processMessage()
                                            ├── Action buttons (apply_patch:, push:, push_only:)
                                            ├── "status" / "cancel" commands
                                            ├── isFullBuildIntent() → runFullBuildInBackground()
                                            │       generateProjectRoadmap() → executeProjectRoadmap()
                                            ├── isResearchRequest() → runResearchInBackground()
                                            │       runResearchPipeline() (Researcher→Writer→Reviewer)
                                            ├── isDevFeatureRequest() → runDevAgentInBackground()
                                            │       processUserMessage({ devMode: true })
                                            └── Normal chat → processUserMessage()
                                                    └── generateText() with LLM tools
```

The webhook always returns HTTP 200 immediately. Heavy work runs via `waitUntil()` (Vercel) or fire-and-forget.

### Key Files

**Orchestrator:**
- `app/api/webhooks/telegram/route.ts` — Telegram webhook handler; all routing logic lives here
- `lib/ai/orchestrator.ts` — `processUserMessage()`: wraps `generateText`, manages Redis history, detects dev/research intent
- `lib/ai/tools.ts` — LLM-callable tools: `search_local_codebase`, `request_patch_approval`, `spotify_*`
- `lib/ai/planner.ts` — `generateProjectRoadmap()`: uses `generateObject` + Zod to produce a `ProjectPlan` (4–8 `BuildStep`s)
- `lib/ai/executor.ts` — `executeProjectRoadmap()`: iterates `BuildStep`s, calls daemon `/execute-build-step`, then `/github/create-repo` + `/git-push`
- `lib/ai/memory.ts` — Long-term user profile extraction (runs every 5 messages, stored in Redis for 30 days)
- `lib/ai/researchAgents.ts` — Multi-agent research pipeline: Researcher → Writer → Reviewer (max 2 review rounds)
- `lib/ai/telemetry.ts` — `withTelemetry()` helper; spread into all `generateText`/`generateObject` calls for OTEL tracing
- `lib/redis.ts` — Upstash Redis singleton; chat history, staged patches/pushes, DevJob tracking, research state
- `lib/adapters/TelegramAdapter.ts` — Translates Telegram `Update` ↔ `StandardMessage`/`StandardResponse`

**RAG Daemon:**
- `main.py` — FastAPI app; all endpoints (`/search`, `/apply-patch`, `/execute-build-step`, `/git/*`, `/spotify/*`, `/save-markdown-as-pdf`, etc.)
- `config.py` — `DaemonSettings` (pydantic-settings); resolves and validates `ALLOWED_WORKSPACES` on startup
- `indexer.py` — Workspace traversal, chunking, embedding with `all-MiniLM-L6-v2`, ChromaDB persistence

### Redis Key Schema

| Key pattern | Purpose |
|---|---|
| `chat:{senderId}` | Chat history list (last 10 messages, 24h TTL) |
| `user_profile:{userId}` | Long-term developer profile (30d TTL) |
| `msg_counter:{userId}` | Counter for memory extraction gating |
| `patch:{id}` | Staged patch awaiting approval (1h TTL) |
| `push:{id}` / `pushonly:{id}` / `commit:{id}` | Staged git operations (1h TTL) |
| `job:{id}` | DevJob progress object (2h TTL) |
| `activejob:{senderId}` | Points to current job ID |
| `activeresearch:{senderId}` | Research pipeline state + cancellation flag |

### LLM Model Configuration

All LLM calls use OpenAI via `@ai-sdk/openai`. Model names are controlled by env vars:
- `LLM_MODEL_NAME` — normal chat and fallback
- `LLM_MODEL_NAME_DEV` — planner, executor summary, dev agent
- `LLM_MODEL_NAME_MEMORY` — profile extraction
- `LLM_MODELS_RESEARCH` — comma-separated list for research agents (index 0=Researcher, 1=Writer, 2=Reviewer, 3=Summariser)

### DevMode Build Pipeline

1. User sends a message matching `isFullBuildIntent()` phrases (e.g. "build a full X from scratch")
2. `generateProjectRoadmap()` calls `generateObject` to produce a validated `ProjectPlan` (kebab-case `projectName`, `techStack`, 4–8 `BuildStep`s)
3. `executeProjectRoadmap()` iterates steps sequentially, POSTing each to daemon `/execute-build-step` (which runs `aider` or shell commands in the project directory)
4. On success, calls `/github/create-repo` then `/git-push`; job state tracked in Redis and progress sent via Telegram

### Security Model

- All Telegram chat IDs are allowlisted via `ALLOWED_TELEGRAM_CHAT_IDS`; unknown senders get a silent 200
- Optional `TELEGRAM_WEBHOOK_SECRET` validates the `X-Telegram-Bot-Api-Secret-Token` header
- Daemon enforces `ALLOWED_WORKSPACES` containment on all file/git operations using resolved canonical paths
- `/apply-patch` validates with `git apply --check` before applying; patches arrive via stdin, never temp files
- Daemon binds to `127.0.0.1` only; not safe to expose to the network
