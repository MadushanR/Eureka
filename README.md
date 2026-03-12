# Eureka

A personal AI coding assistant delivered via a Telegram bot. It combines a Next.js orchestrator (deployed on Vercel) with a local Python host worker that communicates through Upstash Redis — no tunnels or open ports required.

## Why Eureka?

- **Your code never leaves your machine.** Semantic search and indexing run entirely on the local host worker (ChromaDB + sentence-transformers). The cloud LLM only sees search results and snippets you approve — your full codebase stays local.

- **Telegram as the only interface.** No separate dashboard or IDE plugin. Chat, approve patches, trigger full builds, run research, and control optional integrations (e.g. Spotify) from one place, with progress and action buttons (Apply / Push / Cancel) in the thread.

- **Human-in-the-loop edits.** The assistant proposes patches; you approve or reject them in Telegram. No silent file writes — every code change goes through explicit approval, with optional one-tap push to your repo.

- **"Build it from scratch" in one message.** Full-build intents trigger a structured pipeline: the LLM generates a project roadmap (4–8 steps), the worker runs each step (via aider or shell) in your workspace, then can create a GitHub repo and push. End-to-end from natural language.

- **Multi-agent research pipeline.** Research requests are handled by a dedicated pipeline (Researcher → Writer → Reviewer with review rounds), not a single LLM call, for more coherent long-form output.

- **Long-term memory.** A developer profile is extracted periodically from the conversation and stored in Redis (e.g. 30 days), so the assistant can adapt to your stack, style, and preferences over time.

- **No open ports.** The orchestrator (Vercel) and the local host worker communicate exclusively through an Upstash Redis queue. The worker uses `BLPOP` to pull tasks; no tunnel or exposed HTTP port is needed.

- **Designed for one user.** Allowlisted Telegram chat IDs, strict workspace containment for all file/git operations, and patches validated with `git apply --check` before apply. Built for a single developer's machine, not multi-tenant.

## Components

| Component | Stack | Purpose |
|-----------|--------|---------|-
| **orchestrator/** | Next.js 16 (Vercel) | Telegram webhook receiver, LLM orchestration, Redis-backed chat/job state |
| **rag-daemon/host_worker.py** | Python + Redis + ChromaDB | Local task runner — pulls commands from Redis queue, runs code search, patches, git ops, Spotify, etc. |

Both must be running for the bot to work.

## Quick Start

### 1. Orchestrator

```bash
cd orchestrator
npm install
cp .env.local.example .env.local   # fill in TELEGRAM_BOT_TOKEN, ALLOWED_TELEGRAM_CHAT_IDS,
                                    # OPENAI_API_KEY, UPSTASH_REDIS_REST_URL/TOKEN, etc.
npm run dev
```

Deploy to Vercel and [register your Telegram webhook](https://core.telegram.org/bots/api#setwebhook) pointing to `https://<your-app>.vercel.app/api/webhooks/telegram`.

### 2. Host Worker

```bash
cd rag-daemon
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate        # Linux / macOS
pip install -r requirements.txt
cp .env.example .env                # set UPSTASH_REDIS_URL, TELEGRAM_BOT_TOKEN, ALLOWED_WORKSPACES
python host_worker.py
```

On Windows you can run it headlessly at startup using the included `start_invisible.vbs` launcher (logs go to `rag-daemon/logs/worker.log`).

> **Note:** `main.py` (FastAPI) is no longer the primary entry point. All orchestrator ↔ worker communication goes through the Upstash Redis queue (`eureka:host_commands`). `LOCAL_DAEMON_URL` is no longer required.

## Docs

- **orchestrator/** — [orchestrator/README.md](orchestrator/README.md) (Next.js app)
- **rag-daemon/** — [rag-daemon/README.md](rag-daemon/README.md) (worker, config, security)
- **CLAUDE.md** — architecture, request flow, env vars, key files (for AI/contributors)

## License

See [LICENSE](LICENSE).
