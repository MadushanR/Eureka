# Eureka Orchestrator

Next.js 16 app deployed on Vercel. Receives Telegram webhook updates, runs LLM orchestration via the Vercel AI SDK, and dispatches local tasks to the host worker through an Upstash Redis queue.

## Stack

| Layer | Library |
|---|---|
| Framework | Next.js 16 (App Router) |
| LLM | Vercel AI SDK (`ai`, `@ai-sdk/openai`) |
| State / queue | Upstash Redis (`@upstash/redis`) |
| Telegram | Raw Bot API (webhook) |

## Setup

```bash
npm install
cp .env.local.example .env.local   # fill in required values (see below)
npm run dev                         # http://localhost:3000
```

### Environment Variables (`orchestrator/.env.local`)

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | ✓ | From @BotFather |
| `ALLOWED_TELEGRAM_CHAT_IDS` | ✓ | Comma-separated chat IDs; all others are silently dropped |
| `OPENAI_API_KEY` | ✓ | OpenAI key used by `@ai-sdk/openai` |
| `UPSTASH_REDIS_REST_URL` | ✓ | From Upstash console → Connect → REST |
| `UPSTASH_REDIS_REST_TOKEN` | ✓ | From Upstash console → Connect → REST |
| `TELEGRAM_WEBHOOK_SECRET` | — | Validates `X-Telegram-Bot-Api-Secret-Token` header |
| `LOCAL_DAEMON_WORKSPACE_PATH` | — | Default workspace path for patch/git operations |
| `EUREKA_SELF_PATH` | — | Absolute path to this Eureka repo (enables self-modification) |
| `LLM_MODEL_NAME` | — | Defaults to `gpt-4.1-mini` |
| `LLM_MODEL_NAME_DEV` | — | Model for planner/executor/dev agent |
| `LLM_MODEL_NAME_MEMORY` | — | Model for profile extraction |
| `LLM_MODELS_RESEARCH` | — | Comma-separated: Researcher, Writer, Reviewer, Summariser |

### Registering the Telegram Webhook

After deploying to Vercel, register your webhook once:

```
https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://<your-app>.vercel.app/api/webhooks/telegram&secret_token=<TELEGRAM_WEBHOOK_SECRET>
```

## Commands

```bash
npm run dev      # development server
npm run build    # production build
npm run lint     # ESLint
```

## Key Files

| File | Purpose |
|---|---|
| `app/api/webhooks/telegram/route.ts` | Webhook handler — all routing (actions, status, build/research/dev intents, normal chat) |
| `lib/ai/orchestrator.ts` | `processUserMessage()` — wraps `generateText`, manages Redis history |
| `lib/ai/tools.ts` | LLM-callable tools (codebase search, file ops, git, GUI, Spotify, etc.) |
| `lib/ai/planner.ts` | `generateProjectRoadmap()` — produces a validated `ProjectPlan` via `generateObject` |
| `lib/ai/executor.ts` | `executeProjectRoadmap()` — iterates build steps via the worker queue |
| `lib/ai/memory.ts` | Long-term developer profile (runs every 5 messages, 30-day TTL) |
| `lib/ai/researchAgents.ts` | Multi-agent pipeline: Researcher → Writer → Reviewer |
| `lib/ai/telemetry.ts` | `withTelemetry()` helper for OTEL tracing |
| `lib/redis.ts` | Upstash Redis client — chat history, job state, `pushHostCommand`, `pollResult` |
| `lib/adapters/TelegramAdapter.ts` | Translates Telegram `Update` ↔ `StandardMessage`/`StandardResponse` |

## Communication with the Host Worker

All local operations go through the Upstash Redis queue — never direct HTTP:

```ts
// Synchronous (waits for result)
const taskId = await pushHostCommand("search", { query });
const result = await pollResult(taskId, 30_000);

// Fire-and-forget (worker sends Telegram reply itself)
await pushHostCommand("apply_patch", payload, { senderId, botToken, async: true });
```

The worker (`rag-daemon/host_worker.py`) BLPOPs from `eureka:host_commands` and publishes results to `eureka:result:{taskId}`.
