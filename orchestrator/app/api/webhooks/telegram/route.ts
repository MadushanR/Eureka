/**
 * app/api/webhooks/telegram/route.ts — Telegram Webhook Receiver
 * ===============================================================
 * Next.js App Router POST handler for incoming Telegram Bot webhook updates.
 *
 * How it works
 * ------------
 * 1. Telegram sends a POST request here whenever a user messages the bot.
 * 2. We MUST respond with HTTP 200 within a few seconds or Telegram will
 *    retry the delivery up to 50 times over the next few minutes.
 * 3. We parse the payload, perform a security allowlist check, and return
 *    200 immediately — even when we drop unauthorised requests.
 * 4. Any actual processing (LLM calls, RAG queries) happens asynchronously
 *    after we've already sent the 200.
 *
 * Security model
 * --------------
 * ALLOWED_TELEGRAM_CHAT_ID is a comma-separated list of numeric Telegram
 * chat IDs loaded from the environment.  Any message from a chat ID not on
 * this list is silently dropped (we still return 200 to stop Telegram retrying).
 *
 * Returning an error status for unauthorised users would:
 *   a) cause Telegram to retry endlessly, flooding our logs, and
 *   b) potentially reveal to the attacker that their ID is blocked
 *      (if we returned a distinct 403 vs 200 for known IDs).
 *
 * Environment variables required
 * ------------------------------
 *   TELEGRAM_BOT_TOKEN       — Secret token from @BotFather.
 *   TELEGRAM_WEBHOOK_SECRET  — (Optional, recommended) Secret token set via
 *                              setWebhook's `secret_token` parameter.
 *                              If set, every request must include the header
 *                              X-Telegram-Bot-Api-Secret-Token.
 *   ALLOWED_TELEGRAM_CHAT_IDS — Comma-separated list of permitted chat IDs.
 *                               Example: "123456789,987654321"
 */

import { NextRequest, NextResponse } from "next/server";
import { TelegramAdapter, type TelegramUpdate } from "@/lib/adapters/TelegramAdapter";
import type { StandardMessage } from "@/types/messaging";
import { processUserMessage, isDevFeatureRequest } from "@/lib/ai/orchestrator";
import { buildPushCompletionUrl } from "@/lib/pushCallback";
import { getPendingPatch, getPendingPush, getPendingPushOnly, getActiveJob, updateJob } from "@/lib/redis";

// ---------------------------------------------------------------------------
// Environment variable loading (validated at module-initialisation time so
// the server fails loudly on startup rather than at request time).
// ---------------------------------------------------------------------------

/** Telegram bot token.  Required. */
const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;

/**
 * Optional webhook secret.  When set, Telegram sends it in every request
 * header; we reject requests that are missing or don't match.
 * Set this via the `setWebhook` API call:
 *   https://api.telegram.org/bot<TOKEN>/setWebhook?url=...&secret_token=<SECRET>
 */
const WEBHOOK_SECRET = process.env.TELEGRAM_WEBHOOK_SECRET;

/**
 * Set of permitted Telegram chat IDs.  A message from any ID not in this set
 * is silently dropped.
 *
 * Parsing strategy: split on commas, trim whitespace, filter empty strings.
 * This is tolerant of accidental spaces (e.g. "123 , 456") or trailing commas.
 */
const ALLOWED_CHAT_IDS: ReadonlySet<string> = new Set(
    (process.env.ALLOWED_TELEGRAM_CHAT_IDS ?? "")
        .split(",")
        .map((id) => id.trim())
        .filter(Boolean)
);

// Warn loudly in logs if the allowlist is empty — this is almost certainly a
// misconfiguration and would silently reject every legitimate message.
if (ALLOWED_CHAT_IDS.size === 0) {
    console.warn(
        "[telegram/webhook] WARNING: ALLOWED_TELEGRAM_CHAT_IDS is empty or not set. " +
        "ALL incoming messages will be dropped. " +
        "Configure this variable in your .env.local file."
    );
}

// ---------------------------------------------------------------------------
// Adapter singleton
// Instantiated once per cold start; SentenceTransformer equivalent here is
// the model/token validation in the constructor.
// ---------------------------------------------------------------------------
let adapter: TelegramAdapter | null = null;

function getAdapter(): TelegramAdapter {
    if (!adapter) {
        if (!BOT_TOKEN) {
            throw new Error(
                "[telegram/webhook] TELEGRAM_BOT_TOKEN environment variable is not set. " +
                "Add it to your .env.local file."
            );
        }
        adapter = new TelegramAdapter(BOT_TOKEN);
    }
    return adapter;
}

// ---------------------------------------------------------------------------
// POST /api/webhooks/telegram
// ---------------------------------------------------------------------------

/**
 * Handles all incoming Telegram webhook updates.
 *
 * Returns HTTP 200 in every non-fatal case (including dropped / unauthorised
 * messages) to prevent Telegram from retrying the delivery.
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
    // ------------------------------------------------------------------ //
    // SECURITY GATE 1: Webhook Secret Token Verification                  //
    // ------------------------------------------------------------------ //
    // If TELEGRAM_WEBHOOK_SECRET is configured we validate the
    // X-Telegram-Bot-Api-Secret-Token header on every request.
    // This ensures that only Telegram (which knows the secret) can call
    // this endpoint, hardening it against replay and spoofing attacks.
    if (WEBHOOK_SECRET) {
        const receivedSecret = request.headers.get("x-telegram-bot-api-secret-token");

        if (receivedSecret !== WEBHOOK_SECRET) {
            // Use a constant-time comparison in production to prevent timing
            // attacks.  For simplicity here we use strict equality — if this
            // becomes a concern, swap in `crypto.timingSafeEqual`.
            console.warn(
                "[telegram/webhook] Request rejected: invalid or missing " +
                "X-Telegram-Bot-Api-Secret-Token header."
            );
            // Return 200 rather than 401/403 to avoid revealing the check to
            // potential attackers scanning the endpoint.
            return NextResponse.json({ ok: true }, { status: 200 });
        }
    }

    // ------------------------------------------------------------------ //
    // Parse the request body                                              //
    // ------------------------------------------------------------------ //
    let rawPayload: unknown;
    try {
        rawPayload = await request.json();
    } catch {
        // Malformed JSON from Telegram is unexpected but shouldn't crash the server.
        console.error("[telegram/webhook] Failed to parse request body as JSON.");
        // Still return 200 so Telegram doesn't retry a clearly bad payload.
        return NextResponse.json({ ok: true }, { status: 200 });
    }

    // ------------------------------------------------------------------ //
    // Parse via the adapter                                              //
    // ------------------------------------------------------------------ //
    let message: StandardMessage;
    try {
        message = getAdapter().parseWebhook(rawPayload as TelegramUpdate);
    } catch (parseError) {
        // This can happen for legitimate but unsupported update types
        // (e.g. inline queries, poll updates).  Log and discard gracefully.
        console.info(
            "[telegram/webhook] Skipping unsupported update type:",
            parseError instanceof Error ? parseError.message : String(parseError)
        );
        return NextResponse.json({ ok: true }, { status: 200 });
    }

    // ------------------------------------------------------------------ //
    // SECURITY GATE 2: Allowed Chat ID Check                             //
    // ------------------------------------------------------------------ //
    // Verify that the sender is on the allowlist BEFORE doing any further
    // processing.  An unauthorised message is silently dropped (200 returned)
    // to avoid leaking information about the allowlist membership to attackers.
    if (!ALLOWED_CHAT_IDS.has(message.senderId)) {
        console.warn(
            `[telegram/webhook] SECURITY: Message from unauthorised sender ` +
            `'${message.senderId}' dropped silently.`
        );
        // Return 200 to Telegram so it doesn't retry.
        return NextResponse.json({ ok: true }, { status: 200 });
    }

    // ------------------------------------------------------------------ //
    // Return 200 immediately — async processing happens below            //
    // ------------------------------------------------------------------ //
    // We must return the NextResponse BEFORE doing any slow work (LLM calls,
    // RAG queries, patch execution) to comply with Telegram's timeout.
    //
    // IMPORTANT: In the current Next.js App Router, background tasks started
    // after returning a response may be killed if the serverless function
    // deallocates.  For production, offload work to a queue (Redis, SQS, etc.).
    // The `waitUntil` API from @vercel/functions can be used on Vercel to
    // extend the function lifetime:
    //   import { waitUntil } from '@vercel/functions';
    //   waitUntil(processMessage(message));

    // Fire-and-forget: kick off downstream processing in the background.
    void processMessage(message);

    return NextResponse.json({ ok: true }, { status: 200 });
}

/** Handle "Approve & Apply" button: apply staged patch via daemon and reply. */
async function handleApplyPatchAction(message: StandardMessage): Promise<boolean> {
    const prefix = "apply_patch:";
    if (!message.isAction || !message.text.startsWith(prefix)) return false;

    const patchId = message.text.slice(prefix.length).trim();
    if (!patchId) return false;

    const adapter = getAdapter();
    if (message.callbackQueryId) {
        await adapter.answerCallbackQuery(message.callbackQueryId);
    }

    const staged = await getPendingPatch(patchId);
    if (!staged) {
        await adapter.sendResponse(
            { text: "This patch has expired or was already applied. Please ask for the change again." },
            message.senderId
        );
        return true;
    }

    const workspacePath =
        staged.workspace_path?.trim() ||
        process.env.LOCAL_DAEMON_WORKSPACE_PATH?.trim();
    if (!workspacePath) {
        await adapter.sendResponse(
            {
                text:
                    "No workspace path was set for this patch. Add LOCAL_DAEMON_WORKSPACE_PATH to your orchestrator .env.local (e.g. the path to your project).",
            },
            message.senderId
        );
        return true;
    }

    const daemonUrl = process.env.LOCAL_DAEMON_URL?.replace(/\/+$/, "");
    if (!daemonUrl) {
        await adapter.sendResponse(
            { text: "LOCAL_DAEMON_URL is not configured. Cannot apply the patch." },
            message.senderId
        );
        return true;
    }

    try {
        const res = await fetch(`${daemonUrl}/apply-patch`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patch_string: staged.patch_string,
                workspace_path: workspacePath,
            }),
        });
        const data = (await res.json().catch(() => ({}))) as {
            success?: boolean;
            message?: string;
            stdout?: string;
            stderr?: string;
        };

        if (data.success) {
            await adapter.sendResponse(
                { text: `Patch applied successfully. ${data.message ?? ""}`.trim() },
                message.senderId
            );
        } else {
            const detail = [data.stdout, data.stderr].filter(Boolean).join("\n");
            await adapter.sendResponse(
                {
                    text: `Patch could not be applied. ${data.message ?? "Unknown error."}${detail ? `\n\n${detail}` : ""}`,
                },
                message.senderId
            );
        }
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        await adapter.sendResponse(
            { text: `Failed to reach the daemon to apply the patch: ${msg}` },
            message.senderId
        );
    }

    return true;
}

/** Handle "Approve & Push" button: commit and push staged request via daemon. */
async function handlePushAction(message: StandardMessage): Promise<boolean> {
    const prefix = "push:";
    if (!message.isAction || !message.text.startsWith(prefix)) return false;

    const pushId = message.text.slice(prefix.length).trim();
    if (!pushId) return false;

    const adapter = getAdapter();
    if (message.callbackQueryId) {
        await adapter.answerCallbackQuery(message.callbackQueryId);
    }

    const staged = await getPendingPush(pushId);
    if (!staged) {
        await adapter.sendResponse(
            { text: "This push request has expired. Ask again to see the diff and approve." },
            message.senderId
        );
        return true;
    }

    const daemonUrl = process.env.LOCAL_DAEMON_URL?.replace(/\/+$/, "");
    if (!daemonUrl) {
        await adapter.sendResponse(
            { text: "LOCAL_DAEMON_URL is not configured. Cannot push." },
            message.senderId
        );
        return true;
    }

    const completionUrl = buildPushCompletionUrl(message.senderId, "commit-and-push");

    try {
        const body: { workspace_path: string; commit_message: string; completion_webhook_url?: string } = {
            workspace_path: staged.workspace_path,
            commit_message: staged.commit_message,
        };
        if (completionUrl) body.completion_webhook_url = completionUrl;

        const res = await fetch(`${daemonUrl}/git/commit-and-push`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = (await res.json().catch(() => ({}))) as {
            success?: boolean;
            message?: string;
            stdout?: string;
            stderr?: string;
            started?: boolean;
        };

        if (res.status === 202 && data.started) {
            await adapter.sendResponse(
                { text: "Push started in the background. You'll get a message when it completes." },
                message.senderId
            );
            return true;
        }

        if (data.success) {
            await adapter.sendResponse(
                { text: `Push completed. ${data.message ?? ""}`.trim() },
                message.senderId
            );
        } else {
            const detail = [data.stdout, data.stderr].filter(Boolean).join("\n");
            await adapter.sendResponse(
                {
                    text: `Push failed. ${data.message ?? "Unknown error."}${detail ? `\n\n${detail}` : ""}`,
                },
                message.senderId
            );
        }
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        await adapter.sendResponse(
            { text: `Failed to reach the daemon to push: ${msg}` },
            message.senderId
        );
    }

    return true;
}

/** Handle push-only button: run git push without staging/committing. */
async function handlePushOnlyAction(message: StandardMessage): Promise<boolean> {
    const prefix = "push_only:";
    if (!message.text.startsWith(prefix)) return false;

    const pushOnlyId = message.text.slice(prefix.length).trim();
    if (!pushOnlyId) return false;

    const adapter = getAdapter();
    if (message.callbackQueryId) {
        await adapter.answerCallbackQuery(message.callbackQueryId);
    }

    const staged = await getPendingPushOnly(pushOnlyId);
    if (!staged) {
        await adapter.sendResponse(
            { text: "This push request has expired. Ask again to create a new push-only approval." },
            message.senderId
        );
        return true;
    }

    const daemonUrl = process.env.LOCAL_DAEMON_URL?.replace(/\/+$/, "");
    if (!daemonUrl) {
        await adapter.sendResponse(
            { text: "LOCAL_DAEMON_URL is not configured. Cannot push." },
            message.senderId
        );
        return true;
    }

    const completionUrl = buildPushCompletionUrl(message.senderId, "push-only");

    try {
        const body: { workspace_path: string; completion_webhook_url?: string } = {
            workspace_path: staged.workspace_path,
        };
        if (completionUrl) body.completion_webhook_url = completionUrl;

        const res = await fetch(`${daemonUrl}/git/push-only`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = (await res.json().catch(() => ({}))) as {
            success?: boolean;
            message?: string;
            stdout?: string;
            stderr?: string;
            started?: boolean;
        };

        if (res.status === 202 && data.started) {
            await adapter.sendResponse(
                { text: "Push started in the background. You'll get a message when it completes." },
                message.senderId
            );
            return true;
        }

        if (data.success) {
            await adapter.sendResponse(
                { text: `Push-only completed. ${data.message ?? ""}`.trim() },
                message.senderId
            );
        } else {
            const detail = [data.stdout, data.stderr].filter(Boolean).join("\n");
            await adapter.sendResponse(
                {
                    text: `Push-only failed. ${data.message ?? "Unknown error."}${detail ? `\n\n${detail}` : ""}`,
                },
                message.senderId
            );
        }
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        await adapter.sendResponse(
            { text: `Failed to reach the daemon to push: ${msg}` },
            message.senderId
        );
    }

    return true;
}

// ---------------------------------------------------------------------------
// Downstream message processing — wired to the LLM orchestrator
// ---------------------------------------------------------------------------

/**
 * Process an authorised, normalised message through the full LLM pipeline:
 *   1. Load chat history from Redis.
 *   2. Run generateText (Vercel AI SDK) with tools (RAG search, patch approval).
 *   3. Persist the updated history to Redis.
 *   4. Send the final StandardResponse back via the Telegram adapter.
 *
 * Any unhandled errors must NOT propagate to the route handler —
 * that would break the 200 guarantee.  All errors are caught and logged.
 */
async function processMessage(message: StandardMessage): Promise<void> {
    try {
        console.info(
            `[telegram/webhook] Processing message from sender=${message.senderId}: ` +
            `"${message.text.slice(0, 80)}${message.text.length > 80 ? "…" : ""}"`
        );

        // Handle "Approve & Apply" button: apply staged patch via daemon, then done.
        if (await handleApplyPatchAction(message)) {
            console.info(`[telegram/webhook] Apply-patch action handled for sender=${message.senderId}.`);
            return;
        }

        // Handle push-only approval button: run git push only, then done.
        if (await handlePushOnlyAction(message)) {
            console.info(`[telegram/webhook] Push-only action handled for sender=${message.senderId}.`);
            return;
        }

        // Handle "Approve & Push" button: commit and push via daemon, then done.
        if (await handlePushAction(message)) {
            console.info(`[telegram/webhook] Push action handled for sender=${message.senderId}.`);
            return;
        }

        // "status" command: return progress of active dev job
        const lowerText = message.text.trim().toLowerCase();
        if (lowerText === "status" || lowerText === "/status") {
            const activeJob = await getActiveJob(message.senderId);
            if (!activeJob) {
                await getAdapter().sendResponse({ text: "No active job running." }, message.senderId);
            } else {
                const elapsed = Math.round((Date.now() - activeJob.started_at) / 1000);
                const lines = [
                    `Job: ${activeJob.goal.slice(0, 80)}`,
                    `Status: ${activeJob.status}`,
                    `Step: ${activeJob.current_step}/${activeJob.total_steps}`,
                    `Current: ${activeJob.current_action}`,
                    `Elapsed: ${elapsed}s`,
                ];
                if (activeJob.steps_completed.length > 0) {
                    lines.push(`Done: ${activeJob.steps_completed.join(", ")}`);
                }
                if (activeJob.errors.length > 0) {
                    lines.push(`Errors: ${activeJob.errors.slice(-2).join("; ")}`);
                }
                await getAdapter().sendResponse({ text: lines.join("\n") }, message.senderId);
            }
            return;
        }

        // "cancel" command: cancel active dev job
        if (lowerText === "cancel" || lowerText === "/cancel") {
            const activeJob = await getActiveJob(message.senderId);
            if (!activeJob) {
                await getAdapter().sendResponse({ text: "No active job to cancel." }, message.senderId);
            } else {
                await updateJob(activeJob.id, { status: "cancelled", finished_at: Date.now() });
                await getAdapter().sendResponse(
                    { text: `Cancelled job: ${activeJob.goal.slice(0, 80)}\n(Note: changes already applied to files remain.)` },
                    message.senderId,
                );
            }
            return;
        }

        // Dev feature requests: send immediate ack, then run dev-agent with progress updates.
        if (isDevFeatureRequest(message.text)) {
            const tgAdapter = getAdapter();
            await tgAdapter.sendResponse(
                { text: "Starting dev agent. I'll send you live updates as I work." },
                message.senderId,
            );

            let lastProgressTime = 0;
            const MIN_PROGRESS_INTERVAL_MS = 3000;

            const onProgress = async (update: string) => {
                const now = Date.now();
                if (now - lastProgressTime < MIN_PROGRESS_INTERVAL_MS) return;
                lastProgressTime = now;
                try {
                    await tgAdapter.sendResponse({ text: `[progress] ${update}` }, message.senderId);
                } catch (e) {
                    console.error(`[telegram/webhook] Failed to send progress update:`, e);
                }
            };

            const { response } = await processUserMessage(message, { devMode: true, onProgress });
            await tgAdapter.sendResponse(response, message.senderId);
            console.info(`[telegram/webhook] Dev flow completed; summary sent to sender=${message.senderId}.`);
            return;
        }

        // Delegate to the orchestrator — it handles Redis history, LLM calls,
        // tool execution (RAG search, patch approval), and persistence.
        const { response } = await processUserMessage(message);

        // Send the final StandardResponse back to Telegram.
        await getAdapter().sendResponse(response, message.senderId);

        console.info(`[telegram/webhook] Reply sent to sender=${message.senderId}.`);
    } catch (error) {
        // Log the error but do NOT re-throw — we've already returned 200 to
        // Telegram and cannot change that now.
        console.error(
            `[telegram/webhook] Error while processing message from sender=${message.senderId}:`,
            error
        );
    }
}
