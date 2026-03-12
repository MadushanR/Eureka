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
import { processUserMessage, isDevFeatureRequest, isResearchRequest } from "@/lib/ai/orchestrator";
import { countWords, parseResearchMessage, runResearchPipeline } from "@/lib/ai/researchAgents";
import { shouldRunExtraction, updateUserProfile } from "@/lib/ai/memory";
import { generateProjectRoadmap } from "@/lib/ai/planner";
import { executeProjectRoadmap } from "@/lib/ai/executor";
import {
    getChatHistory,
    getPendingPatch,
    getPendingPush,
    getPendingPushOnly,
    deleteStagedPatch,
    deleteStagedPush,
    deleteStagedPushOnly,
    getActiveJob,
    updateJob,
    clearActiveJob,
    getActiveResearch,
    setActiveResearch,
    clearActiveResearch,
    setResearchCancelled,
    pushHostCommand,
} from "@/lib/redis";
import { ResearchCancelledError } from "@/lib/ai/researchAgents";

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

    // Extend the Vercel function lifetime so processMessage() is not killed
    // the moment the HTTP response is sent.  Without waitUntil(), Vercel
    // terminates the serverless function immediately after returning 200,
    // which means no LLM calls, no Redis writes, no logs.
    const messagePromise = processMessage(message);
    try {
        const { waitUntil } = await import("@vercel/functions");
        waitUntil(messagePromise);
    } catch {
        // Not running on Vercel (local dev) — fire-and-forget is fine.
        void messagePromise;
    }

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

    try {
        // Push the patch task to the Redis queue.  The worker will apply the
        // patch and send a Telegram notification directly when done.
        await pushHostCommand(
            "apply_patch",
            { patch_string: staged.patch_string, workspace_path: workspacePath },
            { senderId: message.senderId, botToken: process.env.TELEGRAM_BOT_TOKEN, async: true }
        );
        await adapter.sendResponse(
            { text: "Patch queued. You'll receive a message once it's applied." },
            message.senderId
        );
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        await adapter.sendResponse(
            { text: `Failed to queue the patch task: ${msg}` },
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

    try {
        // Push the commit-and-push task to the Redis queue.  The worker will
        // run the git operations and send a Telegram notification when done.
        await pushHostCommand(
            "git_commit_and_push",
            { workspace_path: staged.workspace_path, commit_message: staged.commit_message },
            { senderId: message.senderId, botToken: process.env.TELEGRAM_BOT_TOKEN, async: true }
        );
        await adapter.sendResponse(
            { text: "Push queued. You'll receive a message once the commit and push complete." },
            message.senderId
        );
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        await adapter.sendResponse(
            { text: `Failed to queue the push task: ${msg}` },
            message.senderId
        );
    }

    return true;
}

/** Handle push-only button: run git push without staging/committing. */
async function handlePushOnlyAction(message: StandardMessage): Promise<boolean> {
    const prefix = "push_only:";
    if (!message.isAction || !message.text.startsWith(prefix)) return false;

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

    try {
        // Push the push-only task to the Redis queue.  The worker will run
        // git push and send a Telegram notification when done.
        await pushHostCommand(
            "git_push_only",
            { workspace_path: staged.workspace_path },
            { senderId: message.senderId, botToken: process.env.TELEGRAM_BOT_TOKEN, async: true }
        );
        await adapter.sendResponse(
            { text: "Push queued. You'll receive a message once it completes." },
            message.senderId
        );
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        await adapter.sendResponse(
            { text: `Failed to queue the push-only task: ${msg}` },
            message.senderId
        );
    }

    return true;
}

async function handleRejectPatchAction(message: StandardMessage): Promise<boolean> {
    const prefix = "reject_patch:";
    if (!message.isAction || !message.text.startsWith(prefix)) return false;
    const patchId = message.text.slice(prefix.length).trim();
    if (!patchId) return false;
    const adapter = getAdapter();
    if (message.callbackQueryId) await adapter.answerCallbackQuery(message.callbackQueryId);
    await deleteStagedPatch(patchId);
    await adapter.sendResponse({ text: "Patch rejected and discarded." }, message.senderId);
    return true;
}

async function handleRejectPushAction(message: StandardMessage): Promise<boolean> {
    const prefix = "reject_push:";
    if (!message.isAction || !message.text.startsWith(prefix)) return false;
    const pushId = message.text.slice(prefix.length).trim();
    if (!pushId) return false;
    const adapter = getAdapter();
    if (message.callbackQueryId) await adapter.answerCallbackQuery(message.callbackQueryId);
    await deleteStagedPush(pushId);
    await adapter.sendResponse({ text: "Push rejected and discarded." }, message.senderId);
    return true;
}

async function handleRejectPushOnlyAction(message: StandardMessage): Promise<boolean> {
    const prefix = "reject_push_only:";
    if (!message.isAction || !message.text.startsWith(prefix)) return false;
    const pushOnlyId = message.text.slice(prefix.length).trim();
    if (!pushOnlyId) return false;
    const adapter = getAdapter();
    if (message.callbackQueryId) await adapter.answerCallbackQuery(message.callbackQueryId);
    await deleteStagedPushOnly(pushOnlyId);
    await adapter.sendResponse({ text: "Push rejected and discarded." }, message.senderId);
    return true;
}

// ---------------------------------------------------------------------------
// Background dev agent runner (fully isolated from normal message flow)
// ---------------------------------------------------------------------------

async function runDevAgentInBackground(
    message: StandardMessage,
    tgAdapter: TelegramAdapter,
): Promise<void> {
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

    try {
        const { response } = await processUserMessage(message, { devMode: true, onProgress });
        await tgAdapter.sendResponse(response, message.senderId);
        console.info(`[telegram/webhook] Dev flow completed; summary sent to sender=${message.senderId}.`);
    } catch (error) {
        console.error(`[telegram/webhook] Dev agent background error for sender=${message.senderId}:`, error);
        try {
            await tgAdapter.sendResponse(
                { text: "The dev agent hit an unexpected error. Please try again." },
                message.senderId,
            );
        } catch { /* best effort */ }
    }
}

// ---------------------------------------------------------------------------
// Background research runner (multi-agent pipeline)
// ---------------------------------------------------------------------------

async function runResearchInBackground(
    message: StandardMessage,
    tgAdapter: TelegramAdapter,
): Promise<void> {
    const senderId = message.senderId;
    let lastProgressTime = 0;
    const MIN_PROGRESS_INTERVAL_MS = 3000;

    const onProgress = async (update: string) => {
        const now = Date.now();
        if (now - lastProgressTime < MIN_PROGRESS_INTERVAL_MS) return;
        lastProgressTime = now;
        try {
            await tgAdapter.sendResponse({ text: `[research] ${update}` }, senderId);
        } catch (e) {
            console.error(`[telegram/webhook] Failed to send research progress:`, e);
        }
    };

    const getIsCancelled = async (): Promise<boolean> => {
        const r = await getActiveResearch(senderId);
        return r?.cancelled === true;
    };

    try {
        const { topic, targetWords } = parseResearchMessage(message.text);
        console.info(`[telegram/webhook] Starting research pipeline for sender=${senderId}${targetWords != null ? ` (target: ${targetWords} words)` : ""}`);
        const result = await runResearchPipeline(topic, onProgress, getIsCancelled, { targetWords });

        // Save paper as PDF on Desktop via the Redis worker queue.
        const workspacePath = process.env.LOCAL_DAEMON_WORKSPACE_PATH ?? "";
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        const filename = `research-paper-${timestamp}.pdf`;

        let savedPath = "";
        try {
            // Fire-and-forget: worker saves the PDF and notifies via Telegram.
            await pushHostCommand(
                "save_markdown_as_pdf",
                { workspace_path: workspacePath, filename, content: result.paper },
                { senderId, botToken: process.env.TELEGRAM_BOT_TOKEN, async: true }
            );
            savedPath = `${workspacePath}/${filename}`;
            console.info(`[telegram/webhook] Research PDF save queued: ${savedPath}`);
        } catch (e) {
            console.error(`[telegram/webhook] Error queuing research paper save:`, e);
        }

        const wordCount = countWords(result.paper);
        const responseLines = [
            "Research complete! Here are the key findings:",
            "",
            result.summary,
            "",
            savedPath
                ? `The full paper (~${wordCount} words, ${result.reviewRounds} review round${result.reviewRounds !== 1 ? "s" : ""}) has been saved on your Desktop as a PDF:\n${savedPath}`
                : `The paper is ~${wordCount} words (${result.reviewRounds} review round${result.reviewRounds !== 1 ? "s" : ""}).`,
        ];

        await tgAdapter.sendResponse({ text: responseLines.join("\n") }, senderId);
        console.info(`[telegram/webhook] Research completed for sender=${senderId}.`);
    } catch (error) {
        if (error instanceof ResearchCancelledError) {
            console.info(`[telegram/webhook] Research cancelled by user for sender=${senderId}.`);
            try {
                await tgAdapter.sendResponse({ text: "Research cancelled." }, senderId);
            } catch { /* best effort */ }
        } else {
            console.error(`[telegram/webhook] Research pipeline error for sender=${senderId}:`, error);
            try {
                await tgAdapter.sendResponse(
                    { text: "The research pipeline hit an unexpected error. Please try again." },
                    senderId,
                );
            } catch { /* best effort */ }
        }
    } finally {
        await clearActiveResearch(senderId);
    }
}

// ---------------------------------------------------------------------------
// Full-project build (single-prompt autonomous build) — intent from user input
// ---------------------------------------------------------------------------

/** Phrases that indicate the user wants a full app built from one prompt (no keyword). */
const FULL_BUILD_INTENT_PHRASES = [
    "build an entire",
    "build a full",
    "build me a complete",
    "build me a full",
    "build me an entire",
    "build a complete",
    "create an entire",
    "create a full",
    "create a complete",
    "create me a complete",
    "create me a full",
    "build from scratch",
    "create from scratch",
    "build a whole",
    "create a whole",
    "build me a whole ",
];

function isFullBuildIntent(text: string): boolean {
    const t = text.trim().toLowerCase();
    if (t.length < 15) return false;
    return FULL_BUILD_INTENT_PHRASES.some((phrase) => t.includes(phrase));
}

/** Parse optional "repo name: X" or "project name: X" from the message. Returns prompt (with that part removed) and preferred name. */
function parseFullBuildMessage(text: string): { prompt: string; preferredProjectName?: string } {
    const trimmed = text.trim();
    const match =
        trimmed.match(/\b(?:repo|project)\s*name\s*:\s*([^\n,]+)/i) ||
        trimmed.match(/\b(?:repo|project)\s*:\s*([^\n,]+)/i);
    const preferredProjectName = match ? match[1].trim().replace(/\s+/g, "-") : undefined;
    let prompt = trimmed;
    if (preferredProjectName) {
        prompt = trimmed
            .replace(/\b(?:repo|project)\s*name\s*:\s*[^\n,]+/gi, "")
            .replace(/\b(?:repo|project)\s*:\s*[^\n,]+/gi, "")
            .replace(/\s{2,}/g, " ")
            .replace(/^[\s,]+|[\s,]+$/g, "")
            .trim();
    }
    return { prompt, preferredProjectName };
}

async function runFullBuildInBackground(
    message: StandardMessage,
    tgAdapter: TelegramAdapter,
): Promise<void> {
    const senderId = message.senderId;
    let lastProgressTime = 0;
    const MIN_UPDATE_INTERVAL_MS = 15_000;

    const sendUpdate = async (update: string) => {
        const now = Date.now();
        const elapsed = now - lastProgressTime;
        if (elapsed < MIN_UPDATE_INTERVAL_MS && lastProgressTime > 0) {
            await new Promise((r) => setTimeout(r, MIN_UPDATE_INTERVAL_MS - elapsed));
        }
        lastProgressTime = Date.now();
        try {
            await tgAdapter.sendResponse({ text: `[build] ${update}` }, senderId);
        } catch (e) {
            console.error(`[build] Failed to send progress:`, e);
        }
    };

    const { prompt, preferredProjectName } = parseFullBuildMessage(message.text);

    try {
        console.info(
            `[build] Planning for sender=${senderId}: "${prompt.slice(0, 80)}"` +
                (preferredProjectName ? ` (repo: ${preferredProjectName})` : ""),
        );

        const plan = await generateProjectRoadmap(prompt, senderId, {
            preferredProjectName,
        });

        await sendUpdate(
            `Roadmap ready: ${plan.projectName} (${plan.techStack}). ${plan.steps.length} phases. Building in background...`,
        );

        // Phase 2: Execute all steps + push
        await executeProjectRoadmap(plan, senderId, sendUpdate);
    } catch (error) {
        const errMsg = error instanceof Error ? error.message : String(error);
        console.error(`[build] Fatal error for sender=${senderId}:`, error);
            try {
                await tgAdapter.sendResponse(
                    { text: `Build failed:\n${errMsg.slice(0, 500)}` },
                    senderId,
                );
            } catch { /* best effort */ }
    }
}

// ---------------------------------------------------------------------------
// Long-term memory extraction (background, non-blocking)
// ---------------------------------------------------------------------------

function triggerMemoryExtraction(senderId: string): void {
    const run = async () => {
        try {
            const shouldRun = await shouldRunExtraction(senderId);
            if (!shouldRun) return;
            console.info(`[memory] Triggering profile extraction for sender=${senderId}`);
            const history = await getChatHistory(senderId);
            if (history.length === 0) return;
            await updateUserProfile(senderId, history);
        } catch (error) {
            console.error(`[memory] Background extraction failed for sender=${senderId}:`, error);
        }
    };

    (async () => {
        try {
            const { waitUntil } = await import("@vercel/functions");
            waitUntil(run());
        } catch {
            void run();
        }
    })();
}

// ---------------------------------------------------------------------------
// Downstream message processing — wired to the LLM orchestrator
// ---------------------------------------------------------------------------

/**
 * Process an authorised, normalised message through the full LLM pipeline.
 * Normal (non-dev) messages only — dev requests are handled by runDevAgentInBackground.
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

        // Handle "Reject" buttons for staged patches and pushes.
        if (await handleRejectPatchAction(message)) {
            console.info(`[telegram/webhook] Reject-patch action handled for sender=${message.senderId}.`);
            return;
        }
        if (await handleRejectPushAction(message)) {
            console.info(`[telegram/webhook] Reject-push action handled for sender=${message.senderId}.`);
            return;
        }
        if (await handleRejectPushOnlyAction(message)) {
            console.info(`[telegram/webhook] Reject-push-only action handled for sender=${message.senderId}.`);
            return;
        }

        // "status" command: return progress of active dev job and/or research (both can run at once)
        const lowerText = message.text.trim().toLowerCase();
        if (lowerText === "status" || lowerText === "/status") {
            const activeJob = await getActiveJob(message.senderId);
            const activeResearch = await getActiveResearch(message.senderId);
            const parts: string[] = [];

            if (activeJob && (activeJob.status === "planning" || activeJob.status === "executing")) {
                const elapsed = Math.round((Date.now() - activeJob.started_at) / 1000);
                const lines = [
                    "**Build**",
                    `Job: ${activeJob.goal.slice(0, 80)}`,
                    `Status: ${activeJob.status}`,
                ];
                if (activeJob.phases && activeJob.phases.length > 0) {
                    const cp = activeJob.current_phase ?? 0;
                    lines.push(`Phase: ${cp}/${activeJob.total_phases ?? activeJob.phases.length}`);
                    for (const p of activeJob.phases) {
                        const icon = p.status === "complete" ? "done" : p.status === "executing" ? "..." : p.status === "failed" ? "FAIL" : "-";
                        lines.push(`  [${icon}] ${p.name}`);
                    }
                } else {
                    lines.push(`Step: ${activeJob.current_step}/${activeJob.total_steps}`);
                }
                lines.push(`Current: ${activeJob.current_action}`);
                lines.push(`Elapsed: ${elapsed}s`);
                if (activeJob.steps_completed.length > 0) {
                    lines.push(`Done: ${activeJob.steps_completed.join(", ")}`);
                }
                if (activeJob.errors.length > 0) {
                    lines.push(`Errors: ${activeJob.errors.slice(-2).join("; ")}`);
                }
                parts.push(lines.join("\n"));
            }
            if (activeResearch && !activeResearch.cancelled) {
                const elapsed = Math.round((Date.now() - activeResearch.started_at) / 1000);
                parts.push(`**Research** (${elapsed}s)\nTopic: ${activeResearch.topic.slice(0, 80)}\nSend "cancel" to stop.`);
            }
            if (parts.length === 0) {
                await getAdapter().sendResponse({ text: "No active build or research running." }, message.senderId);
            } else {
                await getAdapter().sendResponse({ text: parts.join("\n\n") }, message.senderId);
            }
            return;
        }

        // "cancel" command: cancel active dev job and/or research (cancels both if both running)
        if (lowerText === "cancel" || lowerText === "/cancel") {
            const activeJob = await getActiveJob(message.senderId);
            const activeResearch = await getActiveResearch(message.senderId);
            const cancelled: string[] = [];

            if (activeJob && (activeJob.status === "planning" || activeJob.status === "executing")) {
                await updateJob(activeJob.id, { status: "cancelled", finished_at: Date.now() });
                await clearActiveJob(message.senderId);
                cancelled.push(`Build: ${activeJob.goal.slice(0, 60)}`);
            }
            if (activeResearch && !activeResearch.cancelled) {
                await setResearchCancelled(message.senderId);
                cancelled.push("Research");
            }
            if (cancelled.length === 0) {
                await getAdapter().sendResponse({ text: "No active build or research to cancel." }, message.senderId);
            } else {
                await getAdapter().sendResponse(
                    { text: `Cancelled: ${cancelled.join(", ")}.\n(Note: file changes from the build remain.)` },
                    message.senderId,
                );
            }
            return;
        }

        // Full-project build: single-prompt autonomous build (intent from user input)
        if (isFullBuildIntent(message.text)) {
            const activeJob = await getActiveJob(message.senderId);
            if (activeJob && (activeJob.status === "planning" || activeJob.status === "executing")) {
                await getAdapter().sendResponse(
                    { text: `A build job is already running: "${activeJob.goal.slice(0, 60)}"\nSend "status" or "cancel" first.` },
                    message.senderId,
                );
                return;
            }
            // Allow starting a build even if research is running (they can run in parallel)

            const tgAdapter = getAdapter();
            await tgAdapter.sendResponse(
                {
                    text: "Building your project from your prompt…\n\n" +
                        "This runs in the background — you can keep chatting. I'll only message you for important milestones (roadmap ready, each phase done, push, done). Send \"status\" to check progress or \"cancel\" to abort.",
                },
                message.senderId,
            );

            const runBuild = () => runFullBuildInBackground(message, tgAdapter);
            try {
                const { waitUntil } = await import("@vercel/functions");
                waitUntil(runBuild());
            } catch {
                void runBuild();
            }
            return;
        }

        // Research requests: multi-agent pipeline (Researcher → Writer → Reviewer)
        if (isResearchRequest(message.text)) {
            const activeResearch = await getActiveResearch(message.senderId);
            if (activeResearch && !activeResearch.cancelled) {
                await getAdapter().sendResponse(
                    { text: `Research is already in progress: "${activeResearch.topic.slice(0, 60)}"\nSend "status" to check or "cancel" to stop.` },
                    message.senderId,
                );
                return;
            }
            // Allow starting research even if a build is running (they can run in parallel)

            const tgAdapter = getAdapter();
            await setActiveResearch(message.senderId, message.text);
            await tgAdapter.sendResponse(
                {
                    text: "Assembling the research team. I'll run in the background — you can keep chatting. I'll notify you when the paper is ready. Send \"status\" to check progress or \"cancel\" to stop.",
                },
                message.senderId,
            );

            const runResearch = () => runResearchInBackground(message, tgAdapter);
            try {
                const { waitUntil } = await import("@vercel/functions");
                waitUntil(runResearch());
            } catch {
                void runResearch();
            }
            return;
        }

        // Dev feature requests: check for active job or research, then fire off background agent.
        if (isDevFeatureRequest(message.text)) {
            const activeJob = await getActiveJob(message.senderId);
            const activeResearch = await getActiveResearch(message.senderId);
            if (activeJob && (activeJob.status === "planning" || activeJob.status === "executing")) {
                await getAdapter().sendResponse(
                    { text: `A dev job is already running: "${activeJob.goal.slice(0, 60)}"\nSend "status" to check progress or "cancel" to stop it.` },
                    message.senderId,
                );
                return;
            }
            if (activeResearch && !activeResearch.cancelled) {
                await getAdapter().sendResponse(
                    { text: `Research is still running: "${activeResearch.topic.slice(0, 60)}"\nSend "status" or "cancel" first.` },
                    message.senderId,
                );
                return;
            }

            const tgAdapter = getAdapter();
            await tgAdapter.sendResponse(
                { text: "Starting dev agent. I'll send you live updates as I work. You can keep chatting — I'll notify you when done." },
                message.senderId,
            );

            // Use waitUntil so Vercel keeps the process alive for the full dev agent run.
            try {
                const { waitUntil } = await import("@vercel/functions");
                waitUntil(runDevAgentInBackground(message, tgAdapter));
            } catch {
                // Not running on Vercel (local dev) — fire-and-forget is fine.
                void runDevAgentInBackground(message, tgAdapter);
            }
            return;
        }

        // Delegate to the orchestrator — it handles Redis history, LLM calls,
        // tool execution (RAG search, patch approval), and persistence.
        const { response } = await processUserMessage(message);

        // Send the final StandardResponse back to Telegram.
        await getAdapter().sendResponse(response, message.senderId);

        console.info(`[telegram/webhook] Reply sent to sender=${message.senderId}.`);

        // Long-term memory: every N messages, extract permanent facts in the background
        triggerMemoryExtraction(message.senderId);
    } catch (error) {
        // Log the error but do NOT re-throw — we've already returned 200 to
        // Telegram and cannot change that now.
        console.error(
            `[telegram/webhook] Error while processing message from sender=${message.senderId}:`,
            error
        );
    }
}
