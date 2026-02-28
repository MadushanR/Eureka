/**
 * lib/redis.ts — Upstash Redis Client & Chat History Helpers
 * ===========================================================
 * Provides a singleton Redis client and two chat-history operations:
 *
 *   getChatHistory(senderId)            — Retrieve the last N messages.
 *   saveChatMessage(senderId, message)  — Append a message and reset the TTL.
 *
 * Data model
 * ----------
 * Each conversation is stored as a Redis List under the key:
 *   chat:{senderId}
 *
 * Every element is a JSON-serialised ModelMessage (Vercel AI SDK v6 format):
 *   { role: "user" | "assistant" | "tool", content: string | ... }
 *
 * A 24-hour TTL is (re)set on every write so idle conversations are
 * automatically evicted and the context window never grows unbounded.
 *
 * Environment variables required
 * ------------------------------
 *   UPSTASH_REDIS_REST_URL   — From the Upstash console.
 *   UPSTASH_REDIS_REST_TOKEN — From the Upstash console.
 */

import { Redis } from "@upstash/redis";
import type { ModelMessage } from "ai";

// ---------------------------------------------------------------------------
// Redis client singleton
// ---------------------------------------------------------------------------

/**
 * Lazily-initialised Redis client.
 *
 * `Redis.fromEnv()` reads UPSTASH_REDIS_REST_URL and
 * UPSTASH_REDIS_REST_TOKEN from the environment automatically.
 * It throws at instantiation time if either variable is missing, which
 * surfaces misconfiguration at startup rather than at runtime.
 */
let _redis: Redis | null = null;

export function getRedisClient(): Redis {
    if (!_redis) {
        _redis = Redis.fromEnv();
    }
    return _redis;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Number of recent messages retrieved from Redis per request.
 * Keeping this value small limits token consumption and API costs.
 * Increase it if the LLM needs more context for multi-turn conversations.
 */
const HISTORY_WINDOW = 10;

/**
 * Time-to-live in seconds for each chat history key (24 hours).
 * Every write resets this TTL, so active conversations never expire mid-session.
 */
const CHAT_TTL_SECONDS = 24 * 60 * 60; // 86 400 s

/**
 * Time-to-live in seconds for a staged patch (1 hour).
 * Staged patches are short-lived artefacts used to bridge the gap between
 * LLM-generated patches and user-approved application via Telegram buttons.
 */
const PATCH_TTL_SECONDS = 60 * 60; // 3 600 s

/**
 * Build the Redis key for a given sender's chat history.
 * Namespacing under `chat:` avoids collisions with other key types.
 */
const chatKey = (senderId: string): string => `chat:${senderId}`;

/**
 * Build the Redis key used to temporarily store a pending patch.
 */
const patchKey = (patchId: string): string => `patch:${patchId}`;

/**
 * Build the Redis key used to temporarily store a pending push (commit+push) request.
 */
const pushKey = (pushId: string): string => `push:${pushId}`;

/**
 * Build the Redis key used to temporarily store a pending commit request.
 */
const commitKey = (commitId: string): string => `commit:${commitId}`;

/**
 * Build the Redis key used to temporarily store a pending push-only request.
 */
const pushOnlyKey = (pushOnlyId: string): string => `pushonly:${pushOnlyId}`;

const jobKey = (jobId: string): string => `job:${jobId}`;
const activeJobKey = (senderId: string): string => `activejob:${senderId}`;
const JOB_TTL_SECONDS = 2 * 60 * 60; // 2 hours

// ---------------------------------------------------------------------------
// Chat history helpers
// ---------------------------------------------------------------------------

/**
 * Retrieve the most recent `HISTORY_WINDOW` messages for a conversation.
 *
 * Redis LRANGE returns elements in insertion order (left → right), so we
 * take the last N elements from the tail of the list using negative indices.
 *
 * Returns an empty array if no history exists yet (new conversation) or if
 * the Redis connection fails — we always prefer a degraded (context-free)
 * response over a hard failure to the user.
 *
 * @param senderId - The platform-scoped conversation identifier.
 * @returns An ordered array of `ModelMessage` objects, oldest first.
 */
export async function getChatHistory(senderId: string): Promise<ModelMessage[]> {
    try {
        const redis = getRedisClient();
        const key = chatKey(senderId);

        // LRANGE key -N -1 returns the last N elements.
        // @upstash/redis automatically deserialises JSON strings to objects.
        const raw = await redis.lrange<ModelMessage>(key, -HISTORY_WINDOW, -1);

        return raw ?? [];
    } catch (error) {
        // A Redis failure should degrade gracefully — the LLM can still respond,
        // just without prior context.
        console.error(
            `[redis] getChatHistory failed for sender="${senderId}":`,
            error instanceof Error ? error.message : error
        );
        return [];
    }
}

/**
 * Append a single message to the tail of a conversation's history list and
 * reset the 24-hour TTL.
 *
 * Uses a Redis pipeline (RPUSH + EXPIRE in a single round-trip) for
 * efficiency and atomicity.
 *
 * @param senderId - The platform-scoped conversation identifier.
 * @param message  - A `ModelMessage` (Vercel AI SDK v6) to persist.
 */
export async function saveChatMessage(
    senderId: string,
    message: ModelMessage
): Promise<void> {
    try {
        const redis = getRedisClient();
        const key = chatKey(senderId);

        // Pipeline: RPUSH (append to list) + EXPIRE (reset TTL) in one round-trip.
        const pipeline = redis.pipeline();
        pipeline.rpush(key, message);
        pipeline.expire(key, CHAT_TTL_SECONDS);
        await pipeline.exec();
    } catch (error) {
        // Log but don't throw — failure to persist history is not fatal for the
        // current response, just degrades future context.
        console.error(
            `[redis] saveChatMessage failed for sender="${senderId}":`,
            error instanceof Error ? error.message : error
        );
    }
}

/**
 * Append multiple messages atomically in a single pipeline.
 * Useful for saving the user message and assistant reply together after
 * an LLM turn completes, ensuring the list is always in a consistent state.
 *
 * @param senderId - The platform-scoped conversation identifier.
 * @param messages - Ordered array of `ModelMessage` objects to append.
 */
export async function saveChatMessages(
    senderId: string,
    messages: ModelMessage[]
): Promise<void> {
    if (messages.length === 0) return;

    try {
        const redis = getRedisClient();
        const key = chatKey(senderId);

        const pipeline = redis.pipeline();
        for (const msg of messages) {
            pipeline.rpush(key, msg);
        }
        pipeline.expire(key, CHAT_TTL_SECONDS);
        await pipeline.exec();
    } catch (error) {
        console.error(
            `[redis] saveChatMessages failed for sender="${senderId}":`,
            error instanceof Error ? error.message : error
        );
    }
}

/**
 * Delete the entire chat history for a conversation.
 * Useful for implementing a "reset" command.
 *
 * @param senderId - The platform-scoped conversation identifier.
 */
export async function clearChatHistory(senderId: string): Promise<void> {
    try {
        await getRedisClient().del(chatKey(senderId));
    } catch (error) {
        console.error(
            `[redis] clearChatHistory failed for sender="${senderId}":`,
            error instanceof Error ? error.message : error
        );
    }
}

// ---------------------------------------------------------------------------
// Patch staging helpers
// ---------------------------------------------------------------------------

/** Staged patch data for apply-patch flow. */
export interface StagedPatch {
    patch_string: string;
    workspace_path?: string;
}

/**
 * Stage a patch (and optional workspace path) in Redis and return a short
 * unique identifier that can be safely embedded in a Telegram callback action.
 *
 * The value is stored under `patch:{id}` as JSON with a 1-hour TTL.
 *
 * @param patchString   - Unified diff or patch text generated by the LLM.
 * @param workspacePath - Optional absolute path to the git repo (must be in daemon ALLOWED_WORKSPACES).
 * @returns The generated patch identifier.
 */
export async function stagePatch(
    patchString: string,
    workspacePath?: string
): Promise<string> {
    const redis = getRedisClient();

    const patchId = Math.random().toString(36).slice(2, 10);
    const key = patchKey(patchId);
    const value: StagedPatch = { patch_string: patchString };
    if (workspacePath != null && workspacePath.trim() !== "") {
        value.workspace_path = workspacePath.trim();
    }

    try {
        await redis.set(key, JSON.stringify(value), { ex: PATCH_TTL_SECONDS });
    } catch (error) {
        console.error(
            `[redis] stagePatch failed for patchId="${patchId}":`,
            error instanceof Error ? error.message : error
        );
    }

    return patchId;
}

/**
 * Retrieve a previously staged patch by its identifier.
 *
 * Returns `null` if the patch does not exist or has expired.
 * Handles both string (JSON) and already-parsed object from Redis.
 *
 * @param patchId - Identifier returned by `stagePatch`.
 */
export async function getPendingPatch(patchId: string): Promise<StagedPatch | null> {
    try {
        const redis = getRedisClient();
        const key = patchKey(patchId);
        const value = await redis.get<string | StagedPatch | null>(key);
        if (value == null) return null;
        const parsed: StagedPatch =
            typeof value === "object" && value !== null && "patch_string" in value
                ? (value as StagedPatch)
                : (JSON.parse(value as string) as StagedPatch);
        if (typeof parsed.patch_string !== "string") return null;
        return parsed;
    } catch (error) {
        console.error(
            `[redis] getPendingPatch failed for patchId="${patchId}":`,
            error instanceof Error ? error.message : error
        );
        return null;
    }
}

// ---------------------------------------------------------------------------
// Push staging (commit-and-push approval flow)
// ---------------------------------------------------------------------------

/** Staged push request for "Approve & Push" flow. */
export interface StagedPush {
    workspace_path: string;
    commit_message: string;
}

/**
 * Stage a push request (workspace path + commit message) and return an id for the approval button.
 */
export async function stagePush(
    workspacePath: string,
    commitMessage: string
): Promise<string> {
    const redis = getRedisClient();
    const pushId = Math.random().toString(36).slice(2, 10);
    const key = pushKey(pushId);
    const value: StagedPush = {
        workspace_path: workspacePath.trim(),
        commit_message: commitMessage.trim() || "Update from Eureka",
    };
    try {
        await redis.set(key, JSON.stringify(value), { ex: PATCH_TTL_SECONDS });
    } catch (error) {
        console.error(
            `[redis] stagePush failed for pushId="${pushId}":`,
            error instanceof Error ? error.message : error
        );
    }
    return pushId;
}

/**
 * Retrieve a staged push request by id. Returns null if expired or not found.
 * Handles both string (JSON) and already-parsed object from Redis.
 */
export async function getPendingPush(pushId: string): Promise<StagedPush | null> {
    try {
        const redis = getRedisClient();
        const key = pushKey(pushId);
        const value = await redis.get<string | StagedPush | null>(key);
        if (value == null) return null;
        const parsed: StagedPush =
            typeof value === "object" && value !== null && "workspace_path" in value
                ? (value as StagedPush)
                : (JSON.parse(value as string) as StagedPush);
        if (typeof parsed.workspace_path !== "string") return null;
        return parsed;
    } catch (error) {
        console.error(
            `[redis] getPendingPush failed for pushId="${pushId}":`,
            error instanceof Error ? error.message : error
        );
        return null;
    }
}

// ---------------------------------------------------------------------------
// Push-only staging (push without committing)
// ---------------------------------------------------------------------------

/** Staged push-only request for a pure `git push` flow. */
export interface StagedPushOnly {
    workspace_path: string;
}

/**
 * Stage a push-only request (workspace path) and return an id for the approval button.
 */
export async function stagePushOnly(workspacePath: string): Promise<string> {
    const redis = getRedisClient();
    const pushOnlyId = Math.random().toString(36).slice(2, 10);
    const key = pushOnlyKey(pushOnlyId);
    const value: StagedPushOnly = {
        workspace_path: workspacePath.trim(),
    };
    try {
        await redis.set(key, JSON.stringify(value), { ex: PATCH_TTL_SECONDS });
    } catch (error) {
        console.error(
            `[redis] stagePushOnly failed for pushOnlyId="${pushOnlyId}":`,
            error instanceof Error ? error.message : error
        );
    }
    return pushOnlyId;
}

/**
 * Retrieve a staged push-only request by id. Returns null if expired or not found.
 * Handles both string (JSON) and already-parsed object from Redis.
 */
export async function getPendingPushOnly(pushOnlyId: string): Promise<StagedPushOnly | null> {
    try {
        const redis = getRedisClient();
        const key = pushOnlyKey(pushOnlyId);
        const value = await redis.get<string | StagedPushOnly | null>(key);
        if (value == null) return null;
        const parsed: StagedPushOnly =
            typeof value === "object" && value !== null && "workspace_path" in value
                ? (value as StagedPushOnly)
                : (JSON.parse(value as string) as StagedPushOnly);
        if (typeof parsed.workspace_path !== "string") return null;
        return parsed;
    } catch (error) {
        console.error(
            `[redis] getPendingPushOnly failed for pushOnlyId="${pushOnlyId}":`,
            error instanceof Error ? error.message : error
        );
        return null;
    }
}

// ---------------------------------------------------------------------------
// Job manager (dev-agent job tracking)
// ---------------------------------------------------------------------------

export interface DevJobPhase {
    name: string;
    description: string;
    files: string[];
    status: "pending" | "executing" | "complete" | "failed";
}

export interface DevJob {
    id: string;
    sender_id: string;
    goal: string;
    status: "planning" | "executing" | "complete" | "failed" | "cancelled";
    plan: string[];
    current_step: number;
    total_steps: number;
    current_action: string;
    steps_completed: string[];
    errors: string[];
    files_created: string[];
    started_at: number;
    finished_at?: number;
    /** Multi-phase fields */
    phases?: DevJobPhase[];
    current_phase?: number;
    total_phases?: number;
}

export async function createJob(senderId: string, goal: string): Promise<DevJob> {
    const redis = getRedisClient();
    const id = Math.random().toString(36).slice(2, 10);
    const job: DevJob = {
        id,
        sender_id: senderId,
        goal,
        status: "planning",
        plan: [],
        current_step: 0,
        total_steps: 0,
        current_action: "Planning...",
        steps_completed: [],
        errors: [],
        files_created: [],
        started_at: Date.now(),
    };
    try {
        await redis.set(jobKey(id), JSON.stringify(job), { ex: JOB_TTL_SECONDS });
        await redis.set(activeJobKey(senderId), id, { ex: JOB_TTL_SECONDS });
    } catch (error) {
        console.error(`[redis] createJob failed:`, error instanceof Error ? error.message : error);
    }
    return job;
}

export async function getJob(jobId: string): Promise<DevJob | null> {
    try {
        const redis = getRedisClient();
        const value = await redis.get<string | DevJob | null>(jobKey(jobId));
        if (value == null) return null;
        return typeof value === "object" ? value as DevJob : JSON.parse(value as string) as DevJob;
    } catch (error) {
        console.error(`[redis] getJob failed:`, error instanceof Error ? error.message : error);
        return null;
    }
}

export async function updateJob(jobId: string, updates: Partial<DevJob>): Promise<void> {
    try {
        const existing = await getJob(jobId);
        if (!existing) return;
        const updated = { ...existing, ...updates };
        await getRedisClient().set(jobKey(jobId), JSON.stringify(updated), { ex: JOB_TTL_SECONDS });
    } catch (error) {
        console.error(`[redis] updateJob failed:`, error instanceof Error ? error.message : error);
    }
}

export async function getActiveJob(senderId: string): Promise<DevJob | null> {
    try {
        const redis = getRedisClient();
        const jobId = await redis.get<string>(activeJobKey(senderId));
        if (!jobId) return null;
        return getJob(jobId);
    } catch (error) {
        console.error(`[redis] getActiveJob failed:`, error instanceof Error ? error.message : error);
        return null;
    }
}

export async function clearActiveJob(senderId: string): Promise<void> {
    try {
        await getRedisClient().del(activeJobKey(senderId));
    } catch (error) {
        console.error(`[redis] clearActiveJob failed:`, error instanceof Error ? error.message : error);
    }
}

// ---------------------------------------------------------------------------
// Active research job (for status/cancel while research runs in background)
// ---------------------------------------------------------------------------

export interface ActiveResearch {
    topic: string;
    started_at: number;
    cancelled?: boolean;
}

const activeResearchKey = (senderId: string): string => `activeresearch:${senderId}`;

export async function setActiveResearch(senderId: string, topic: string): Promise<void> {
    try {
        const data: ActiveResearch = { topic, started_at: Date.now() };
        await getRedisClient().set(activeResearchKey(senderId), JSON.stringify(data), { ex: JOB_TTL_SECONDS });
    } catch (error) {
        console.error(`[redis] setActiveResearch failed:`, error instanceof Error ? error.message : error);
    }
}

export async function getActiveResearch(senderId: string): Promise<ActiveResearch | null> {
    try {
        const raw = await getRedisClient().get<string>(activeResearchKey(senderId));
        if (!raw) return null;
        return JSON.parse(raw) as ActiveResearch;
    } catch (error) {
        console.error(`[redis] getActiveResearch failed:`, error instanceof Error ? error.message : error);
        return null;
    }
}

export async function clearActiveResearch(senderId: string): Promise<void> {
    try {
        await getRedisClient().del(activeResearchKey(senderId));
    } catch (error) {
        console.error(`[redis] clearActiveResearch failed:`, error instanceof Error ? error.message : error);
    }
}

export async function setResearchCancelled(senderId: string): Promise<void> {
    try {
        const existing = await getActiveResearch(senderId);
        if (existing) {
            existing.cancelled = true;
            await getRedisClient().set(activeResearchKey(senderId), JSON.stringify(existing), { ex: JOB_TTL_SECONDS });
        }
    } catch (error) {
        console.error(`[redis] setResearchCancelled failed:`, error instanceof Error ? error.message : error);
    }
}
