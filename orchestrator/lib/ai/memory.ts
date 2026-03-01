/**
 * lib/ai/memory.ts — Long-Term Memory / Fact Extraction Engine
 * =============================================================
 * Periodically extracts permanent facts about the developer's infrastructure,
 * repos, and coding preferences from recent chat messages.  The extracted
 * profile is stored in Upstash Redis and injected into the system prompt so
 * the LLM always knows the user's stack without being told again.
 *
 * Public API:
 *   updateUserProfile(userId, recentMessages)  — extract & merge facts
 *   getUserProfile(userId)                     — read the stored profile
 *   shouldRunExtraction(userId)                — probabilistic gate
 */

import { generateObject } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import type { ModelMessage } from "ai";
import { getRedisClient } from "@/lib/redis";
import { withTelemetry } from "./telemetry";

// ---------------------------------------------------------------------------
// Zod schema — the canonical shape of a developer's long-term profile
// ---------------------------------------------------------------------------

export const UserProfileSchema = z.object({
    preferredLanguages: z
        .array(z.string())
        .describe("Programming languages the user prefers or uses most (e.g. TypeScript, Python, Rust)."),
    frameworks: z
        .array(z.string())
        .describe("Frameworks / libraries the user relies on (e.g. Next.js, FastAPI, TailwindCSS)."),
    activeRepositories: z
        .array(z.string())
        .describe("Names or URLs of repos the user is actively working on."),
    awsRegion: z
        .string()
        .default("")
        .describe("Preferred AWS region (e.g. us-east-1). Use empty string if unknown."),
    cloudProviders: z
        .array(z.string())
        .describe("Cloud/infra providers the user uses (e.g. AWS, Vercel, Cloudflare, GCP)."),
    databases: z
        .array(z.string())
        .describe("Databases the user works with (e.g. PostgreSQL, Redis, DynamoDB)."),
    os: z
        .string()
        .default("")
        .describe("Operating system (e.g. Windows 11, macOS Sonoma). Use empty string if unknown."),
    editor: z
        .string()
        .default("")
        .describe("Primary code editor (e.g. Cursor, VS Code). Use empty string if unknown."),
    deploymentTargets: z
        .array(z.string())
        .describe("Where the user deploys apps (e.g. Vercel, Docker, EC2)."),
    customPreferences: z
        .array(z.string())
        .describe("Any other permanent technical preferences, conventions, or constraints the user has mentioned."),
});

export type UserProfile = z.infer<typeof UserProfileSchema>;

// ---------------------------------------------------------------------------
// Redis key helpers
// ---------------------------------------------------------------------------

const profileKey = (userId: string): string => `user_profile:${userId}`;
const counterKey = (userId: string): string => `msg_counter:${userId}`;

const PROFILE_TTL_SECONDS = 30 * 24 * 60 * 60; // 30 days
const EXTRACTION_INTERVAL = 5; // run extraction every N messages

const MEMORY_MODEL = process.env.LLM_MODEL_NAME_MEMORY || "gpt-4.1-mini";

// ---------------------------------------------------------------------------
// Read profile from Redis
// ---------------------------------------------------------------------------

const EMPTY_PROFILE: UserProfile = {
    preferredLanguages: [],
    frameworks: [],
    activeRepositories: [],
    awsRegion: "",
    cloudProviders: [],
    databases: [],
    os: "",
    editor: "",
    deploymentTargets: [],
    customPreferences: [],
};

export async function getUserProfile(userId: string): Promise<UserProfile> {
    try {
        const redis = getRedisClient();
        const raw = await redis.get<string | UserProfile | null>(profileKey(userId));
        if (!raw) return { ...EMPTY_PROFILE };

        const parsed: unknown = typeof raw === "string" ? JSON.parse(raw) : raw;
        const result = UserProfileSchema.safeParse(parsed);
        return result.success ? result.data : { ...EMPTY_PROFILE };
    } catch (error) {
        console.error(`[memory] getUserProfile failed for userId="${userId}":`, error);
        return { ...EMPTY_PROFILE };
    }
}

// ---------------------------------------------------------------------------
// Counter-based gate — should we run extraction on this turn?
// ---------------------------------------------------------------------------

export async function shouldRunExtraction(userId: string): Promise<boolean> {
    try {
        const redis = getRedisClient();
        const count = await redis.incr(counterKey(userId));
        await redis.expire(counterKey(userId), PROFILE_TTL_SECONDS);
        return count % EXTRACTION_INTERVAL === 0;
    } catch (error) {
        console.error(`[memory] shouldRunExtraction counter failed:`, error);
        return false;
    }
}

// ---------------------------------------------------------------------------
// The extraction engine
// ---------------------------------------------------------------------------

const EXTRACTION_SYSTEM_PROMPT =
    "You are a fact-extraction engine. You receive a developer's recent chat messages " +
    "and their existing profile (which may be mostly empty). " +
    "Your job is to extract NEW, PERMANENT technical facts — such as their preferred " +
    "languages, frameworks, cloud providers, deployment targets, operating system, " +
    "active repos, and coding conventions — and MERGE them into the existing profile.\n\n" +
    "Rules:\n" +
    "- Only add facts that are clearly stated or strongly implied. Do NOT guess.\n" +
    "- Preserve all existing profile data unless the user explicitly contradicts it.\n" +
    "- Do NOT add transient information (e.g. current bug they're debugging, file they're reading).\n" +
    "- De-duplicate arrays — no repeated entries.\n" +
    "- If there are no new facts to extract, return the existing profile unchanged.\n" +
    "- Return a complete, merged profile object — not a diff.";

export async function updateUserProfile(
    userId: string,
    recentMessages: ModelMessage[],
): Promise<UserProfile> {
    const existing = await getUserProfile(userId);

    const messagesText = recentMessages
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => {
            const content = typeof m.content === "string"
                ? m.content
                : JSON.stringify(m.content);
            return `[${m.role}]: ${content}`;
        })
        .join("\n")
        .slice(0, 6000);

    if (!messagesText.trim()) return existing;

    try {
        const { object: updated } = await generateObject({
            model: openai(MEMORY_MODEL),
            schema: UserProfileSchema,
            messages: [
                { role: "system", content: EXTRACTION_SYSTEM_PROMPT },
                {
                    role: "user",
                    content:
                        `EXISTING PROFILE:\n${JSON.stringify(existing, null, 2)}\n\n` +
                        `RECENT MESSAGES:\n${messagesText}\n\n` +
                        `Return the updated profile JSON.`,
                },
            ],
            ...withTelemetry("memory.extractProfile"),
        } as Parameters<typeof generateObject>[0]);

        const redis = getRedisClient();
        await redis.set(profileKey(userId), JSON.stringify(updated), { ex: PROFILE_TTL_SECONDS });

        console.info(`[memory] Profile updated for userId="${userId}"`);
        return updated;
    } catch (error) {
        console.error(`[memory] updateUserProfile failed for userId="${userId}":`, error);
        return existing;
    }
}

// ---------------------------------------------------------------------------
// Format profile for system prompt injection
// ---------------------------------------------------------------------------

export function formatProfileForPrompt(profile: UserProfile): string {
    const lines: string[] = [];

    if (profile.preferredLanguages.length > 0)
        lines.push(`Languages: ${profile.preferredLanguages.join(", ")}`);
    if (profile.frameworks.length > 0)
        lines.push(`Frameworks: ${profile.frameworks.join(", ")}`);
    if (profile.activeRepositories.length > 0)
        lines.push(`Active repos: ${profile.activeRepositories.join(", ")}`);
    if (profile.awsRegion)
        lines.push(`AWS region: ${profile.awsRegion}`);
    if (profile.cloudProviders.length > 0)
        lines.push(`Cloud: ${profile.cloudProviders.join(", ")}`);
    if (profile.databases.length > 0)
        lines.push(`Databases: ${profile.databases.join(", ")}`);
    if (profile.os)
        lines.push(`OS: ${profile.os}`);
    if (profile.editor)
        lines.push(`Editor: ${profile.editor}`);
    if (profile.deploymentTargets.length > 0)
        lines.push(`Deploys to: ${profile.deploymentTargets.join(", ")}`);
    if (profile.customPreferences.length > 0)
        lines.push(`Preferences: ${profile.customPreferences.join("; ")}`);

    return lines.length > 0
        ? `\n\n--- Developer Profile (long-term memory) ---\n${lines.join("\n")}\n--- End Profile ---`
        : "";
}
