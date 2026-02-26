/**
 * lib/ai/orchestrator.ts — LLM Orchestration Loop
 * ===============================================
 * Wires together:
 *   - Redis chat state (Upstash)
 *   - Vercel AI SDK (`generateText`)
 *   - LLM tools (`search_local_codebase`, `request_patch_approval`)
 *
 * Entry point:
 *   processUserMessage(message: StandardMessage)
 *
 * This function:
 *   1. Loads recent chat history for `message.senderId` from Redis.
 *   2. Appends the new user message.
 *   3. Calls `generateText` with tools enabled.
 *   4. Persists the user + assistant messages back to Redis.
 *   5. Returns a `StandardResponse` suitable for all adapters.
 */

import { generateText, type ModelMessage } from "ai";
import { openai } from "@ai-sdk/openai";
import type { StandardMessage, StandardResponse } from "@/types/messaging";
import { getChatHistory, saveChatMessages } from "@/lib/redis";
import { aiTools } from "./tools";

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

/**
 * Resolve the model to use for this orchestrator.
 *
 * You can override the default by setting LLM_MODEL_NAME in the environment,
 * e.g. "gpt-4.1" or "gpt-4.1-mini".
 */
const MODEL_NAME = process.env.LLM_MODEL_NAME || "gpt-4.1-mini";

const model = openai(MODEL_NAME);

// ---------------------------------------------------------------------------
// Type guards
// ---------------------------------------------------------------------------

function isStandardResponse(value: unknown): value is StandardResponse {
    if (!value || typeof value !== "object") return false;
    const maybe = value as Partial<StandardResponse>;
    return typeof maybe.text === "string";
}

/**
 * Turn a tool result (e.g. list_workspace_folders, search) into a short
 * user-facing message when the model didn't output any final text.
 */
function formatToolResultForUser(value: unknown, toolName?: string): string {
    if (value === null || value === undefined) {
        return "I ran a tool but got no result to show.";
    }
    const obj = value as Record<string, unknown>;

    // list_workspace_folders: { workspaces: [ { workspace_path, folders } ] }
    if (Array.isArray(obj.workspaces)) {
        const lines: string[] = ["Here are the workspaces and their top-level folders:"];
        for (const w of obj.workspaces as Array<{ workspace_path?: string; folders?: string[] }>) {
            const path = w.workspace_path ?? "(unknown path)";
            const folders = Array.isArray(w.folders) ? w.folders.join(", ") : "(none)";
            lines.push(`• ${path}`);
            lines.push(`  Folders: ${folders}`);
        }
        return lines.join("\n");
    }

    // delete_path / Spotify: { success, message } or { success: false, error }
    if (typeof obj.success === "boolean") {
        if (obj.success && typeof obj.message === "string") {
            // Spotify status can include track info
            if (obj.track != null || obj.is_playing != null) {
                const parts: string[] = [obj.message];
                if (typeof obj.track === "string") {
                    parts.push(`Track: ${obj.artist ? `${obj.artist} — ` : ""}${obj.track}`);
                    if (obj.album) parts.push(`Album: ${obj.album}`);
                }
                if (typeof obj.device === "string") parts.push(`Device: ${obj.device}`);
                return parts.join("\n");
            }
            return obj.message;
        }
        if (!obj.success && typeof obj.error === "string") {
            return `Failed: ${obj.error}`;
        }
        return obj.success ? "Done." : "Failed.";
    }

    // list_folder_contents: { folder_path, entries: [ { name, type } ] }
    if (typeof obj.folder_path === "string" && Array.isArray(obj.entries)) {
        const entries = obj.entries as Array<{ name?: string; type?: string }>;
        const lines: string[] = [`Here are the items inside ${obj.folder_path}:`];
        if (entries.length === 0) {
            lines.push("  (folder is empty)");
        } else {
            for (const e of entries.slice(0, 50)) {
                const name = e.name ?? "(unnamed)";
                const kind =
                    e.type === "folder"
                        ? "folder"
                        : e.type === "file"
                          ? "file"
                          : "item";
                lines.push(`• ${name} (${kind})`);
            }
            if (entries.length > 50) {
                lines.push(`… and ${entries.length - 50} more.`);
            }
        }
        return lines.join("\n");
    }

    // search / RAG: { query, results: [ { file_path, text, ... } ] } or { error }
    if (typeof obj.error === "string") {
        return `The search couldn't complete: ${obj.error}`;
    }
    if (Array.isArray(obj.results) && (obj.results as unknown[]).length > 0) {
        const results = obj.results as Array<{ file_path?: string; text?: string; start_line?: number; end_line?: number }>;
        const lines: string[] = [`Found ${results.length} result(s) for "${obj.query ?? "your query"}":`];
        for (let i = 0; i < Math.min(results.length, 5); i++) {
            const r = results[i];
            const loc = r.file_path && r.start_line != null ? `${r.file_path}:${r.start_line}-${r.end_line ?? "?"}` : r.file_path ?? "?";
            const preview = typeof r.text === "string" ? r.text.slice(0, 120).replace(/\n/g, " ") + (r.text.length > 120 ? "…" : "") : "";
            lines.push(`• ${loc}`);
            lines.push(`  ${preview}`);
        }
        if (results.length > 5) lines.push(`… and ${results.length - 5} more.`);
        return lines.join("\n");
    }
    if (Array.isArray(obj.results) && (obj.results as unknown[]).length === 0) {
        return `No code snippets matched "${obj.query ?? "your query"}".`;
    }

    // list_git_repos: { repos: [ { name, path } ] }
    if (Array.isArray(obj.repos)) {
        const repos = obj.repos as Array<{ name?: string; path?: string }>;
        if (repos.length === 0) {
            return "I couldn't find any git repositories under the allowed workspaces.";
        }
        const lines: string[] = ["Here are the git repositories I found:"];
        for (const r of repos.slice(0, 20)) {
            const name = r.name || "(unnamed)";
            const path = r.path || "(unknown path)";
            lines.push(`• ${name}: ${path}`);
        }
        if (repos.length > 20) {
            lines.push(`… and ${repos.length - 20} more.`);
        }
        return lines.join("\n");
    }

    // Fallback: show a compact JSON summary (avoid huge payloads)
    try {
        const str = JSON.stringify(value);
        if (str.length > 800) return "I got a long result from the tool; here’s a summary:\n" + str.slice(0, 800) + "…";
        return "Here’s what I found:\n" + str;
    } catch {
        return "I ran a tool and got a result, but it couldn’t be formatted.";
    }
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

interface ProcessUserMessageResult {
    response: StandardResponse;
}

/**
 * Primary orchestration entry point for a single user message.
 *
 * @param message - Platform-agnostic inbound message.
 * @returns An object containing the final `StandardResponse` to send.
 */
export async function processUserMessage(
    message: StandardMessage,
): Promise<ProcessUserMessageResult> {
    const { senderId, text } = message;

    // -----------------------------------------------------------------------
    // 1. Load existing chat history (best-effort; degrades gracefully).
    // -----------------------------------------------------------------------
    const history: ModelMessage[] = await getChatHistory(senderId);

    // Current user turn as a ModelMessage (Vercel AI SDK core format).
    const userMessage: ModelMessage = {
        role: "user",
        content: text,
    };

    // Prepend system instruction so the model always replies after using tools.
    const systemMessage: ModelMessage = {
        role: "system",
        content:
            "You are a helpful coding assistant. When you use a tool (search, list folders, etc.), " +
            "always end with a short reply to the user summarizing what you did or found. Never leave the user without a text response. " +
            "The 'Approve & Apply' button is only for applying code patches (unified diff / git apply). " +
            "Do NOT use it for moving files or other non-code tasks. " +
            "For deleting a file or folder, use the delete_path tool with the full absolute path. " +
            "When the user wants to work with a git repo (e.g. 'Eureka'), first call list_git_repos to discover all repos under the allowed workspaces " +
            "and resolve the correct repo path (for example, a repo named 'Eureka' will usually live at C:\\Users\\madus\\Desktop\\Eureka). " +
            "When the user wants to push: first call prepare_push_approval(workspace_path) without a commit message to show the diff and ask for a commit message. " +
            "When they reply with a message (or 'default'), call prepare_push_approval(workspace_path, commit_message) with their reply to show the Approve & Push button.",
    };

    const messages: ModelMessage[] = [systemMessage, ...history, userMessage];

    // -----------------------------------------------------------------------
    // 2. Call the LLM with tools enabled.
    // -----------------------------------------------------------------------
    let finalText = "I'm sorry, I wasn't able to generate a response.";
    let toolResponse: StandardResponse | null = null;

    try {
        const result = await generateText({
            model,
            messages,
            tools: aiTools,
            maxSteps: 5,
        });

        // Debug: log result shape when text is empty (remove once "no response" is fixed)
        if (typeof result.text !== "string" || result.text.trim().length === 0) {
            const shape = {
                hasText: typeof result.text,
                textLength: typeof result.text === "string" ? result.text.length : 0,
                toolResultsLength: Array.isArray(result.toolResults) ? result.toolResults.length : "none",
                stepsLength: Array.isArray((result as { steps?: unknown[] }).steps)
                    ? (result as { steps: unknown[] }).steps.length
                    : "none",
                resultKeys: Object.keys(result),
            };
            console.warn(
                "[orchestrator] generateText returned empty text. Result shape:",
                JSON.stringify(shape),
            );

            // Extra debug: log the first tool result and first step in a compact form.
            try {
                const firstTool = Array.isArray(result.toolResults)
                    ? result.toolResults[0]
                    : null;
                const stepsDebug = (result as { steps?: unknown[] }).steps;
                const firstStep =
                    Array.isArray(stepsDebug) && stepsDebug.length > 0
                        ? stepsDebug[0]
                        : null;

                console.warn(
                    "[orchestrator] first toolResults[0]:",
                    JSON.stringify(firstTool),
                );
                console.warn(
                    "[orchestrator] first steps[0]:",
                    JSON.stringify(firstStep),
                );
            } catch (debugError) {
                console.warn(
                    "[orchestrator] failed to log detailed result debug:",
                    debugError,
                );
            }
        }

        // The primary natural-language reply from the assistant.
        if (typeof result.text === "string" && result.text.trim().length > 0) {
            finalText = result.text.trim();
        }

        // Collect tool results from whatever shape the SDK returned. In AI SDK v6,
        // tool results are exposed on `result.toolResults`, and each entry has an
        // `output` field (for normal tools) or `result` (for some abstractions).
        const allToolResults: Array<{ result?: unknown; output?: unknown; toolName?: string }> = [];
        if (Array.isArray(result.toolResults)) {
            for (const tr of result.toolResults) {
                const t = tr as { result?: unknown; output?: unknown; toolName?: string };
                allToolResults.push(t);
            }
        }

        // If any tool returned a StandardResponse (e.g. request_patch_approval), use it.
        for (const toolResult of allToolResults) {
            const value = toolResult.result ?? toolResult.output;
            if (isStandardResponse(value)) {
                toolResponse = value;
                break;
            }
        }

        // When the model left result.text empty but we have tool output, show it.
        if (finalText === "I'm sorry, I wasn't able to generate a response." && allToolResults.length > 0) {
            const last = allToolResults[allToolResults.length - 1];
            const value = (last?.result ?? last?.output);
            if (value !== undefined && value !== null) {
                finalText = formatToolResultForUser(value, last.toolName);
            }
        }
    } catch (error) {
        console.error(
            `[orchestrator] generateText failed for sender="${senderId}":`,
            error,
        );
        finalText =
            "I ran into an error while talking to the language model. " +
            "Please try again in a moment.";
    }

    // -----------------------------------------------------------------------
    // 3. Persist user + assistant messages back to Redis.
    // -----------------------------------------------------------------------
    const assistantMessage: ModelMessage = {
        role: "assistant",
        content: finalText,
    };

    // Best-effort persistence; errors are logged inside saveChatMessages.
    await saveChatMessages(senderId, [userMessage, assistantMessage]);

    // -----------------------------------------------------------------------
    // 4. Construct the StandardResponse for the adapter.
    // -----------------------------------------------------------------------
    const response: StandardResponse =
        toolResponse ??
        ({
            text: finalText,
        } satisfies StandardResponse);

    return { response };
}

