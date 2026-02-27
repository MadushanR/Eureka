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

    // read_file: { success, path, content, error } — check before generic success
    if (typeof obj.content === "string" && obj.success === true) {
        const path = typeof obj.path === "string" ? obj.path : "file";
        const preview = (obj.content as string).slice(0, 300);
        return `Read ${path} (${(obj.content as string).length} chars):\n${preview}${(obj.content as string).length > 300 ? "…" : ""}`;
    }
    if (obj.success === false && typeof obj.path === "string" && typeof obj.error === "string") {
        return `Read file failed (${obj.path}): ${obj.error}`;
    }

    // run_tests: { success, stdout, stderr, exit_code, error } — check before generic success
    if (typeof obj.exit_code === "number" && (typeof obj.stdout === "string" || typeof obj.stderr === "string")) {
        const code = obj.exit_code;
        const out = typeof obj.stdout === "string" ? obj.stdout.trim().slice(0, 500) : "";
        const err = typeof obj.stderr === "string" ? obj.stderr.trim().slice(0, 500) : "";
        const parts = [`Tests exited with code ${code}.`];
        if (out) parts.push("Stdout:\n" + out);
        if (err) parts.push("Stderr:\n" + err);
        return parts.join("\n");
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
// Dev feature request detection
// ---------------------------------------------------------------------------

/**
 * Heuristic: treat the message as a "dev feature" request when the user asks
 * to build, implement, or add something (e.g. "build auth for Eureka").
 * Used to trigger a two-phase reply: immediate "working on it" then a summary.
 */
export function isDevFeatureRequest(text: string): boolean {
    const t = text.trim().toLowerCase();
    const triggers = [
        "build ",
        "implement ",
        "add feature",
        "add a feature",
        "create a ",
        "create ",
        "add ",
        "implementing ",
        "building ",
    ];
    return triggers.some((phrase) => t.includes(phrase));
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

interface ProcessUserMessageResult {
    response: StandardResponse;
}

export interface ProcessUserMessageOptions {
    /** Use dev-agent prompt and more steps (plan, edit, test, summarize). */
    devMode?: boolean;
}

const SYSTEM_PROMPT_NORMAL =
    "You are a helpful coding assistant. First infer the user's intent from their message, then choose the single best tool (or answer directly) to fulfil that intent. " +
    "Use the available tools based on their descriptions; do not follow fixed recipes. " +
    "When the user names a repo (for example 'Eureka'), pass that name as repo_name to tools that accept it (such as get_uncommitted_changes, remove_line, remove_lines_matching, or delete_code) so they can resolve the path themselves. " +
    "When the user asks to delete a function, endpoint, or class (e.g. 'delete the GET /testing'), use delete_code with the search_term that identifies it (e.g. '/testing'). Do NOT use remove_line for this — delete_code removes the entire block. " +
    "Do not reply with only a list of repositories when the user asked for an action in a repo (such as checking uncommitted changes or editing code); instead, call the appropriate tool and report the result. " +
    "Always finish with a concise reply that explains what you did or found.";

const SYSTEM_PROMPT_DEV =
    "You are a dev agent. The user asked you to implement or build something (a feature, fix, or change). " +
    "You have these tools:\n" +
    "- read_file: Read a file's contents (with line numbers). Always read the target file first.\n" +
    "- insert_code: INSERT NEW CODE after a specific line number. Use this for adding new features, endpoints, functions, classes. Provide file_path, after_line (line number from read_file output), and new_code.\n" +
    "- edit_file: MODIFY EXISTING CODE via search/replace. Use this for changing or fixing existing code. Provide file_path, search_string, replace_string.\n" +
    "- delete_code: DELETE an entire function, class, or endpoint block. Provide file_path and search_term (e.g. '/testing' or 'def ping').\n" +
    "- search_local_codebase: Semantic code search to find relevant files.\n" +
    "- run_tests: Run tests after changes.\n" +
    "- list_git_repos, get_uncommitted_changes: Git helpers.\n" +
    "When the user names a repo (e.g. 'Eureka'), use repo_name in tools that accept it.\n\n" +
    "Critical rules:\n" +
    "1. ALWAYS read the target file first with read_file so you can see line numbers.\n" +
    "2. For ADDING new code (new endpoints, new functions, new features): use insert_code with after_line set to the line number after which to insert.\n" +
    "3. For MODIFYING existing code: use edit_file with search_string and replace_string.\n" +
    "4. Do NOT stop after read_file. You MUST follow through with insert_code or edit_file so the user gets an 'Approve & Apply' button.\n" +
    "5. Do NOT write unified diffs yourself. The daemon generates them.\n" +
    "Work step by step. End with a short summary of what you built or changed.";

/**
 * Primary orchestration entry point for a single user message.
 *
 * @param message - Platform-agnostic inbound message.
 * @param options - Optional: devMode for multi-step dev-agent flow.
 * @returns An object containing the final `StandardResponse` to send.
 */
export async function processUserMessage(
    message: StandardMessage,
    options?: ProcessUserMessageOptions,
): Promise<ProcessUserMessageResult> {
    const { senderId, text } = message;
    const devMode = options?.devMode === true;

    // -----------------------------------------------------------------------
    // 1. Load existing chat history (best-effort; degrades gracefully).
    // -----------------------------------------------------------------------
    const history: ModelMessage[] = await getChatHistory(senderId);

    // Current user turn as a ModelMessage (Vercel AI SDK core format).
    const userMessage: ModelMessage = {
        role: "user",
        content: text,
    };

    const systemContent = devMode ? SYSTEM_PROMPT_DEV : SYSTEM_PROMPT_NORMAL;
    const systemMessage: ModelMessage = {
        role: "system",
        content: systemContent,
    };

    const messages: ModelMessage[] = [systemMessage, ...history, userMessage];

    // -----------------------------------------------------------------------
    // 2. Call the LLM with tools enabled.
    // -----------------------------------------------------------------------
    let finalText = "I'm sorry, I wasn't able to generate a response.";
    let toolResponse: StandardResponse | null = null;
    const maxSteps = devMode ? 15 : 5;

    try {
        const result = await generateText({
            model,
            messages,
            tools: aiTools,
            maxSteps,
        } as Parameters<typeof generateText>[0]);

        // When the model returns empty text we still use tool output; only log a short summary.
        if (typeof result.text !== "string" || result.text.trim().length === 0) {
            const toolCount = Array.isArray(result.toolResults) ? result.toolResults.length : 0;
            const stepsCount = Array.isArray((result as { steps?: unknown[] }).steps)
                ? (result as { steps: unknown[] }).steps.length
                : 0;
            const firstToolName =
                Array.isArray(result.toolResults) && result.toolResults.length > 0
                    ? (result.toolResults[0] as { toolName?: string }).toolName ?? "?"
                    : null;
            console.info(
                `[orchestrator] generateText returned no text; toolResults=${toolCount}, steps=${stepsCount}, firstTool=${firstToolName}`,
            );
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

