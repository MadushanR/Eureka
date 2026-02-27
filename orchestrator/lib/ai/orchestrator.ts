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
import { getChatHistory, saveChatMessages, createJob, updateJob, clearActiveJob, type DevJob } from "@/lib/redis";
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

export type ProgressCallback = (update: string) => Promise<void>;

export interface ProcessUserMessageOptions {
    /** Use dev-agent prompt and more steps (plan, edit, test, summarize). */
    devMode?: boolean;
    /** Called after each phase/step so the adapter can push live updates. */
    onProgress?: ProgressCallback;
}

const SYSTEM_PROMPT_NORMAL =
    "You are a helpful coding assistant. First infer the user's intent from their message, then choose the single best tool (or answer directly) to fulfil that intent. " +
    "Use the available tools based on their descriptions; do not follow fixed recipes. " +
    "When the user names a repo (for example 'Eureka'), pass that name as repo_name to tools that accept it (such as get_uncommitted_changes, remove_line, remove_lines_matching, or delete_code) so they can resolve the path themselves. " +
    "When the user asks to delete a function, endpoint, or class (e.g. 'delete the GET /testing'), use delete_code with the search_term that identifies it (e.g. '/testing'). Do NOT use remove_line for this — delete_code removes the entire block. " +
    "Do not reply with only a list of repositories when the user asked for an action in a repo (such as checking uncommitted changes or editing code); instead, call the appropriate tool and report the result. " +
    "Always finish with a concise reply that explains what you did or found.";

const SYSTEM_PROMPT_DEV_PLAN =
    "You are a planning agent. Given the user's request, produce a short numbered plan of concrete steps. " +
    "Each step should name which tool to use and what arguments to pass.\n\n" +
    "Available tools:\n" +
    "- read_file(path, start_line?, end_line?): Read a file with line numbers.\n" +
    "- insert_code(file_path, after_line, new_code, repo_name?): Insert new code after a line number. Use for adding new features.\n" +
    "- edit_file(file_path, search_string, replace_string, repo_name?): Modify existing code via search/replace.\n" +
    "- delete_code(file_path, search_term, repo_name?): Delete an entire function/class/endpoint block.\n" +
    "- create_file(file_path, content, repo_name?): Create a single new file.\n" +
    "- batch_create_files(files[], repo_name?): Create multiple new files at once (for multi-file features).\n" +
    "- search_local_codebase(query): Semantic code search.\n" +
    "- run_tests(workspace_path, command_line): Run tests/builds (npm test, pytest, npm install, npm run build, etc.).\n" +
    "- list_git_repos, get_uncommitted_changes: Git helpers.\n\n" +
    "Rules:\n" +
    "- ALWAYS start by reading the target file(s) to see their current content and line numbers.\n" +
    "- For new code, use insert_code. For modifications, use edit_file. For deletions, use delete_code.\n" +
    "- For multi-file features, prefer batch_create_files to create all new files in one step.\n" +
    "- Do NOT write unified diffs. The daemon generates them.\n" +
    "- Keep the plan short (3-8 steps max).\n" +
    "- Output ONLY the numbered plan, no other text.";

const SYSTEM_PROMPT_DEV_EXECUTE =
    "You are a dev agent executing a plan. Follow the plan below step by step using the available tools.\n\n" +
    "Available tools:\n" +
    "- read_file: Read a file's contents (with line numbers). Use start_line/end_line for large files.\n" +
    "- insert_code: INSERT NEW CODE after a specific line number. Provide file_path, after_line, new_code.\n" +
    "- edit_file: MODIFY EXISTING CODE via search/replace. Provide file_path, search_string, replace_string.\n" +
    "- delete_code: DELETE an entire function/class/endpoint block. Provide file_path and search_term.\n" +
    "- create_file: CREATE A SINGLE NEW FILE. Provide file_path and content.\n" +
    "- batch_create_files: CREATE MULTIPLE NEW FILES at once. Provide an array of { file_path, content }.\n" +
    "- search_local_codebase: Semantic code search.\n" +
    "- run_tests: Run tests/build/install commands (npm test, pytest, npm install, npm run build, etc.).\n" +
    "- list_git_repos, get_uncommitted_changes: Git helpers.\n\n" +
    "When the user names a repo (e.g. 'Eureka'), use repo_name in tools that accept it.\n\n" +
    "Critical rules:\n" +
    "1. Execute each step of the plan in order.\n" +
    "2. Do NOT stop after read_file — always follow through with the action tool (insert_code, edit_file, delete_code, create_file, or batch_create_files).\n" +
    "3. Do NOT write unified diffs yourself. The daemon generates them.\n" +
    "4. If a tool fails, try to fix the issue (e.g. re-read the file, adjust arguments) and retry.\n" +
    "5. End with a short summary of what you did.";

const SYSTEM_PROMPT_DEV_EVALUATE =
    "You are a QA evaluator. The user asked for a specific change. The dev agent executed some steps. " +
    "Based on the tool results below, determine: did the agent fully complete the task?\n\n" +
    "Reply with EXACTLY one of:\n" +
    "- COMPLETE: <one sentence summary of what was done>\n" +
    "- INCOMPLETE: <what is still missing and what the agent should do next>\n" +
    "- FAILED: <what went wrong>";

// ---------------------------------------------------------------------------
// Helper: run one generateText call and extract results
// ---------------------------------------------------------------------------

interface GenerateResult {
    text: string;
    toolResponse: StandardResponse | null;
    allToolResults: Array<{ result?: unknown; output?: unknown; toolName?: string }>;
    hasApproveButton: boolean;
}

async function runGenerate(
    messages: ModelMessage[],
    maxSteps: number,
): Promise<GenerateResult> {
    const result = await generateText({
        model,
        messages,
        tools: aiTools,
        maxSteps,
    } as Parameters<typeof generateText>[0]);

    let text = "";
    if (typeof result.text === "string" && result.text.trim().length > 0) {
        text = result.text.trim();
    }

    const allToolResults: Array<{ result?: unknown; output?: unknown; toolName?: string }> = [];
    if (Array.isArray(result.toolResults)) {
        for (const tr of result.toolResults) {
            allToolResults.push(tr as { result?: unknown; output?: unknown; toolName?: string });
        }
    }

    let toolResponse: StandardResponse | null = null;
    let hasApproveButton = false;
    for (const toolResult of allToolResults) {
        const value = toolResult.result ?? toolResult.output;
        if (isStandardResponse(value)) {
            toolResponse = value;
            if (
                value.interactiveButtons?.some(
                    (b) => typeof b.action === "string" && b.action.startsWith("apply_patch:"),
                )
            ) {
                hasApproveButton = true;
            }
            break;
        }
    }

    if (!text && allToolResults.length > 0) {
        const last = allToolResults[allToolResults.length - 1];
        const value = last?.result ?? last?.output;
        if (value !== undefined && value !== null) {
            text = formatToolResultForUser(value, last.toolName);
        }
    }

    return { text, toolResponse, allToolResults, hasApproveButton };
}

// ---------------------------------------------------------------------------
// Normal (non-dev) orchestration
// ---------------------------------------------------------------------------

async function processNormal(
    senderId: string,
    text: string,
    history: ModelMessage[],
): Promise<ProcessUserMessageResult> {
    const userMessage: ModelMessage = { role: "user", content: text };
    const systemMessage: ModelMessage = { role: "system", content: SYSTEM_PROMPT_NORMAL };
    const messages: ModelMessage[] = [systemMessage, ...history, userMessage];

    let finalText = "I'm sorry, I wasn't able to generate a response.";
    let toolResponse: StandardResponse | null = null;

    try {
        const result = await runGenerate(messages, 5);
        if (result.text) finalText = result.text;
        if (result.toolResponse) toolResponse = result.toolResponse;
    } catch (error) {
        console.error(`[orchestrator] generateText failed for sender="${senderId}":`, error);
        finalText = "I ran into an error while talking to the language model. Please try again in a moment.";
    }

    const assistantMessage: ModelMessage = { role: "assistant", content: finalText };
    await saveChatMessages(senderId, [userMessage, assistantMessage]);

    return { response: toolResponse ?? { text: finalText } };
}

// ---------------------------------------------------------------------------
// Dev-mode agentic orchestration (plan → execute → evaluate → retry)
// ---------------------------------------------------------------------------

const MAX_DEV_ATTEMPTS = 3;

function summariseToolCalls(results: Array<{ result?: unknown; output?: unknown; toolName?: string }>): string {
    return results.map((tr) => {
        const name = tr.toolName ?? "unknown_tool";
        const value = tr.result ?? tr.output;
        const preview = typeof value === "object" && value !== null
            ? JSON.stringify(value).slice(0, 200)
            : String(value ?? "").slice(0, 200);
        return `${name}: ${preview}`;
    }).join("\n");
}

function extractToolNames(results: Array<{ toolName?: string }>): string[] {
    return results.map((r) => r.toolName ?? "unknown").filter((n) => n !== "unknown");
}

async function processDev(
    senderId: string,
    text: string,
    history: ModelMessage[],
    onProgress?: ProgressCallback,
): Promise<ProcessUserMessageResult> {
    const userMessage: ModelMessage = { role: "user", content: text };
    const job = await createJob(senderId, text);

    const progress = async (msg: string, jobUpdates?: Partial<DevJob>) => {
        if (jobUpdates) await updateJob(job.id, jobUpdates);
        if (onProgress) {
            try { await onProgress(msg); } catch (e) {
                console.error(`[orchestrator:dev] onProgress error:`, e);
            }
        }
    };

    // -----------------------------------------------------------------------
    // Phase 1: Plan
    // -----------------------------------------------------------------------
    console.info(`[orchestrator:dev] Phase 1 — Planning for sender="${senderId}"`);
    await progress("Planning your request...", { status: "planning", current_action: "Planning..." });

    let plan = "";
    let planSteps: string[] = [];
    try {
        const planMessages: ModelMessage[] = [
            { role: "system", content: SYSTEM_PROMPT_DEV_PLAN },
            ...history,
            userMessage,
        ];
        const planResult = await generateText({
            model,
            messages: planMessages,
        } as Parameters<typeof generateText>[0]);
        plan = (typeof planResult.text === "string" ? planResult.text.trim() : "") || "";
        planSteps = plan.split("\n").filter((l) => /^\d+[\.\)]/.test(l.trim()));
        console.info(`[orchestrator:dev] Plan (${planSteps.length} steps):\n${plan}`);
    } catch (error) {
        console.error(`[orchestrator:dev] Planning failed:`, error);
    }

    const totalSteps = planSteps.length || 1;
    await progress(
        `Plan ready (${totalSteps} step${totalSteps > 1 ? "s" : ""}):\n${planSteps.map((s) => s.trim()).join("\n") || "(executing directly)"}`,
        { plan: planSteps, total_steps: totalSteps, status: "executing", current_action: "Starting execution..." },
    );

    // -----------------------------------------------------------------------
    // Phase 2: Execute with retry loop
    // -----------------------------------------------------------------------
    let finalText = "I attempted to work on your request but wasn't able to complete it.";
    let toolResponse: StandardResponse | null = null;
    let attempt = 0;
    const filesCreated: string[] = [];

    const executeMessages: ModelMessage[] = [
        {
            role: "system",
            content: SYSTEM_PROMPT_DEV_EXECUTE + (plan ? `\n\nPLAN:\n${plan}` : ""),
        },
        ...history,
        userMessage,
    ];

    while (attempt < MAX_DEV_ATTEMPTS) {
        attempt++;
        console.info(`[orchestrator:dev] Phase 2 — Execute attempt ${attempt}/${MAX_DEV_ATTEMPTS} for sender="${senderId}"`);
        await progress(
            `Executing (attempt ${attempt}/${MAX_DEV_ATTEMPTS})...`,
            { current_step: attempt, current_action: `Executing (attempt ${attempt})` },
        );

        try {
            const result = await runGenerate(executeMessages, 15);

            if (result.text) finalText = result.text;
            if (result.toolResponse) toolResponse = result.toolResponse;

            const toolNames = extractToolNames(result.allToolResults);
            const actionTools = toolNames.filter((n) => ["insert_code", "edit_file", "delete_code", "create_file"].includes(n));
            if (actionTools.length > 0) {
                const stepsCompleted = actionTools.map((t) => `Used ${t}`);
                await progress(
                    `Executed: ${actionTools.join(", ")}`,
                    { steps_completed: stepsCompleted, current_action: `Completed: ${actionTools.join(", ")}` },
                );
            }

            for (const tr of result.allToolResults) {
                if (tr.toolName === "create_file") {
                    const val = tr.result ?? tr.output;
                    if (val && typeof val === "object" && "path" in (val as Record<string, unknown>)) {
                        filesCreated.push(String((val as Record<string, unknown>).path));
                    }
                }
            }

            if (result.hasApproveButton) {
                console.info(`[orchestrator:dev] Got Approve & Apply button on attempt ${attempt}.`);
                await progress(
                    "Changes ready for approval.",
                    { status: "complete", current_action: "Awaiting approval", files_created: filesCreated, finished_at: Date.now() },
                );
                break;
            }

            // ---------------------------------------------------------------
            // Phase 3: Evaluate — did the agent complete the task?
            // ---------------------------------------------------------------
            if (attempt < MAX_DEV_ATTEMPTS) {
                console.info(`[orchestrator:dev] Phase 3 — Evaluating attempt ${attempt}`);
                await progress("Evaluating progress...", { current_action: "Evaluating..." });

                const toolSummary = summariseToolCalls(result.allToolResults);

                try {
                    const evalMessages: ModelMessage[] = [
                        { role: "system", content: SYSTEM_PROMPT_DEV_EVALUATE },
                        {
                            role: "user",
                            content:
                                `User request: "${text}"\n\n` +
                                `Agent's text response: "${finalText}"\n\n` +
                                `Tool calls and results:\n${toolSummary || "(none)"}`,
                        },
                    ];
                    const evalResult = await generateText({
                        model,
                        messages: evalMessages,
                    } as Parameters<typeof generateText>[0]);
                    const verdict = (typeof evalResult.text === "string" ? evalResult.text.trim() : "") || "";
                    console.info(`[orchestrator:dev] Evaluation verdict: ${verdict}`);

                    if (verdict.startsWith("COMPLETE")) {
                        await progress(
                            "Task complete.",
                            { status: "complete", current_action: "Done", files_created: filesCreated, finished_at: Date.now() },
                        );
                        break;
                    }

                    const nudge = verdict.startsWith("INCOMPLETE")
                        ? verdict.replace(/^INCOMPLETE:\s*/, "")
                        : verdict.startsWith("FAILED")
                          ? `The previous attempt failed: ${verdict.replace(/^FAILED:\s*/, "")}. Try a different approach.`
                          : "The task is not complete. Continue executing the plan.";

                    await progress(`Retrying: ${nudge.slice(0, 120)}`, { errors: [nudge] });

                    executeMessages.push(
                        { role: "assistant", content: finalText || "(no text)" },
                        { role: "user", content: `[SYSTEM] ${nudge}` },
                    );
                } catch (evalError) {
                    console.error(`[orchestrator:dev] Evaluation failed:`, evalError);
                    await progress("Evaluation failed, retrying...");
                    executeMessages.push(
                        { role: "assistant", content: finalText || "(no text)" },
                        {
                            role: "user",
                            content: "[SYSTEM] The task is not complete. You stopped after reading or searching. You MUST call insert_code, edit_file, delete_code, or create_file to make the change. Continue.",
                        },
                    );
                }
            }
        } catch (error) {
            console.error(`[orchestrator:dev] Execute attempt ${attempt} failed:`, error);
            const errMsg = error instanceof Error ? error.message : String(error);
            if (attempt >= MAX_DEV_ATTEMPTS) {
                finalText = "I ran into an error while working on your request. Please try again.";
                await progress(
                    `Failed after ${attempt} attempts: ${errMsg.slice(0, 120)}`,
                    { status: "failed", errors: [errMsg], finished_at: Date.now() },
                );
            } else {
                await progress(`Attempt ${attempt} errored, retrying...`, { errors: [errMsg] });
                executeMessages.push(
                    { role: "assistant", content: "(error occurred)" },
                    { role: "user", content: "[SYSTEM] The previous attempt hit an error. Try again with a different approach." },
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Persist and return
    // -----------------------------------------------------------------------
    await clearActiveJob(senderId);
    const assistantMessage: ModelMessage = { role: "assistant", content: finalText };
    await saveChatMessages(senderId, [userMessage, assistantMessage]);

    return { response: toolResponse ?? { text: finalText } };
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

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
    const history: ModelMessage[] = await getChatHistory(senderId);

    if (options?.devMode) {
        return processDev(senderId, text, history, options.onProgress);
    }
    return processNormal(senderId, text, history);
}

