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
import { getChatHistory, saveChatMessages, createJob, getJob, updateJob, clearActiveJob, getPendingPatch, getPendingPush, type DevJob, type DevJobPhase } from "@/lib/redis";
import { getUserProfile, formatProfileForPrompt } from "./memory";
import { aiTools } from "./tools";

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

/**
 * LLM_MODEL_NAME     — used for normal chat / quick queries.
 * LLM_MODEL_NAME_DEV — used for the dev agent (plan, execute, evaluate).
 *                       Falls back to LLM_MODEL_NAME if not set.
 */
const MODEL_NAME = process.env.LLM_MODEL_NAME || "gpt-4.1-mini";
const MODEL_NAME_DEV = process.env.LLM_MODEL_NAME_DEV || MODEL_NAME;

const model = openai(MODEL_NAME);
const devModel = openai(MODEL_NAME_DEV);

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
export function isResearchRequest(text: string): boolean {
    const t = text.trim().toLowerCase();
    const triggers = [
        "do research on",
        "research on",
        "research about",
        "write a paper",
        "write a research",
        "write paper",
        "research paper",
        "investigate ",
    ];
    return triggers.some((phrase) => t.includes(phrase));
}

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
    "When the user asks to create a new GitHub repo, use github_create_repo. When they ask to clone a repo, use git_clone. " +
    "Do not reply with only a list of repositories when the user asked for an action in a repo (such as checking uncommitted changes or editing code); instead, call the appropriate tool and report the result. " +
    "Always finish with a concise reply that explains what you did or found.";

const SYSTEM_PROMPT_DEV_PLAN =
    "You are a planning agent. Given the user's request, produce a short numbered plan of concrete steps. " +
    "Each step should name which tool to use and what arguments to pass.\n\n" +
    "Available tools:\n" +
    "- github_create_repo(name, description?, private?): Create or get a GitHub repo. If it already exists, returns the existing repo's URL. Returns html_url and clone_url.\n" +
    "- git_clone(clone_url): Clone a repo locally. Only pass clone_url. If already cloned, returns existing local_path.\n" +
    "- batch_create_files(workspace_path, files[]): Create multiple new files at once.\n" +
    "- create_file(workspace_path, file_path, content): Create a single new file.\n" +
    "- read_file(path, start_line?, end_line?): Read a file with line numbers.\n" +
    "- insert_code(file_path, after_line, new_code): Insert new code after a line number.\n" +
    "- edit_file(file_path, search_string, replace_string): Modify existing code via search/replace.\n" +
    "- delete_code(file_path, search_term): Delete an entire function/class/endpoint block.\n" +
    "- run_tests(workspace_path, command_line): Run tests/builds (pytest, npm test, etc.).\n" +
    "- prepare_push_approval(workspace_path, commit_message): Commit+push for user approval.\n\n" +
    "CRITICAL — When the user says 'create a repo' or 'create a new project', the plan MUST follow this exact pattern:\n" +
    "  1. github_create_repo(name=...) — create the repo on GitHub\n" +
    "  2. git_clone(clone_url=<from step 1>) — clone it locally\n" +
    "  3. batch_create_files(workspace_path=<local_path from step 2>, files=[...]) — write ALL app code, test files, AND requirements.txt (include pytest)\n" +
    "  4. run_tests(workspace_path=<local_path from step 2>, command_line='pip install -r requirements.txt') — install dependencies\n" +
    "  5. run_tests(workspace_path=<local_path from step 2>, command_line='python -m pytest') — run tests\n" +
    "  6. prepare_push_approval(workspace_path=<local_path from step 2>, commit_message=...) — commit and push\n" +
    "Do NOT skip github_create_repo or git_clone. Do NOT try to read_file on a repo that doesn't exist yet.\n\n" +
    "For changes to EXISTING repos, start by reading the target file(s).\n\n" +
    "Rules:\n" +
    "- Do NOT write unified diffs. The daemon generates them.\n" +
    "- Keep the plan short (3-8 steps).\n" +
    "- Output ONLY the numbered plan, no other text.";

const SYSTEM_PROMPT_DEV_EXECUTE =
    "You are a dev agent. Execute ALL steps of the plan below using the available tools. " +
    "You MUST complete every step in a SINGLE pass — do NOT stop partway through.\n\n" +
    "Available tools:\n" +
    "- github_create_repo: Create a GitHub repo. Provide name. Returns html_url and clone_url. If repo already exists, it returns the existing repo's URL — that's fine, just continue.\n" +
    "- git_clone: Clone a repo. Provide ONLY clone_url. Do NOT pass parent_directory. Returns local_path. If already cloned, returns the existing local_path — that's fine, just continue.\n" +
    "- batch_create_files: Create multiple files. Provide workspace_path and files array [{file_path, content}].\n" +
    "- create_file: Create one file. Provide workspace_path, file_path, and content.\n" +
    "- read_file: Read a file with line numbers.\n" +
    "- insert_code: Insert code after a line number.\n" +
    "- edit_file: Modify code via search/replace.\n" +
    "- delete_code: Delete a function/class/endpoint block.\n" +
    "- run_tests: Run a command (pytest, npm test, etc.). Provide workspace_path and command_line.\n" +
    "- prepare_push_approval: Commit+push for approval. Provide workspace_path and commit_message.\n\n" +
    "CRITICAL RULES:\n" +
    "1. Execute ALL steps — do NOT stop after one or two tool calls.\n" +
    "2. For new projects: github_create_repo → git_clone → batch_create_files → run_tests → prepare_push_approval, ALL in sequence.\n" +
    "3. For batch_create_files: pass workspace_path = the local_path from git_clone. Write COMPLETE file contents — not stubs.\n" +
    "4. ALWAYS include a requirements.txt (with pytest for Python projects). Run 'pip install -r requirements.txt' BEFORE running tests.\n" +
    "5. Do NOT stop after github_create_repo or git_clone — you MUST continue to create files, install deps, test, and push.\n" +
    "6. Do NOT write unified diffs. The daemon handles that.\n" +
    "7. If a tool returns success=true (even with 'already exists'), treat it as DONE and move to the NEXT step. Never retry a succeeded step.\n" +
    "8. ALWAYS include the GitHub repo URL (html_url) in your final summary.\n" +
    "9. End with a concise summary of everything you did.\n" +
    "10. NEVER stop and ask for user input. Always proceed autonomously through every step.";

const SYSTEM_PROMPT_DEV_EVALUATE =
    "You are a QA evaluator. The user asked for a specific change. The dev agent executed some steps. " +
    "Based on the tool results below, determine: did the agent fully complete the task?\n\n" +
    "IMPORTANT: If a tool returned success=true with a message like 'already exists' or 'using existing', " +
    "that step IS complete — do NOT mark it as incomplete. Focus only on what has NOT been done yet.\n\n" +
    "Reply with EXACTLY one of:\n" +
    "- COMPLETE: <one sentence summary of what was done>\n" +
    "- INCOMPLETE: <what is still missing and what the agent should do next — be specific about the NEXT action>\n" +
    "- FAILED: <what went wrong>";

// ---------------------------------------------------------------------------
// Multi-phase prompts
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT_DEV_DECOMPOSE =
    "You are a project architect. Break the user's request into sequential build phases.\n\n" +
    "Rules:\n" +
    "- Phase 0 is ALWAYS 'setup': create GitHub repo, clone, scaffold (if applicable), install deps.\n" +
    "- Subsequent phases are feature modules, each self-contained (3-8 files max per phase).\n" +
    "- Order phases so later ones can depend on earlier ones (e.g. models before API, API before frontend).\n" +
    "- Each phase must have: name (short), description (what to build), files (list of relative file paths to create/edit).\n" +
    "- For small projects (<=8 files total), use just 2 phases: setup + implementation.\n" +
    "- For medium projects (9-20 files), use 3-5 phases.\n" +
    "- For large projects (20+ files), use 5-10 phases.\n\n" +
    "Available scaffold commands (for phase 0):\n" +
    "- npx create-next-app@latest <name> --typescript --tailwind --eslint --app --use-npm\n" +
    "- npx create-vite <name> --template react-ts\n" +
    "- django-admin startproject <name>\n" +
    "- python -m venv venv\n" +
    "- npm init -y / yarn init -y / pnpm init\n\n" +
    "Output ONLY valid JSON — no markdown fences, no explanation. Format:\n" +
    '{\n' +
    '  "project_name": "my-project",\n' +
    '  "scaffold_command": "npx create-next-app@latest my-project --typescript --tailwind --eslint --app --use-npm",\n' +
    '  "phases": [\n' +
    '    { "name": "Setup", "description": "Create repo, clone, scaffold, install deps", "files": ["package.json", "tsconfig.json"] },\n' +
    '    { "name": "Data Models", "description": "Create database models and types", "files": ["src/types/product.ts", "src/lib/db.ts"] }\n' +
    '  ]\n' +
    '}';

const SYSTEM_PROMPT_PHASE_EXECUTE =
    "You are a dev agent building ONE module of a larger project. " +
    "The repo already exists locally. Build ONLY the files for this phase.\n\n" +
    "Available tools:\n" +
    "- run_scaffold: Run scaffold/init commands (npx create-next-app, django-admin, npm init, pip install, etc.).\n" +
    "- batch_create_files: Create multiple files. Provide workspace_path and files array [{file_path, content}].\n" +
    "- create_file: Create one file. Provide workspace_path, file_path, and content.\n" +
    "- read_file: Read a file with line numbers.\n" +
    "- insert_code: Insert code after a line number.\n" +
    "- edit_file: Modify code via search/replace.\n" +
    "- delete_code: Delete a function/class/endpoint block.\n" +
    "- run_tests: Run a command (pytest, npm test, npm run build, etc.).\n\n" +
    "CRITICAL RULES:\n" +
    "1. Write COMPLETE file contents — never stubs or placeholders.\n" +
    "2. If a file already exists and you need to modify it, use edit_file or insert_code — not create_file.\n" +
    "3. Make files work with the existing codebase from previous phases.\n" +
    "4. If the phase includes tests, run them after creating the files.\n" +
    "5. Do NOT write unified diffs. The daemon handles that.\n" +
    "6. Do NOT call github_create_repo, git_clone, or prepare_push_approval — those are handled externally.\n" +
    "7. End with a concise summary of what you built in this phase.";

const SYSTEM_PROMPT_PHASE_EVALUATE =
    "You are a QA evaluator for a single build phase of a larger project. " +
    "The phase had a specific goal and list of files to create. " +
    "Based on the tool results below, determine: did the agent complete THIS phase?\n\n" +
    "IMPORTANT: Only evaluate whether THIS phase's goals were met — not the overall project.\n" +
    "If files were created successfully and any tests passed (or no tests were required), mark COMPLETE.\n\n" +
    "Reply with EXACTLY one of:\n" +
    "- COMPLETE: <one sentence summary>\n" +
    "- INCOMPLETE: <what is still missing for THIS phase>\n" +
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
    overrideModel?: ReturnType<typeof openai>,
): Promise<GenerateResult> {
    const result = await generateText({
        model: overrideModel ?? model,
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

    const profile = await getUserProfile(senderId);
    const profileSnippet = formatProfileForPrompt(profile);
    const systemMessage: ModelMessage = { role: "system", content: SYSTEM_PROMPT_NORMAL + profileSnippet };
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

/** Base number of execute attempts for simple plans. Complex plans get more (see maxDevAttempts). */
const MAX_DEV_ATTEMPTS_BASE = 5;
const MAX_DEV_ATTEMPTS_CAP = 10;

/** Dynamic max attempts: longer plans get more retries (e.g. create repo + clone + build + test + push). */
function maxDevAttempts(planStepCount: number): number {
    const steps = planStepCount || 1;
    return Math.min(MAX_DEV_ATTEMPTS_CAP, Math.max(MAX_DEV_ATTEMPTS_BASE, steps + 2));
}

/**
 * Extract all button actions from tool results.
 */
function extractButtonActions(
    toolResults: Array<{ result?: unknown; output?: unknown; toolName?: string }>,
): string[] {
    const actions: string[] = [];
    for (const tr of toolResults) {
        const value = tr.result ?? tr.output;
        if (value && typeof value === "object" && "interactiveButtons" in (value as Record<string, unknown>)) {
            const buttons = (value as Record<string, unknown>).interactiveButtons;
            if (Array.isArray(buttons)) {
                for (const b of buttons) {
                    const action = (b as Record<string, unknown>).action;
                    if (typeof action === "string") actions.push(action);
                }
            }
        }
    }
    return actions;
}

/**
 * Auto-apply all staged patches and auto-push found in tool results.
 * Used by the dev agent so it doesn't stop for user approval on every edit.
 */
async function autoApplyAll(
    toolResults: Array<{ result?: unknown; output?: unknown; toolName?: string }>,
): Promise<{ applied: number; pushed: boolean; errors: string[] }> {
    const daemonUrl = process.env.LOCAL_DAEMON_URL?.replace(/\/+$/, "");
    if (!daemonUrl) return { applied: 0, pushed: false, errors: ["LOCAL_DAEMON_URL not configured"] };

    const actions = extractButtonActions(toolResults);
    let applied = 0;
    let pushed = false;
    const errors: string[] = [];

    // Auto-apply patches
    for (const action of actions) {
        if (action.startsWith("apply_patch:")) {
            const patchId = action.slice("apply_patch:".length);
            try {
                const staged = await getPendingPatch(patchId);
                if (!staged) { errors.push(`Patch ${patchId} expired`); continue; }
                const workspace = staged.workspace_path?.trim() || process.env.LOCAL_DAEMON_WORKSPACE_PATH?.trim();
                if (!workspace) { errors.push(`No workspace for patch ${patchId}`); continue; }
                const res = await fetch(`${daemonUrl}/apply-patch`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ patch_string: staged.patch_string, workspace_path: workspace }),
                });
                const data = (await res.json().catch(() => ({}))) as { success?: boolean; message?: string };
                if (data.success) {
                    applied++;
                    console.info(`[orchestrator:dev] Auto-applied patch ${patchId}`);
                } else {
                    errors.push(`Patch ${patchId}: ${data.message ?? "failed"}`);
                    console.warn(`[orchestrator:dev] Auto-apply failed for ${patchId}: ${data.message}`);
                }
            } catch (e) {
                errors.push(`Patch ${patchId}: ${e instanceof Error ? e.message : String(e)}`);
            }
        }
    }

    // Auto-push (commit + push)
    for (const action of actions) {
        if (action.startsWith("push:")) {
            const pushId = action.slice("push:".length);
            try {
                const staged = await getPendingPush(pushId);
                if (!staged) { errors.push(`Push ${pushId} expired`); continue; }
                const res = await fetch(`${daemonUrl}/git/commit-and-push`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ workspace_path: staged.workspace_path, commit_message: staged.commit_message }),
                });
                const data = (await res.json().catch(() => ({}))) as { success?: boolean; message?: string };
                if (data.success) {
                    pushed = true;
                    console.info(`[orchestrator:dev] Auto-pushed ${pushId}`);
                } else {
                    errors.push(`Push: ${data.message ?? "failed"}`);
                    console.warn(`[orchestrator:dev] Auto-push failed: ${data.message}`);
                }
            } catch (e) {
                errors.push(`Push: ${e instanceof Error ? e.message : String(e)}`);
            }
        }
    }

    return { applied, pushed, errors };
}

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

    const profile = await getUserProfile(senderId);
    const profileSnippet = formatProfileForPrompt(profile);

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
            { role: "system", content: SYSTEM_PROMPT_DEV_PLAN + profileSnippet },
            ...history,
            userMessage,
        ];
        const planResult = await generateText({
            model: devModel,
            messages: planMessages,
        } as Parameters<typeof generateText>[0]);
        plan = (typeof planResult.text === "string" ? planResult.text.trim() : "") || "";
        planSteps = plan.split("\n").filter((l) => /^\d+[\.\)]/.test(l.trim()));
        console.info(`[orchestrator:dev] Plan (${planSteps.length} steps) [model=${MODEL_NAME_DEV}]:\n${plan}`);
    } catch (error) {
        console.error(`[orchestrator:dev] Planning failed:`, error);
    }

    const totalSteps = planSteps.length || 1;
    const maxAttempts = maxDevAttempts(totalSteps);
    await progress(
        `Plan ready (${totalSteps} step${totalSteps > 1 ? "s" : ""}, up to ${maxAttempts} attempts):\n${planSteps.map((s) => s.trim()).join("\n") || "(executing directly)"}`,
        { plan: planSteps, total_steps: totalSteps, status: "executing", current_action: "Starting execution..." },
    );

    // -----------------------------------------------------------------------
    // Phase 2: Execute with retry loop
    // -----------------------------------------------------------------------
    let finalText = "I attempted to work on your request but wasn't able to complete it.";
    let toolResponse: StandardResponse | null = null;
    let attempt = 0;
    const filesCreated: string[] = [];
    const allSuccessfulTools: string[] = [];

    const executeMessages: ModelMessage[] = [
        {
            role: "system",
            content: SYSTEM_PROMPT_DEV_EXECUTE + profileSnippet + (plan ? `\n\nPLAN:\n${plan}` : ""),
        },
        ...history,
        userMessage,
    ];

    while (attempt < maxAttempts) {
        // Check for cancellation before each attempt
        const currentJob = await getJob(job.id);
        if (currentJob?.status === "cancelled") {
            console.info(`[orchestrator:dev] Job ${job.id} was cancelled by user.`);
            finalText = "Job cancelled.";
            break;
        }

        attempt++;
        console.info(`[orchestrator:dev] Phase 2 — Execute attempt ${attempt}/${maxAttempts} for sender="${senderId}"`);
        await progress(
            `Executing (attempt ${attempt}/${maxAttempts})...`,
            { current_step: attempt, current_action: `Executing (attempt ${attempt})` },
        );

        try {
            const result = await runGenerate(executeMessages, 25, devModel);

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

            // Auto-apply patches and auto-push during dev mode
            const buttonActions = extractButtonActions(result.allToolResults);
            if (buttonActions.length > 0) {
                console.info(`[orchestrator:dev] Auto-applying ${buttonActions.length} action(s) from attempt ${attempt}...`);
                const autoResult = await autoApplyAll(result.allToolResults);
                if (autoResult.applied > 0) {
                    await progress(`Auto-applied ${autoResult.applied} patch(es).`);
                }
                if (autoResult.pushed) {
                    await progress("Auto-pushed to GitHub.", { status: "complete", current_action: "Pushed", finished_at: Date.now() });
                    // Push is the final step — task is done
                    break;
                }
                if (autoResult.errors.length > 0) {
                    console.warn(`[orchestrator:dev] Auto-apply errors: ${autoResult.errors.join("; ")}`);
                }
                // Continue to evaluate — more steps may remain
            }

            // ---------------------------------------------------------------
            // Phase 3: Evaluate — runs on EVERY attempt (including last)
            // ---------------------------------------------------------------
            console.info(`[orchestrator:dev] Phase 3 — Evaluating attempt ${attempt}`);
            await progress("Evaluating progress...", { current_action: "Evaluating..." });

            const toolSummary = summariseToolCalls(result.allToolResults);

            // Accumulate successful tool calls across ALL attempts
            for (const tr of result.allToolResults) {
                const val = tr.result ?? tr.output;
                const isSuccess = val && typeof val === "object"
                    && "success" in (val as Record<string, unknown>)
                    && (val as Record<string, unknown>).success === true;
                if (isSuccess || (val != null && tr.toolName)) {
                    const name = tr.toolName ?? "unknown";
                    const preview = typeof val === "object" && val !== null
                        ? JSON.stringify(val).slice(0, 300)
                        : String(val ?? "").slice(0, 300);
                    allSuccessfulTools.push(`${name}: ${preview}`);
                }
            }

            try {
                const evalMessages: ModelMessage[] = [
                    { role: "system", content: SYSTEM_PROMPT_DEV_EVALUATE },
                    {
                        role: "user",
                        content:
                            `User request: "${text}"\n\n` +
                            `Agent's text response: "${finalText}"\n\n` +
                            `Tool calls and results (attempt ${attempt}):\n${toolSummary || "(none)"}` +
                            (allSuccessfulTools.length > 0
                                ? `\n\nAll successful operations across all attempts:\n${allSuccessfulTools.join("\n")}`
                                : ""),
                    },
                ];
                const evalResult = await generateText({
                    model: devModel,
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

                if (attempt >= maxAttempts) {
                    // Last attempt — no more retries
                    await progress(
                        `Could not complete after ${attempt} attempts.`,
                        { status: "failed", finished_at: Date.now() },
                    );
                    break;
                }

                const missing = verdict.startsWith("INCOMPLETE")
                    ? verdict.replace(/^INCOMPLETE:\s*/, "")
                    : verdict.startsWith("FAILED")
                      ? verdict.replace(/^FAILED:\s*/, "")
                      : "The task is not complete.";

                const alreadyDone = allSuccessfulTools.length > 0
                    ? `\n\nALREADY COMPLETED (do NOT repeat these — they are DONE):\n${allSuccessfulTools.join("\n")}`
                    : "";

                const nudge =
                    `Continue from where you left off. Do NOT repeat steps that already succeeded.` +
                    ` If github_create_repo or git_clone already returned success, skip them entirely and move on.${alreadyDone}\n\n` +
                    `WHAT STILL NEEDS TO BE DONE: ${missing}`;

                await progress(`Retrying: ${missing.slice(0, 120)}`, { errors: [missing] });

                executeMessages.push(
                    { role: "assistant", content: finalText || "(no text)" },
                    { role: "user", content: `[SYSTEM] ${nudge}` },
                );
            } catch (evalError) {
                console.error(`[orchestrator:dev] Evaluation failed:`, evalError);

                if (attempt >= maxAttempts) {
                    await progress(
                        `Failed after ${attempt} attempts (evaluation error).`,
                        { status: "failed", finished_at: Date.now() },
                    );
                    break;
                }

                await progress("Evaluation failed, retrying...");

                const alreadyDone = allSuccessfulTools.length > 0
                    ? `\nALREADY COMPLETED (do NOT repeat):\n${allSuccessfulTools.join("\n")}\n\n`
                    : "";

                executeMessages.push(
                    { role: "assistant", content: finalText || "(no text)" },
                    {
                        role: "user",
                        content: `[SYSTEM] The task is not complete. ${alreadyDone}Continue from where you left off. Do NOT re-create repos or re-clone. Call the next action tool (batch_create_files, run_tests, prepare_push_approval).`,
                    },
                );
            }
        } catch (error) {
            console.error(`[orchestrator:dev] Execute attempt ${attempt} failed:`, error);
            const errMsg = error instanceof Error ? error.message : String(error);
            if (attempt >= maxAttempts) {
                finalText = "I ran into an error while working on your request. Please try again.";
                await progress(
                    `Failed after ${attempt} attempts: ${errMsg.slice(0, 120)}`,
                    { status: "failed", errors: [errMsg], finished_at: Date.now() },
                );
            } else {
                const alreadyDone = allSuccessfulTools.length > 0
                    ? `\nALREADY COMPLETED (do NOT repeat):\n${allSuccessfulTools.join("\n")}\n\n`
                    : "";
                await progress(`Attempt ${attempt} errored, retrying...`, { errors: [errMsg] });
                executeMessages.push(
                    { role: "assistant", content: "(error occurred)" },
                    { role: "user", content: `[SYSTEM] The previous attempt hit an error: ${errMsg.slice(0, 200)}. ${alreadyDone}Skip already-completed steps and continue with the NEXT step.` },
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
// Multi-phase dev orchestration
// ---------------------------------------------------------------------------

const MAX_PHASE_ATTEMPTS = 3;
const daemonUrl = () => process.env.LOCAL_DAEMON_URL?.replace(/\/+$/, "") ?? "";

interface DecomposedProject {
    project_name: string;
    scaffold_command?: string;
    phases: Array<{ name: string; description: string; files: string[] }>;
}

async function commitPhase(workspacePath: string, message: string): Promise<{ success: boolean; error?: string }> {
    try {
        const res = await fetch(`${daemonUrl()}/git/commit-all`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ workspace_path: workspacePath, commit_message: message }),
        });
        const data = (await res.json().catch(() => ({}))) as { success?: boolean; message?: string };
        return { success: data.success === true, error: data.message };
    } catch (e) {
        return { success: false, error: e instanceof Error ? e.message : String(e) };
    }
}

async function pushRepo(workspacePath: string, commitMessage: string): Promise<{ success: boolean; error?: string }> {
    try {
        const res = await fetch(`${daemonUrl()}/git/commit-and-push`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ workspace_path: workspacePath, commit_message: commitMessage }),
        });
        const data = (await res.json().catch(() => ({}))) as { success?: boolean; message?: string };
        return { success: data.success === true, error: data.message };
    } catch (e) {
        return { success: false, error: e instanceof Error ? e.message : String(e) };
    }
}

async function executePhase(
    phase: DecomposedProject["phases"][number],
    phaseIndex: number,
    totalPhases: number,
    workspacePath: string,
    existingFiles: string[],
    history: ModelMessage[],
    onProgress?: ProgressCallback,
): Promise<{ success: boolean; text: string; filesCreated: string[] }> {
    const phaseLabel = `[Phase ${phaseIndex + 1}/${totalPhases}: ${phase.name}]`;
    const filesCreated: string[] = [];

    const phaseContext =
        `${phaseLabel}\n` +
        `Workspace: ${workspacePath}\n` +
        `Phase goal: ${phase.description}\n` +
        `Files to create/modify: ${phase.files.join(", ")}\n` +
        (existingFiles.length > 0
            ? `Files already in the project from previous phases:\n${existingFiles.join("\n")}\n`
            : "This is the first content phase — no existing files yet.\n");

    const executeMessages: ModelMessage[] = [
        { role: "system", content: SYSTEM_PROMPT_PHASE_EXECUTE + "\n\n" + phaseContext },
        ...history.slice(-4),
        { role: "user", content: `Build this phase: ${phase.description}\n\nFiles: ${phase.files.join(", ")}` },
    ];

    let finalText = "";
    let attempt = 0;
    const allSuccessfulTools: string[] = [];

    while (attempt < MAX_PHASE_ATTEMPTS) {
        attempt++;
        if (onProgress) {
            try { await onProgress(`${phaseLabel} Executing (attempt ${attempt}/${MAX_PHASE_ATTEMPTS})...`); } catch {}
        }

        try {
            const result = await runGenerate(executeMessages, 25, devModel);
            if (result.text) finalText = result.text;

            // Track created files
            for (const tr of result.allToolResults) {
                if (tr.toolName === "create_file" || tr.toolName === "batch_create_files") {
                    const val = tr.result ?? tr.output;
                    if (val && typeof val === "object") {
                        if ("path" in (val as Record<string, unknown>)) {
                            filesCreated.push(String((val as Record<string, unknown>).path));
                        }
                        if ("results" in (val as Record<string, unknown>)) {
                            const results = (val as Record<string, unknown>).results as Array<{ file_path?: string; success?: boolean }> | undefined;
                            if (results) {
                                for (const r of results) {
                                    if (r.success && r.file_path) filesCreated.push(r.file_path);
                                }
                            }
                        }
                    }
                }
            }

            // Auto-apply any patches generated during this phase
            const buttonActions = extractButtonActions(result.allToolResults);
            if (buttonActions.length > 0) {
                const autoResult = await autoApplyAll(result.allToolResults);
                if (autoResult.applied > 0 && onProgress) {
                    try { await onProgress(`${phaseLabel} Auto-applied ${autoResult.applied} patch(es).`); } catch {}
                }
            }

            // Accumulate successful tools
            for (const tr of result.allToolResults) {
                const val = tr.result ?? tr.output;
                const isSuccess = val && typeof val === "object"
                    && "success" in (val as Record<string, unknown>)
                    && (val as Record<string, unknown>).success === true;
                if (isSuccess || (val != null && tr.toolName)) {
                    const name = tr.toolName ?? "unknown";
                    const preview = typeof val === "object" && val !== null
                        ? JSON.stringify(val).slice(0, 200)
                        : String(val ?? "").slice(0, 200);
                    allSuccessfulTools.push(`${name}: ${preview}`);
                }
            }

            // Evaluate this phase
            const toolSummary = summariseToolCalls(result.allToolResults);
            try {
                const evalMessages: ModelMessage[] = [
                    { role: "system", content: SYSTEM_PROMPT_PHASE_EVALUATE },
                    {
                        role: "user",
                        content:
                            `Phase: "${phase.name}"\nGoal: ${phase.description}\nExpected files: ${phase.files.join(", ")}\n\n` +
                            `Agent response: "${finalText}"\n\nTool results:\n${toolSummary || "(none)"}`,
                    },
                ];
                const evalResult = await generateText({
                    model: devModel,
                    messages: evalMessages,
                } as Parameters<typeof generateText>[0]);
                const verdict = (typeof evalResult.text === "string" ? evalResult.text.trim() : "") || "";
                console.info(`[orchestrator:phase] ${phaseLabel} Eval: ${verdict}`);

                if (verdict.startsWith("COMPLETE")) {
                    return { success: true, text: finalText, filesCreated };
                }

                if (attempt >= MAX_PHASE_ATTEMPTS) {
                    return { success: false, text: `Phase "${phase.name}" incomplete after ${attempt} attempts: ${verdict}`, filesCreated };
                }

                const missing = verdict.replace(/^(INCOMPLETE|FAILED):\s*/, "");
                const alreadyDone = allSuccessfulTools.length > 0
                    ? `\n\nALREADY DONE (do NOT repeat):\n${allSuccessfulTools.join("\n")}`
                    : "";

                executeMessages.push(
                    { role: "assistant", content: finalText || "(no text)" },
                    { role: "user", content: `[SYSTEM] Phase not complete. Continue.${alreadyDone}\n\nSTILL NEEDED: ${missing}` },
                );
            } catch {
                if (attempt >= MAX_PHASE_ATTEMPTS) {
                    return { success: true, text: finalText, filesCreated };
                }
            }
        } catch (error) {
            console.error(`[orchestrator:phase] ${phaseLabel} attempt ${attempt} error:`, error);
            if (attempt >= MAX_PHASE_ATTEMPTS) {
                return { success: false, text: `Phase "${phase.name}" failed: ${error instanceof Error ? error.message : String(error)}`, filesCreated };
            }
            executeMessages.push(
                { role: "assistant", content: "(error)" },
                { role: "user", content: "[SYSTEM] Error occurred. Try again." },
            );
        }
    }

    return { success: false, text: finalText || `Phase "${phase.name}" did not complete.`, filesCreated };
}

async function processDevMultiPhase(
    senderId: string,
    text: string,
    history: ModelMessage[],
    onProgress?: ProgressCallback,
): Promise<ProcessUserMessageResult> {
    const userMessage: ModelMessage = { role: "user", content: text };
    const job = await createJob(senderId, text);

    const profile = await getUserProfile(senderId);
    const profileSnippet = formatProfileForPrompt(profile);

    const progress = async (msg: string, jobUpdates?: Partial<DevJob>) => {
        if (jobUpdates) await updateJob(job.id, jobUpdates);
        if (onProgress) {
            try { await onProgress(msg); } catch (e) {
                console.error(`[orchestrator:multi] onProgress error:`, e);
            }
        }
    };

    // -----------------------------------------------------------------------
    // Phase 0: Decompose the project into phases
    // -----------------------------------------------------------------------
    console.info(`[orchestrator:multi] Decomposing project for sender="${senderId}"`);
    await progress("Analyzing project structure...", { status: "planning", current_action: "Decomposing project..." });

    let decomposed: DecomposedProject;
    try {
        const decomposeMessages: ModelMessage[] = [
            { role: "system", content: SYSTEM_PROMPT_DEV_DECOMPOSE + profileSnippet },
            ...history,
            userMessage,
        ];
        const decomposeResult = await generateText({
            model: devModel,
            messages: decomposeMessages,
        } as Parameters<typeof generateText>[0]);
        const rawText = (typeof decomposeResult.text === "string" ? decomposeResult.text.trim() : "") || "{}";

        // Strip markdown fences if present
        const jsonText = rawText.replace(/^```(?:json)?\s*\n?/i, "").replace(/\n?```\s*$/i, "").trim();
        decomposed = JSON.parse(jsonText) as DecomposedProject;

        if (!decomposed.phases || decomposed.phases.length === 0) {
            throw new Error("Decomposer returned no phases");
        }
    } catch (error) {
        console.error(`[orchestrator:multi] Decompose failed, falling back to single-phase:`, error);
        await clearActiveJob(senderId);
        return processDev(senderId, text, history, onProgress);
    }

    const phases = decomposed.phases;
    const totalPhases = phases.length;
    const projectName = decomposed.project_name || "project";

    const jobPhases: DevJobPhase[] = phases.map((p) => ({
        name: p.name,
        description: p.description,
        files: p.files,
        status: "pending" as const,
    }));

    console.info(`[orchestrator:multi] Decomposed into ${totalPhases} phases [model=${MODEL_NAME_DEV}]`);
    await progress(
        `Project decomposed into ${totalPhases} phases:\n${phases.map((p, i) => `${i + 1}. ${p.name}: ${p.description}`).join("\n")}`,
        { status: "executing", phases: jobPhases, total_phases: totalPhases, current_phase: 0 },
    );

    // -----------------------------------------------------------------------
    // Setup: Create repo + clone
    // -----------------------------------------------------------------------
    console.info(`[orchestrator:multi] Running setup: create repo + clone`);
    await progress("Setting up repository...", { current_action: "Creating repo...", current_phase: 0 });

    let workspacePath = "";
    let repoUrl = "";

    // Create the GitHub repo
    try {
        const setupMessages: ModelMessage[] = [
            {
                role: "system",
                content: SYSTEM_PROMPT_DEV_EXECUTE + `\n\nPLAN:\n1. github_create_repo(name="${projectName}")\n2. git_clone(clone_url=<from step 1>)`,
            },
            userMessage,
        ];
        const setupResult = await runGenerate(setupMessages, 10, devModel);

        // Extract workspace_path and repo URL from tool results
        for (const tr of setupResult.allToolResults) {
            const val = (tr.result ?? tr.output) as Record<string, unknown> | undefined;
            if (!val || typeof val !== "object") continue;
            if (val.local_path && typeof val.local_path === "string") {
                workspacePath = val.local_path;
            }
            if (val.html_url && typeof val.html_url === "string") {
                repoUrl = val.html_url;
            }
            if (val.clone_url && typeof val.clone_url === "string" && !workspacePath) {
                // If we got clone_url but not local_path yet, the clone step may come next
            }
        }

        if (!workspacePath) {
            // Fallback: construct from default workspace + project name
            const defaultWs = process.env.LOCAL_DAEMON_WORKSPACE_PATH ?? "C:/Users/madus/Desktop";
            workspacePath = `${defaultWs}/${projectName}`;
            console.warn(`[orchestrator:multi] Could not extract workspace_path from setup, using fallback: ${workspacePath}`);
        }

        console.info(`[orchestrator:multi] Repo ready at: ${workspacePath} (URL: ${repoUrl})`);
        await progress(`Repository ready: ${repoUrl || projectName}`, { current_action: "Repo created" });
    } catch (error) {
        console.error(`[orchestrator:multi] Setup failed:`, error);
        const errMsg = error instanceof Error ? error.message : String(error);
        await progress(`Setup failed: ${errMsg}`, { status: "failed", errors: [errMsg], finished_at: Date.now() });
        await clearActiveJob(senderId);
        return { response: { text: `Failed to set up repository: ${errMsg}` } };
    }

    // Run scaffold command if specified
    if (decomposed.scaffold_command) {
        await progress(`Running scaffold: ${decomposed.scaffold_command}...`, { current_action: "Scaffolding..." });
        try {
            const scaffoldMessages: ModelMessage[] = [
                {
                    role: "system",
                    content: `You are a dev agent. Run this scaffold command and nothing else.\n\nAvailable tools:\n- run_scaffold: Run a scaffold command.\n- run_tests: Run install commands.\n\nRun the command, then stop.`,
                },
                { role: "user", content: `Run this scaffold command in workspace "${workspacePath}": ${decomposed.scaffold_command}` },
            ];
            const scaffoldResult = await runGenerate(scaffoldMessages, 5, devModel);
            console.info(`[orchestrator:multi] Scaffold complete: ${scaffoldResult.text?.slice(0, 100)}`);
            await progress("Scaffold complete.");

            // Commit scaffold
            const commitResult = await commitPhase(workspacePath, `chore: scaffold ${projectName}`);
            if (commitResult.success) {
                console.info(`[orchestrator:multi] Scaffold committed.`);
            }
        } catch (error) {
            console.warn(`[orchestrator:multi] Scaffold command failed (continuing):`, error);
        }
    }

    // -----------------------------------------------------------------------
    // Execute each phase
    // -----------------------------------------------------------------------
    const allFilesCreated: string[] = [];
    const phaseSummaries: string[] = [];
    let lastPhaseSucceeded = true;

    for (let i = 0; i < phases.length; i++) {
        // Check cancellation
        const currentJob = await getJob(job.id);
        if (currentJob?.status === "cancelled") {
            console.info(`[orchestrator:multi] Job cancelled by user at phase ${i + 1}.`);
            break;
        }

        const phase = phases[i];
        const phaseNum = i + 1;

        // Update job state
        jobPhases[i].status = "executing";
        await progress(
            `Building phase ${phaseNum}/${totalPhases}: ${phase.name}...`,
            { current_phase: phaseNum, current_action: `Phase ${phaseNum}: ${phase.name}`, phases: jobPhases },
        );

        console.info(`[orchestrator:multi] Starting phase ${phaseNum}/${totalPhases}: ${phase.name}`);

        const phaseResult = await executePhase(
            phase,
            i,
            totalPhases,
            workspacePath,
            allFilesCreated,
            history,
            onProgress,
        );

        if (phaseResult.success) {
            jobPhases[i].status = "complete";
            allFilesCreated.push(...phaseResult.filesCreated);
            phaseSummaries.push(`Phase ${phaseNum} (${phase.name}): Done`);

            // Commit after each successful phase
            const commitResult = await commitPhase(workspacePath, `feat: ${phase.name.toLowerCase()}`);
            if (commitResult.success) {
                console.info(`[orchestrator:multi] Phase ${phaseNum} committed.`);
            } else {
                console.warn(`[orchestrator:multi] Phase ${phaseNum} commit failed: ${commitResult.error}`);
            }

            await progress(`Phase ${phaseNum}/${totalPhases} complete: ${phase.name}`);
        } else {
            jobPhases[i].status = "failed";
            phaseSummaries.push(`Phase ${phaseNum} (${phase.name}): Failed — ${phaseResult.text.slice(0, 100)}`);
            lastPhaseSucceeded = false;

            // Still commit whatever was created, then continue
            await commitPhase(workspacePath, `wip: partial ${phase.name.toLowerCase()}`);
            await progress(`Phase ${phaseNum} had issues but continuing: ${phaseResult.text.slice(0, 80)}`);
        }
    }

    // -----------------------------------------------------------------------
    // Final push
    // -----------------------------------------------------------------------
    await progress("Pushing to GitHub...", { current_action: "Pushing..." });
    const pushResult = await pushRepo(workspacePath, `feat: complete ${projectName}`);

    let finalText: string;
    if (pushResult.success) {
        const completedCount = jobPhases.filter((p) => p.status === "complete").length;
        finalText =
            `Project "${projectName}" built and pushed!\n\n` +
            `Phases completed: ${completedCount}/${totalPhases}\n` +
            phaseSummaries.join("\n") +
            (repoUrl ? `\n\nRepo: ${repoUrl}` : "") +
            (allFilesCreated.length > 0 ? `\n\nFiles created: ${allFilesCreated.length}` : "");

        await progress("Done! Project pushed to GitHub.", {
            status: "complete",
            current_action: "Done",
            files_created: allFilesCreated,
            phases: jobPhases,
            finished_at: Date.now(),
        });
    } else {
        finalText =
            `Project "${projectName}" was built but push failed: ${pushResult.error}\n\n` +
            phaseSummaries.join("\n") +
            (repoUrl ? `\n\nRepo: ${repoUrl}` : "");

        await progress(`Build complete but push failed: ${pushResult.error}`, {
            status: "failed",
            errors: [pushResult.error ?? "Push failed"],
            phases: jobPhases,
            finished_at: Date.now(),
        });
    }

    await clearActiveJob(senderId);
    const assistantMessage: ModelMessage = { role: "assistant", content: finalText };
    await saveChatMessages(senderId, [userMessage, assistantMessage]);

    return { response: { text: finalText } };
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
        // Use multi-phase for requests that sound like building a full project
        const lowerText = text.toLowerCase();
        const isLargeProject =
            (lowerText.includes("build") || lowerText.includes("create")) &&
            (lowerText.includes("website") || lowerText.includes("web app") ||
             lowerText.includes("application") || lowerText.includes("full") ||
             lowerText.includes("e-commerce") || lowerText.includes("shopping") ||
             lowerText.includes("dashboard") || lowerText.includes("platform") ||
             lowerText.includes("multi-page") || lowerText.includes("full-stack"));

        if (isLargeProject) {
            console.info(`[orchestrator] Using multi-phase flow for sender="${senderId}"`);
            return processDevMultiPhase(senderId, text, history, options.onProgress);
        }

        return processDev(senderId, text, history, options.onProgress);
    }
    return processNormal(senderId, text, history);
}

