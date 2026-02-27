/**
 * lib/ai/tools.ts — LLM Tools (Hybrid RAG & Patch Workflow)
 * =========================================================
 * Defines the tools exposed to the language model via the Vercel AI SDK.
 *
 * Tools:
 *   - search_local_codebase   — Proxy to the local RAG daemon for code search.
 *   - request_patch_approval  — Ask the user to approve a generated patch.
 *   - spotify_*               — Control Spotify via the daemon (play, pause, next, etc.).
 */

import { tool } from "ai";
import { z } from "zod";
import type { InteractiveButton, StandardResponse } from "@/types/messaging";
import { stagePatch, stagePush, stagePushOnly } from "@/lib/redis";

const LOCAL_DAEMON_BASE = () => process.env.LOCAL_DAEMON_URL?.replace(/\/+$/, "") ?? "";

/** Resolve a repo name (e.g. "Eureka") to an absolute workspace path via the daemon. Returns null if not found or daemon unavailable. */
async function resolveRepoName(repoName: string): Promise<string | null> {
    const base = LOCAL_DAEMON_BASE();
    if (!base) return null;
    try {
        const res = await fetch(`${base}/git/find-repos`, { method: "GET" });
        const data = (await res.json().catch(() => ({}))) as { repos?: Array<{ name: string; path: string }> };
        const repos = data.repos ?? [];
        const name = repoName.trim().toLowerCase();
        const match = repos.find((r) => r.name.toLowerCase() === name);
        return match ? match.path : null;
    } catch {
        return null;
    }
}

async function spotifyDaemonCall(
    method: "GET" | "POST",
    path: string,
    body?: object,
): Promise<{ success: boolean; message?: string; error?: string; [k: string]: unknown }> {
    const base = LOCAL_DAEMON_BASE();
    if (!base) {
        return { success: false, error: "LOCAL_DAEMON_URL is not configured." };
    }
    const url = `${base}${path}`;
    try {
        const res = await fetch(url, {
            method,
            headers: { "Content-Type": "application/json" },
            ...(body && method === "POST" ? { body: JSON.stringify(body) } : {}),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            return {
                success: false,
                error: (data as { detail?: string }).detail ?? `Daemon returned ${res.status}`,
            };
        }
        return { ...data, success: (data as { success?: boolean }).success !== false };
    } catch (e) {
        return {
            success: false,
            error: e instanceof Error ? e.message : "Failed to reach the daemon.",
        };
    }
}

/**
 * Tool: search_local_codebase
 * ---------------------------
 * Delegates semantic code search to the developer's local RAG daemon.
 *
 * The daemon is expected to expose an HTTP endpoint:
 *   POST ${LOCAL_DAEMON_URL}/search
 *   Body: { query: string }
 *
 * The response body (typically a list of code snippets) is returned verbatim
 * to the model so it can decide how to use it.
 */
export const searchLocalCodebase = tool({
    description:
        "Search the developer's local codebase via the RAG daemon and return relevant code snippets.",
    inputSchema: z.object({
        query: z
            .string()
            .min(1, "Search query must not be empty.")
            .describe("Natural language description of what to search for in the codebase."),
    }),
    // eslint-disable-next-line @typescript-eslint/require-await
    async execute({ query }: { query: string }): Promise<unknown> {
        const baseUrl = process.env.LOCAL_DAEMON_URL;

        if (!baseUrl) {
            const message =
                "LOCAL_DAEMON_URL is not configured. The local code search daemon is currently unavailable.";
            console.error(
                "[tools.search_local_codebase] Missing LOCAL_DAEMON_URL environment variable.",
            );
            return {
                error: message,
            };
        }

        const url = `${baseUrl.replace(/\/+$/, "")}/search`;

        try {
            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                const bodyText = await response.text().catch(() => "<unreadable body>");
                const errorMessage = `[tools.search_local_codebase] Daemon responded with ${response.status} ${response.statusText}`;

                console.error(errorMessage, "Response body:", bodyText);

                return {
                    error: "The local code search daemon responded with an error.",
                    status: response.status,
                };
            }

            try {
                const json = await response.json();
                return json;
            } catch (parseError) {
                console.error(
                    "[tools.search_local_codebase] Failed to parse daemon JSON response:",
                    parseError,
                );
                return {
                    error: "The local code search daemon returned invalid JSON.",
                };
            }
        } catch (networkError) {
            console.error(
                "[tools.search_local_codebase] Failed to reach local daemon:",
                networkError,
            );

            return {
                error:
                    "Failed to reach the local code search daemon. " +
                    "Ensure your development machine is online and the tunnel is running.",
            };
        }
    },
});

/**
 * Tool: list_workspace_folders
 * ----------------------------
 * Asks the local daemon to list the top-level folders in each allowed workspace.
 * This is a safe, read-only view of the filesystem that the LLM can use to
 * orient itself ("what projects are available?").
 */
export const listWorkspaceFolders = tool({
    description:
        "List the top-level folders in each allowed workspace on the local machine.",
    // No input parameters — just list everything in ALLOWED_WORKSPACES.
    inputSchema: z
        .object({})
        .describe("No arguments. Call this to get a list of workspace folders."),
    // eslint-disable-next-line @typescript-eslint/require-await
    async execute(): Promise<unknown> {
        const baseUrl = process.env.LOCAL_DAEMON_URL;

        if (!baseUrl) {
            console.error(
                "[tools.list_workspace_folders] Missing LOCAL_DAEMON_URL environment variable.",
            );
            return {
                error:
                    "LOCAL_DAEMON_URL is not configured. The local daemon is currently unavailable.",
            };
        }

        const url = `${baseUrl.replace(/\/+$/, "")}/list-folders`;

        try {
            const response = await fetch(url, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
            });

            if (!response.ok) {
                const bodyText = await response.text().catch(() => "<unreadable body>");
                console.error(
                    "[tools.list_workspace_folders] Daemon responded with error:",
                    response.status,
                    response.statusText,
                    bodyText,
                );
                return {
                    error:
                        "The local daemon responded with an error while listing folders. " +
                        "Check the daemon logs for details.",
                    status: response.status,
                };
            }

            try {
                return await response.json();
            } catch (parseError) {
                console.error(
                    "[tools.list_workspace_folders] Failed to parse JSON response:",
                    parseError,
                );
                return {
                    error:
                        "The local daemon returned invalid JSON for the folder listing request.",
                };
            }
        } catch (networkError) {
            console.error(
                "[tools.list_workspace_folders] Failed to reach local daemon:",
                networkError,
            );
            return {
                error:
                    "Failed to reach the local daemon to list folders. " +
                    "Ensure your development machine is online and the daemon is running.",
            };
        }
    },
});

/**
 * Tool: list_git_repos
 * --------------------
 * Ask the local daemon to scan ALLOWED_WORKSPACES for git repositories.
 * This lets the LLM resolve human-friendly names like "Eureka" to concrete
 * repo paths such as "C:\\Users\\madus\\Desktop\\Eureka".
 */
export const listGitRepos = tool({
    description:
        "Find git repositories under the allowed workspaces. Returns a list of repo names and paths. " +
        "For uncommitted changes, removing lines, or removing lines matching text, prefer calling get_uncommitted_changes, remove_line, or remove_lines_matching with repo_name (e.g. 'Eureka') — they resolve the repo automatically. Use this tool when you need to list or discover repos, or when prepare_push_approval needs a path.",
    // No input parameters — just scan all allowed workspaces.
    inputSchema: z
        .object({})
        .describe("No arguments. Call this to get a list of local git repositories."),
    // eslint-disable-next-line @typescript-eslint/require-await
    async execute(): Promise<unknown> {
        const baseUrl = process.env.LOCAL_DAEMON_URL;

        if (!baseUrl) {
            console.error(
                "[tools.list_git_repos] Missing LOCAL_DAEMON_URL environment variable.",
            );
            return {
                error:
                    "LOCAL_DAEMON_URL is not configured. The local daemon is currently unavailable.",
            };
        }

        const base = baseUrl.replace(/\/+$/, "");
        const url = `${base}/git/find-repos`;

        try {
            const response = await fetch(url, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
            });

            if (!response.ok) {
                const bodyText = await response.text().catch(() => "<unreadable body>");
                console.error(
                    "[tools.list_git_repos] Daemon responded with error:",
                    response.status,
                    response.statusText,
                    bodyText,
                );
                return {
                    error:
                        "The local daemon responded with an error while scanning for git repositories. " +
                        "Check the daemon logs for details.",
                    status: response.status,
                };
            }

            try {
                return await response.json();
            } catch (parseError) {
                console.error(
                    "[tools.list_git_repos] Failed to parse JSON response:",
                    parseError,
                );
                return {
                    error: "The local daemon returned invalid JSON for the git repo listing request.",
                };
            }
        } catch (networkError) {
            console.error(
                "[tools.list_git_repos] Failed to reach local daemon:",
                networkError,
            );
            return {
                error:
                    "Failed to reach the local daemon to list git repositories. " +
                    "Ensure your development machine is online and the daemon is running.",
            };
        }
    },
});

/**
 * Tool: get_uncommitted_changes
 * -----------------------------
 * Get uncommitted changes (diff + status) for one repo or for all repos.
 * Accepts either workspace_path (absolute) or repo_name (e.g. "Eureka"); resolves repo_name automatically.
 */
export const getUncommittedChanges = tool({
    description:
        "Get uncommitted changes (diff and status) for a git repo or for all repos. " +
        "Use when the user asks about uncommitted changes (e.g. 'any uncommitted changes in Eureka', 'show changes in all repos'). " +
        "Pass repo_name (e.g. 'Eureka') to target one repo, or workspace_path. Pass neither to get a summary for all repos.",
    inputSchema: z.object({
        workspace_path: z
            .string()
            .optional()
            .describe("Absolute path to one repo. Use repo_name instead if the user said a name like 'Eureka'."),
        repo_name: z
            .string()
            .optional()
            .describe("Repo name the user said (e.g. 'Eureka'). The tool will resolve it to a path. Use this when the user names a repo."),
    }),
    async execute({
        workspace_path,
        repo_name,
    }: {
        workspace_path?: string;
        repo_name?: string;
    }): Promise<{ text: string } | { error: string }> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { error: "LOCAL_DAEMON_URL is not configured." };
        }

        let resolvedPath = workspace_path?.trim();
        if (repo_name?.trim() && !resolvedPath) {
            const path = await resolveRepoName(repo_name);
            if (!path) return { error: `Repository "${repo_name}" not found.` };
            resolvedPath = path;
        }

        const maxDiffLen = 3200;
        const maxDiffLenAllRepos = 1200;

        type DiffData = {
            success?: boolean;
            message?: string;
            has_changes?: boolean;
            diff?: string;
            status_short?: string;
        };

        async function fetchDiff(path: string): Promise<DiffData> {
            const url = `${base}/git/uncommitted-diff?${new URLSearchParams({ workspace_path: path }).toString()}`;
            const res = await fetch(url, { method: "GET" });
            return (await res.json().catch(() => ({}))) as DiffData;
        }

        if (resolvedPath) {
            const data = await fetchDiff(resolvedPath);
            if (!data.success) {
                return { error: data.message ?? "Could not get repo status." };
            }
            if (!data.has_changes) {
                return {
                    text: `**${resolvedPath}**\n\nNo uncommitted changes. Working tree clean.`,
                };
            }
            const diff = (data.diff ?? "").trim();
            const statusShort = (data.status_short ?? "").trim();
            const truncated = diff.length > maxDiffLen;
            const diffText = truncated ? diff.slice(0, maxDiffLen) + "\n\n… (truncated)" : diff;
            return {
                text:
                    `**${resolvedPath}**\n\n` +
                    (statusShort ? `Status: ${statusShort}\n\n` : "") +
                    "Uncommitted changes:\n\n```\n" +
                    diffText +
                    "\n```",
            };
        }

        // All repos: list repos then fetch diff for each.
        const listUrl = `${base}/git/find-repos`;
        let repos: Array<{ name: string; path: string }>;
        try {
            const listRes = await fetch(listUrl, { method: "GET" });
            const listData = (await listRes.json().catch(() => ({}))) as { repos?: Array<{ name: string; path: string }> };
            repos = listData.repos ?? [];
        } catch {
            return { error: "Failed to list git repositories." };
        }
        if (repos.length === 0) {
            return { text: "No git repositories found under the allowed workspaces." };
        }

        const sections: string[] = [];
        for (const repo of repos) {
            const data = await fetchDiff(repo.path);
            if (!data.success) {
                sections.push(`**${repo.name}** (${repo.path})\n\nError: ${data.message ?? "Could not get status."}\n`);
                continue;
            }
            if (!data.has_changes) {
                sections.push(`**${repo.name}** (${repo.path})\n\nNo uncommitted changes.\n`);
                continue;
            }
            const diff = (data.diff ?? "").trim();
            const statusShort = (data.status_short ?? "").trim();
            const truncated = diff.length > maxDiffLenAllRepos;
            const diffText = truncated ? diff.slice(0, maxDiffLenAllRepos) + "\n\n… (truncated)" : diff;
            sections.push(
                `**${repo.name}** (${repo.path})\n\n` +
                    (statusShort ? `Status: ${statusShort}\n\n` : "") +
                    "```\n" +
                    diffText +
                    "\n```\n",
            );
        }
        return {
            text: "Summary for all git repos:\n\n" + sections.join("\n---\n\n"),
        };
    },
});

/**
 * Tool: list_folder_contents
 * --------------------------
 * Ask the local daemon to list the immediate files and folders inside a
 * specific directory within an allowed workspace.
 */
export const listFolderContents = tool({
    description:
        "List all files and folders directly inside a given directory within an allowed workspace.",
    inputSchema: z.object({
        folder_path: z
            .string()
            .min(1, "folder_path must not be empty.")
            .describe(
                "Absolute path to a folder within one of the allowed workspaces. " +
                    'For example: "C:\\\\Users\\\\madus\\\\Desktop\\\\Capestone".',
            ),
    }),
    // eslint-disable-next-line @typescript-eslint/require-await
    async execute({ folder_path }: { folder_path: string }): Promise<unknown> {
        const baseUrl = process.env.LOCAL_DAEMON_URL;

        if (!baseUrl) {
            console.error(
                "[tools.list_folder_contents] Missing LOCAL_DAEMON_URL environment variable.",
            );
            return {
                error:
                    "LOCAL_DAEMON_URL is not configured. The local daemon is currently unavailable.",
            };
        }

        const base = baseUrl.replace(/\/+$/, "");
        const url = `${base}/list-folder-contents?${new URLSearchParams({
            folder_path,
        }).toString()}`;

        try {
            const response = await fetch(url, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
            });

            if (!response.ok) {
                const bodyText = await response.text().catch(() => "<unreadable body>");
                console.error(
                    "[tools.list_folder_contents] Daemon responded with error:",
                    response.status,
                    response.statusText,
                    bodyText,
                );
                return {
                    error:
                        "The local daemon responded with an error while listing folder contents. " +
                        "Check the daemon logs for details.",
                    status: response.status,
                };
            }

            try {
                return await response.json();
            } catch (parseError) {
                console.error(
                    "[tools.list_folder_contents] Failed to parse JSON response:",
                    parseError,
                );
                return {
                    error:
                        "The local daemon returned invalid JSON for the folder contents request.",
                };
            }
        } catch (networkError) {
            console.error(
                "[tools.list_folder_contents] Failed to reach local daemon:",
                networkError,
            );
            return {
                error:
                    "Failed to reach the local daemon to list folder contents. " +
                    "Ensure your development machine is online and the daemon is running.",
            };
        }
    },
});

/**
 * Tool: read_file
 * ---------------
 * Read the full text contents of a file within an allowed workspace.
 * Used by the dev agent to inspect source files before editing.
 */
const READ_FILE_MAX_CHARS = 4000;

export const readFile = tool({
    description:
        "Read a file's contents with line numbers. Returns numbered lines (e.g. '42| code here'). " +
        "Use start_line and end_line to read a specific range (1-based, inclusive). " +
        "Large files are auto-truncated to ~4000 chars showing the first and last portions. " +
        "IMPORTANT: After reading, note the line numbers — you'll need them for insert_code (after_line).",
    inputSchema: z.object({
        path: z
            .string()
            .min(1, "path must not be empty.")
            .describe("Absolute path to a file within an allowed workspace."),
        start_line: z
            .number()
            .int()
            .min(1)
            .optional()
            .describe("Optional 1-based start line."),
        end_line: z
            .number()
            .int()
            .min(1)
            .optional()
            .describe("Optional 1-based end line (inclusive)."),
    }),
    async execute({
        path,
        start_line,
        end_line,
    }: {
        path: string;
        start_line?: number;
        end_line?: number;
    }): Promise<unknown> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { success: false, error: "LOCAL_DAEMON_URL is not configured." };
        }
        try {
            const body: Record<string, unknown> = { path: path.trim() };
            if (start_line != null) body.start_line = start_line;
            if (end_line != null) body.end_line = end_line;
            const res = await fetch(`${base}/read-file`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = (await res.json().catch(() => ({}))) as {
                success?: boolean;
                path?: string;
                content?: string;
                total_lines?: number;
                error?: string;
            };
            if (!res.ok) {
                return { success: false, error: data.error ?? `Daemon returned ${res.status}` };
            }
            // Auto-truncate for the LLM context window
            if (data.content && data.content.length > READ_FILE_MAX_CHARS) {
                const head = data.content.slice(0, 2000);
                const tail = data.content.slice(-2000);
                data.content =
                    head +
                    `\n\n... [TRUNCATED — file has ${data.total_lines ?? "many"} lines, showing first and last ~50 lines. Use start_line/end_line to read a specific section] ...\n\n` +
                    tail;
            }
            return data;
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Failed to reach the daemon." };
        }
    },
});

/**
 * Tool: run_tests
 * ---------------
 * Run the test suite in a workspace (e.g. npm test, pytest). Only whitelisted
 * commands are allowed. Use after making code changes to verify nothing broke.
 */
export const runTests = tool({
    description:
        "Run the test suite in a given workspace (e.g. npm test or pytest). Use after editing code to verify tests pass. Only whitelisted commands are allowed: npm test, npm run test, yarn test, pnpm test, pytest, python -m pytest, dotnet test.",
    inputSchema: z.object({
        workspace_path: z
            .string()
            .min(1, "workspace_path must not be empty.")
            .describe(
                "Absolute path to the workspace (repo root) where tests should run.",
            ),
        command_line: z
            .string()
            .min(1, "command_line must not be empty.")
            .describe(
                "Command to run, e.g. 'npm test' or 'pytest'. Must be one of: npm test, npm run test, yarn test, pnpm test, pnpm run test, pytest, python -m pytest, dotnet test.",
            ),
    }),
    async execute({
        workspace_path,
        command_line,
    }: {
        workspace_path: string;
        command_line: string;
    }): Promise<unknown> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { success: false, error: "LOCAL_DAEMON_URL is not configured." };
        }
        try {
            const res = await fetch(`${base}/run-command`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    workspace_path: workspace_path.trim(),
                    command_line: command_line.trim(),
                }),
            });
            const data = (await res.json().catch(() => ({}))) as {
                success?: boolean;
                stdout?: string;
                stderr?: string;
                exit_code?: number;
                error?: string;
            };
            if (!res.ok) {
                return {
                    success: false,
                    error: data.error ?? `Daemon returned ${res.status}`,
                };
            }
            return data;
        } catch (e) {
            return {
                success: false,
                error: e instanceof Error ? e.message : "Failed to reach the daemon.",
            };
        }
    },
});

/**
 * Tool: insert_code
 * -----------------
 * Add new code to a file by inserting after a specific line number.
 * The LLM reads the file first (via read_file), picks a line number,
 * and the daemon generates a perfect diff. Best for adding new features.
 */
export const insertCode = tool({
    description:
        "Insert new code into a file after a specific line number. Use this when ADDING new features, endpoints, functions, or classes. " +
        "First call read_file to see the file contents with line numbers. Then call this tool with the line number to insert after and the new code. " +
        "after_line is 1-based; set to 0 to prepend to the start of the file. " +
        "The daemon generates a correct unified diff — you do NOT need to write any diff yourself.",
    inputSchema: z.object({
        repo_name: z
            .string()
            .optional()
            .describe("Optional repo name (e.g. 'Eureka'). Resolved to a workspace path."),
        workspace_path: z
            .string()
            .optional()
            .describe("Optional absolute path to the git repo root."),
        file_path: z
            .string()
            .min(1, "file_path must not be empty.")
            .describe("Path to the file relative to the repo root (e.g. 'rag-daemon/main.py')."),
        after_line: z
            .number()
            .int()
            .min(0)
            .describe("1-based line number to insert after. 0 = prepend to start of file."),
        new_code: z
            .string()
            .min(1, "new_code must not be empty.")
            .describe("The new code to insert. Include all necessary indentation, decorators, imports, blank lines, etc."),
    }),
    async execute({
        repo_name,
        workspace_path,
        file_path,
        after_line,
        new_code,
    }: {
        repo_name?: string;
        workspace_path?: string;
        file_path: string;
        after_line: number;
        new_code: string;
    }): Promise<StandardResponse> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { text: "LOCAL_DAEMON_URL is not configured. Cannot prepare a patch." };
        }
        const DEFAULT_WORKSPACE = process.env.LOCAL_DAEMON_WORKSPACE_PATH ?? "";
        let workspace = workspace_path?.trim();
        if (!workspace && repo_name) {
            workspace = (await resolveRepoName(repo_name)) ?? undefined;
        }
        if (!workspace) workspace = DEFAULT_WORKSPACE;
        if (!workspace) {
            return { text: "Could not determine workspace. Set LOCAL_DAEMON_WORKSPACE_PATH or pass workspace_path." };
        }

        try {
            const res = await fetch(`${base}/edit/insert-code`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    workspace_path: workspace,
                    file_path,
                    after_line,
                    new_code,
                }),
            });
            const data = (await res.json().catch(() => ({}))) as {
                success?: boolean;
                message?: string;
                patch?: string;
            };
            if (!res.ok || data.success === false) {
                const err = data.message ?? `Daemon returned ${res.status}`;
                return { text: `Patch computation failed: ${err}` };
            }
            const patch = data.patch ?? "";
            if (!patch) {
                return { text: "Daemon reported success but returned no patch." };
            }

            console.info("[tools.insert_code] Patch computed by daemon:\n", patch.slice(0, 2000));

            const patchId = await stagePatch(patch, workspace);
            const buttons: ReadonlyArray<InteractiveButton> = [
                { action: `apply_patch:${patchId}`, label: "Approve & Apply" },
            ];
            return {
                text: "I've prepared new code to add. Review and press **Approve & Apply** to apply it.\n\n```\n" + patch.slice(0, 1500) + "\n```",
                interactiveButtons: buttons,
            };
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            return { text: `Failed to reach daemon for patch computation: ${msg}` };
        }
    },
});

/**
 * Tool: edit_file
 * ---------------
 * Propose a code change using search/replace. The daemon generates the diff
 * (via difflib.unified_diff) so the LLM never has to produce raw diff format.
 * Surfaces an "Approve & Apply" button to the user.
 */
export const editFile = tool({
    description:
        "Edit a file by specifying an exact search string and its replacement. " +
        "The daemon will compute a correct unified diff from the search/replace pair. " +
        "Use this to add, change, or remove code. Do NOT write a unified diff yourself. " +
        "Provide: file_path (relative to repo root), search_string (exact text currently in the file), and replace_string (what it should become). " +
        "To add new code, use search_string to match the lines just BEFORE where the new code should go, and set replace_string to those same lines plus the new code appended. " +
        "To delete code, set replace_string to empty string.",
    inputSchema: z.object({
        repo_name: z
            .string()
            .optional()
            .describe("Optional human-friendly repo name (e.g. 'Eureka'). Resolved to a workspace path."),
        workspace_path: z
            .string()
            .optional()
            .describe("Optional absolute path to the git repo root. If omitted, uses repo_name or default."),
        file_path: z
            .string()
            .min(1, "file_path must not be empty.")
            .describe("Path to the file relative to the repo root (e.g. 'rag-daemon/main.py')."),
        search_string: z
            .string()
            .min(1, "search_string must not be empty.")
            .describe("Exact substring to find in the file. Must match the file content exactly (including whitespace and newlines)."),
        replace_string: z
            .string()
            .describe("Replacement text. Can be empty to delete the matched text."),
    }),
    async execute({
        repo_name,
        workspace_path,
        file_path,
        search_string,
        replace_string,
    }: {
        repo_name?: string;
        workspace_path?: string;
        file_path: string;
        search_string: string;
        replace_string: string;
    }): Promise<StandardResponse> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { text: "LOCAL_DAEMON_URL is not configured. Cannot prepare a patch." };
        }
        const DEFAULT_WORKSPACE = process.env.LOCAL_DAEMON_WORKSPACE_PATH ?? "";
        let workspace = workspace_path?.trim();
        if (!workspace && repo_name) {
            workspace = (await resolveRepoName(repo_name)) ?? undefined;
        }
        if (!workspace) workspace = DEFAULT_WORKSPACE;
        if (!workspace) {
            return { text: "Could not determine workspace. Set LOCAL_DAEMON_WORKSPACE_PATH or pass workspace_path." };
        }

        try {
            const res = await fetch(`${base}/edit/compute-replace-patch`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    workspace_path: workspace,
                    file_path,
                    search_string,
                    replace_string,
                }),
            });
            const data = (await res.json().catch(() => ({}))) as {
                success?: boolean;
                message?: string;
                patch?: string;
            };
            if (!res.ok || data.success === false) {
                const err = data.message ?? `Daemon returned ${res.status}`;
                return { text: `Patch computation failed: ${err}` };
            }
            const patch = data.patch ?? "";
            if (!patch) {
                return { text: "Daemon reported success but returned no patch." };
            }

            console.info("[tools.edit_file] Patch computed by daemon:\n", patch.slice(0, 2000));

            const patchId = await stagePatch(patch, workspace);
            const buttons: ReadonlyArray<InteractiveButton> = [
                { action: `apply_patch:${patchId}`, label: "Approve & Apply" },
            ];
            return {
                text: "I've prepared a code change. Review and press **Approve & Apply** to apply it.\n\n```\n" + patch.slice(0, 1500) + "\n```",
                interactiveButtons: buttons,
            };
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            return { text: `Failed to reach daemon for patch computation: ${msg}` };
        }
    },
});

/**
 * Tool: request_patch_approval (legacy)
 * Kept for backward compatibility with existing staged patches. Prefer edit_file for new edits.
 */
export const requestPatchApproval = tool({
    description:
        "DEPRECATED — prefer edit_file instead. Only use this if you already have a precomputed valid unified diff. " +
        "Ask the user to approve a raw unified diff patch before applying with git.",
    inputSchema: z.object({
        patch_string: z
            .string()
            .min(1, "Patch string must not be empty.")
            .describe("A valid unified diff."),
        workspace_path: z
            .string()
            .optional()
            .describe("Absolute path to the git repo root."),
    }),
    async execute({
        patch_string,
        workspace_path,
    }: {
        patch_string: string;
        workspace_path?: string;
    }): Promise<StandardResponse> {
        const preview =
            patch_string.length > 2000
                ? `${patch_string.slice(0, 2000)}… [truncated]`
                : patch_string;

        console.info(
            "[tools.request_patch_approval] Patch generated for user approval:\n",
            preview,
        );

        const patchId = await stagePatch(patch_string, workspace_path);

        const buttons: ReadonlyArray<InteractiveButton> = [
            {
                action: `apply_patch:${patchId}`,
                label: "Approve & Apply",
            },
        ];

        return {
            text: "I have generated a patch. Do you want to apply it?",
            interactiveButtons: buttons,
        };
    },
});

/**
 * Tool: remove_line
 * -----------------
 * Ask the daemon to compute a patch that removes a specific line (or range) from a file.
 * Accepts repo_name (e.g. "Eureka") or workspace_path; resolves repo_name automatically.
 */
export const removeLine = tool({
    description:
        "Remove a specific line (or range of lines) from a file in a repo. " +
        "Use when the user asks to remove line N from a file (e.g. 'Remove line 1 from rag-daemon/config.py in Eureka'). " +
        "Pass repo_name (e.g. 'Eureka') if the user named a repo, or workspace_path. file_path is relative to the repo (e.g. rag-daemon/config.py).",
    inputSchema: z.object({
        workspace_path: z.string().optional().describe("Absolute path to the git repository root."),
        repo_name: z.string().optional().describe("Repo name the user said (e.g. 'Eureka'). Use when the user names a repo."),
        file_path: z
            .string()
            .min(1, "file_path must not be empty.")
            .describe("Path to the file relative to the repo root (e.g. rag-daemon/config.py or src/main.py)."),
        line_number: z.number().int().min(1).describe("1-based line number to remove."),
        line_end: z
            .number()
            .int()
            .min(1)
            .optional()
            .describe("Optional 1-based end line for a range (inclusive)."),
    }),
    async execute({
        workspace_path,
        repo_name,
        file_path,
        line_number,
        line_end,
    }: {
        workspace_path?: string;
        repo_name?: string;
        file_path: string;
        line_number: number;
        line_end?: number;
    }): Promise<StandardResponse | { success: boolean; error: string }> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { success: false, error: "LOCAL_DAEMON_URL is not configured." };
        }
        let resolvedPath = workspace_path?.trim();
        if (repo_name?.trim() && !resolvedPath) {
            const path = await resolveRepoName(repo_name);
            if (!path) return { success: false, error: `Repository "${repo_name}" not found.` };
            resolvedPath = path;
        }
        if (!resolvedPath) {
            return { success: false, error: "Provide either workspace_path or repo_name." };
        }
        const url = `${base}/edit/compute-remove-line-patch`;
        let data: { success?: boolean; message?: string; patch?: string };
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    workspace_path: resolvedPath,
                    file_path: file_path.replace(/\\/g, "/"),
                    line_number,
                    ...(line_end != null ? { line_end } : {}),
                }),
            });
            data = (await res.json().catch(() => ({}))) as typeof data;
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Failed to reach the daemon." };
        }
        if (!data.success) {
            return { success: false, error: data.message ?? "Could not compute patch." };
        }
        const patch = (data.patch ?? "").trim();
        if (!patch) {
            return { text: "No patch was produced (check line number is in range)." };
        }
        const patchId = await stagePatch(patch, resolvedPath);
        return {
            text: `I have prepared a patch to remove line ${line_number}${line_end != null ? `–${line_end}` : ""} from \`${file_path}\`. Do you want to apply it?`,
            interactiveButtons: [{ action: `apply_patch:${patchId}`, label: "Approve & Apply" }],
        };
    },
});

/**
 * Tool: remove_lines_matching
 * ---------------------------
 * Ask the daemon to compute a patch that removes all lines containing a pattern.
 * Accepts repo_name (e.g. "Eureka") or workspace_path; resolves repo_name automatically.
 */
export const removeLinesMatching = tool({
    description:
        "Remove all lines that contain a given pattern (literal text) from a file or from all files in a repo. " +
        "Use when the user asks to remove lines containing some text (e.g. 'Remove #testing from Eureka', 'Remove #testing from all files in Eureka'). " +
        "Pass repo_name (e.g. 'Eureka') if the user named a repo, or workspace_path. For one file pass file_path; for all files omit file_path.",
    inputSchema: z.object({
        workspace_path: z.string().optional().describe("Absolute path to the git repository root."),
        repo_name: z.string().optional().describe("Repo name the user said (e.g. 'Eureka'). Use when the user names a repo."),
        pattern: z
            .string()
            .min(1, "pattern must not be empty.")
            .describe("Literal substring to find; every line containing this will be removed."),
        file_path: z
            .string()
            .optional()
            .describe(
                "If set, only this file (relative to repo). If omitted, all files in the repo are scanned.",
            ),
        path_glob: z
            .string()
            .optional()
            .describe("When scanning multiple files, restrict to paths matching this glob (e.g. **/*.py). Default: **/*"),
    }),
    async execute({
        workspace_path,
        repo_name,
        pattern,
        file_path,
        path_glob,
    }: {
        workspace_path?: string;
        repo_name?: string;
        pattern: string;
        file_path?: string;
        path_glob?: string;
    }): Promise<StandardResponse | { success: boolean; error: string }> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { success: false, error: "LOCAL_DAEMON_URL is not configured." };
        }
        let resolvedPath = workspace_path?.trim();
        if (repo_name?.trim() && !resolvedPath) {
            const path = await resolveRepoName(repo_name);
            if (!path) return { success: false, error: `Repository "${repo_name}" not found.` };
            resolvedPath = path;
        }
        if (!resolvedPath) {
            return { success: false, error: "Provide either workspace_path or repo_name." };
        }
        const url = `${base}/edit/compute-remove-lines-matching-patch`;
        let data: { success?: boolean; message?: string; patch?: string; files_affected?: number };
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    workspace_path: resolvedPath,
                    pattern,
                    ...(file_path != null && file_path !== "" ? { file_path: file_path.replace(/\\/g, "/") } : {}),
                    ...(path_glob != null && path_glob !== "" ? { path_glob } : {}),
                }),
            });
            data = (await res.json().catch(() => ({}))) as typeof data;
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Failed to reach the daemon." };
        }
        if (!data.success) {
            return { success: false, error: data.message ?? "Could not compute patch." };
        }
        const patch = (data.patch ?? "").trim();
        const filesAffected = data.files_affected ?? 0;
        if (!patch) {
            return {
                text: filesAffected === 0
                    ? `No lines containing "${pattern}" were found${file_path ? ` in ${file_path}` : " in any file"}.`
                    : "No patch was produced.",
            };
        }
        const patchId = await stagePatch(patch, resolvedPath);
        const scope = file_path ? ` in \`${file_path}\`` : ` across ${filesAffected} file(s)`;
        return {
            text: `I have prepared a patch to remove all lines containing "${pattern}"${scope}. Do you want to apply it?`,
            interactiveButtons: [{ action: `apply_patch:${patchId}`, label: "Approve & Apply" }],
        };
    },
});

/**
 * Tool: prepare_push_approval
 * ---------------------------
 * Two-step flow: (1) Show uncommitted diff and ask for a commit message.
 * (2) When the user replies with a message, call again with that message to show
 * "Approve & Push". Pushing runs git add -A, commit, and push on the daemon.
 */
export const preparePushApproval = tool({
    description:
        "Push flow for a git repo. Call WITHOUT commit_message first: shows uncommitted diff and asks the user for a commit message. " +
        "When the user replies with their commit message (or 'default'), call AGAIN with that as commit_message: then show the 'Approve & Push' button. " +
        "Requires the full path to the repo root (e.g. C:\\Users\\madus\\Desktop\\Eureka).",
    inputSchema: z.object({
        workspace_path: z
            .string()
            .min(1, "workspace_path must not be empty.")
            .describe("Absolute path to the git repository root."),
        commit_message: z
            .string()
            .optional()
            .describe(
                "Commit message. Omit on first call (to ask the user). On second call, use the user's reply or 'default' for 'Update from Eureka'.",
            ),
    }),
    async execute({
        workspace_path,
        commit_message,
    }: {
        workspace_path: string;
        commit_message?: string;
    }): Promise<StandardResponse | { success: boolean; error: string }> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { success: false, error: "LOCAL_DAEMON_URL is not configured." };
        }
        const url = `${base}/git/uncommitted-diff?${new URLSearchParams({ workspace_path }).toString()}`;
        let data: {
            success?: boolean;
            message?: string;
            has_changes?: boolean;
            diff?: string;
            status_short?: string;
        };
        try {
            const res = await fetch(url, { method: "GET" });
            data = (await res.json().catch(() => ({}))) as typeof data;
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Failed to reach the daemon." };
        }
        if (!data.success) {
            return { success: false, error: data.message ?? "Could not get repo status." };
        }
        if (!data.has_changes) {
            return {
                text: "There are no uncommitted changes in this repo. Nothing to push.",
            };
        }

        const maxDiffLen = 3600;
        const diff = (data.diff ?? "").trim();
        const truncated = diff.length > maxDiffLen;
        const diffText = truncated ? diff.slice(0, maxDiffLen) + "\n\n… (truncated)" : diff;

        const askedForMessage = commit_message === undefined || commit_message.trim() === "";
        const finalMessage =
            askedForMessage ? "Update from Eureka" : commit_message!.trim().toLowerCase() === "default" ? "Update from Eureka" : commit_message!.trim();

        if (askedForMessage) {
            return {
                text:
                    "Uncommitted changes (all changes will be added with `git add -A` before commit):\n\n```\n" +
                    diffText +
                    "\n```\n\nWhat commit message would you like? Reply with your message, or say **default** to use \"Update from Eureka\".",
            };
        }

        const pushId = await stagePush(workspace_path, finalMessage);
        return {
            text:
                "Commit message: **" +
                finalMessage +
                "**\n\nUncommitted changes:\n\n```\n" +
                diffText +
                "\n```\n\nApprove to add all changes, commit, and push?",
            interactiveButtons: [
                { action: `push:${pushId}`, label: "Approve & Push" },
            ],
        };
    },
});

/**
 * Tool: prepare_push_only_approval
 * --------------------------------
 * Push-only flow: ask the user to approve running `git push` in a repo that
 * already has the desired commits. This does NOT stage or commit any changes.
 */
export const preparePushOnlyApproval = tool({
    description:
        "Push-only flow for a git repo. Use this when the user wants to push existing commits, even if there are no uncommitted changes. " +
        "Call this with the absolute path to the repository; it will show an 'Approve & Push' button.",
    inputSchema: z.object({
        workspace_path: z
            .string()
            .min(1, "workspace_path must not be empty.")
            .describe("Absolute path to the git repository root."),
    }),
    async execute({
        workspace_path,
    }: {
        workspace_path: string;
    }): Promise<StandardResponse | { success: boolean; error: string }> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { success: false, error: "LOCAL_DAEMON_URL is not configured." };
        }

        const pushOnlyId = await stagePushOnly(workspace_path);
        return {
            text:
                "This will run `git push` in the repository:\n" +
                `\`${workspace_path}\`.\n\nApprove to push the current branch?`,
            interactiveButtons: [
                { action: `push_only:${pushOnlyId}`, label: "Approve & Push" },
            ],
        };
    },
});

/**
 * Tool: delete_path
 * -----------------
 * Ask the local daemon to delete a file or folder within an allowed workspace.
 * Use when the user asks to delete, remove, or get rid of a specific file or folder.
 */
export const deletePath = tool({
    description:
        "Delete a file or folder within an allowed workspace. Use when the user asks to delete, remove, or get rid of a specific file or folder. " +
        "Requires the full absolute path (e.g. C:\\Users\\madus\\Desktop\\Capestone). Cannot delete an entire workspace root.",
    inputSchema: z.object({
        path: z
            .string()
            .min(1, "path must not be empty.")
            .describe(
                "Absolute path to the file or folder to delete. Must be inside an allowed workspace.",
            ),
    }),
    // eslint-disable-next-line @typescript-eslint/require-await
    async execute({ path: pathToDelete }: { path: string }): Promise<unknown> {
        const baseUrl = process.env.LOCAL_DAEMON_URL;

        if (!baseUrl) {
            console.error(
                "[tools.delete_path] Missing LOCAL_DAEMON_URL environment variable.",
            );
            return {
                success: false,
                error: "LOCAL_DAEMON_URL is not configured. The local daemon is currently unavailable.",
            };
        }

        const url = `${baseUrl.replace(/\/+$/, "")}/delete-path`;

        try {
            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ path: pathToDelete }),
            });

            if (!response.ok) {
                const bodyText = await response.text().catch(() => "<unreadable body>");
                console.error(
                    "[tools.delete_path] Daemon responded with error:",
                    response.status,
                    response.statusText,
                    bodyText,
                );
                try {
                    const errJson = JSON.parse(bodyText);
                    return {
                        success: false,
                        error: errJson.detail ?? `Daemon error: ${response.status}`,
                    };
                } catch {
                    return {
                        success: false,
                        error: `The daemon responded with ${response.status}. ${bodyText.slice(0, 200)}`,
                    };
                }
            }

            try {
                return await response.json();
            } catch (parseError) {
                console.error(
                    "[tools.delete_path] Failed to parse JSON response:",
                    parseError,
                );
                return {
                    success: false,
                    error: "The local daemon returned invalid JSON.",
                };
            }
        } catch (networkError) {
            console.error(
                "[tools.delete_path] Failed to reach local daemon:",
                networkError,
            );
            return {
                success: false,
                error:
                    "Failed to reach the local daemon. Ensure your machine is online and the daemon is running.",
            };
        }
    },
});

// ---------------------------------------------------------------------------
// Spotify control (via daemon — requires daemon to have SPOTIFY_* configured)
// ---------------------------------------------------------------------------

export const spotifyPlay = tool({
    description: "Start or resume Spotify playback on the user's active device.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("POST", "/spotify/play");
    },
});

export const spotifyPause = tool({
    description: "Pause Spotify playback.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("POST", "/spotify/pause");
    },
});

export const spotifyNext = tool({
    description: "Skip to the next Spotify track.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("POST", "/spotify/next");
    },
});

export const spotifyPrevious = tool({
    description: "Skip to the previous Spotify track.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("POST", "/spotify/previous");
    },
});

export const spotifyVolume = tool({
    description: "Set Spotify volume. Use when the user asks to turn volume up/down or set a specific level.",
    inputSchema: z.object({
        volume_percent: z
            .number()
            .min(0)
            .max(100)
            .describe("Volume level from 0 to 100."),
    }),
    async execute({ volume_percent }: { volume_percent: number }): Promise<unknown> {
        return spotifyDaemonCall("POST", "/spotify/volume", { volume_percent });
    },
});

export const spotifyStatus = tool({
    description: "Get current Spotify playback status (what's playing, device, etc.).",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("GET", "/spotify/status");
    },
});

export const spotifyClose = tool({
    description:
        "Close or quit the Spotify desktop app on the user's PC. Use when the user asks to close Spotify, exit Spotify, or stop the Spotify app.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("POST", "/spotify/close");
    },
});

// ---------------------------------------------------------------------------
// System control (shutdown, sleep, restart) — via daemon on user's PC
// ---------------------------------------------------------------------------

export const systemShutdown = tool({
    description:
        "Shut down the user's PC. Use when the user asks to shut down, turn off, or power off their computer.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("POST", "/system/shutdown");
    },
});

export const systemRestart = tool({
    description:
        "Restart the user's PC. Use when the user asks to restart or reboot their computer.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("POST", "/system/restart");
    },
});

export const systemSleep = tool({
    description:
        "Put the user's PC to sleep (suspend). Use when the user asks to put the computer to sleep.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("POST", "/system/sleep");
    },
});

export const systemLock = tool({
    description:
        "Lock the user's PC (lock screen, like Win+L). Use when the user asks to lock their computer or lock the screen.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return spotifyDaemonCall("POST", "/system/lock");
    },
});

/**
 * Aggregate export used when wiring tools into generateText / streamText.
 * The keys of this object become the tool names available to the model.
 */
export const aiTools = {
    search_local_codebase: searchLocalCodebase,
    insert_code: insertCode,
    edit_file: editFile,
    request_patch_approval: requestPatchApproval,
    remove_line: removeLine,
    remove_lines_matching: removeLinesMatching,
    prepare_push_approval: preparePushApproval,
    prepare_push_only_approval: preparePushOnlyApproval,
    list_git_repos: listGitRepos,
    get_uncommitted_changes: getUncommittedChanges,
    list_workspace_folders: listWorkspaceFolders,
    list_folder_contents: listFolderContents,
    read_file: readFile,
    run_tests: runTests,
    delete_path: deletePath,
    spotify_play: spotifyPlay,
    spotify_pause: spotifyPause,
    spotify_next: spotifyNext,
    spotify_previous: spotifyPrevious,
    spotify_volume: spotifyVolume,
    spotify_status: spotifyStatus,
    spotify_close: spotifyClose,
    system_shutdown: systemShutdown,
    system_restart: systemRestart,
    system_sleep: systemSleep,
    system_lock: systemLock,
};

export type AiTools = typeof aiTools;

