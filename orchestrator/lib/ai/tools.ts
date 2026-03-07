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

import { tool, generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import type { InteractiveButton, StandardResponse } from "@/types/messaging";
import { stagePatch, stagePush, stagePushOnly, pushHostCommand, pollResult } from "@/lib/redis";

/**
 * Push an action to the Redis host-command queue and wait for the result.
 * Replaces all direct HTTP calls to LOCAL_DAEMON_URL.
 */
async function callWorker(
    action: string,
    payload: Record<string, unknown>,
    timeoutMs = 30_000,
): Promise<unknown> {
    const taskId = await pushHostCommand(action, payload);
    return pollResult(taskId, timeoutMs);
}

/** Resolve a repo name (e.g. "Eureka") to an absolute workspace path via the worker. Returns null if not found. */
async function resolveRepoName(repoName: string): Promise<string | null> {
    try {
        const data = (await callWorker("list_git_repos", {})) as { repos?: Array<{ name: string; path: string }> };
        const repos = data.repos ?? [];
        const name = repoName.trim().toLowerCase();
        const match = repos.find((r) => r.name.toLowerCase() === name);
        return match ? match.path : null;
    } catch {
        return null;
    }
}

/** Route a Spotify or system control action through the worker queue. */
async function workerCall(
    action: string,
    payload: Record<string, unknown> = {},
): Promise<{ success: boolean; message?: string; error?: string;[k: string]: unknown }> {
    try {
        const result = (await callWorker(action, payload)) as { success?: boolean; message?: string; error?: string;[k: string]: unknown };
        return { ...result, success: result.success === true };
    } catch (e) {
        return { success: false, error: e instanceof Error ? e.message : "Worker error." };
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
    async execute({ query }: { query: string }): Promise<unknown> {
        try {
            return await callWorker("search", { query });
        } catch (e) {
            return { error: e instanceof Error ? e.message : "Worker error during search." };
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
    async execute(): Promise<unknown> {
        try {
            return await callWorker("list_folders", {});
        } catch (e) {
            return { error: e instanceof Error ? e.message : "Worker error listing folders." };
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
    async execute(): Promise<unknown> {
        try {
            return await callWorker("list_git_repos", {});
        } catch (e) {
            return { error: e instanceof Error ? e.message : "Worker error listing git repos." };
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
        type DiffData = {
            success?: boolean;
            message?: string;
            has_changes?: boolean;
            diff?: string;
            status_short?: string;
        };

        let resolvedPath = workspace_path?.trim();
        if (repo_name?.trim() && !resolvedPath) {
            const path = await resolveRepoName(repo_name);
            if (!path) return { error: `Repository "${repo_name}" not found.` };
            resolvedPath = path;
        }

        const maxDiffLen = 3200;
        const maxDiffLenAllRepos = 1200;

        if (resolvedPath) {
            const data = (await callWorker("uncommitted_diff", { workspace_path: resolvedPath }).catch(() => ({}))) as DiffData;
            if (!data.success) {
                return { error: data.message ?? "Could not get repo status." };
            }
            if (!data.has_changes) {
                return { text: `**${resolvedPath}**\n\nNo uncommitted changes. Working tree clean.` };
            }
            const diff = (data.diff ?? "").trim();
            const statusShort = (data.status_short ?? "").trim();
            const diffText = diff.length > maxDiffLen ? diff.slice(0, maxDiffLen) + "\n\n… (truncated)" : diff;
            return {
                text:
                    `**${resolvedPath}**\n\n` +
                    (statusShort ? `Status: ${statusShort}\n\n` : "") +
                    "Uncommitted changes:\n\n```\n" + diffText + "\n```",
            };
        }

        // All repos: list repos then fetch diff for each.
        let repos: Array<{ name: string; path: string }>;
        try {
            const listData = (await callWorker("list_git_repos", {})) as { repos?: Array<{ name: string; path: string }> };
            repos = listData.repos ?? [];
        } catch {
            return { error: "Failed to list git repositories." };
        }
        if (repos.length === 0) {
            return { text: "No git repositories found under the allowed workspaces." };
        }

        const sections: string[] = [];
        for (const repo of repos) {
            const data = (await callWorker("uncommitted_diff", { workspace_path: repo.path }).catch(() => ({}))) as DiffData;
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
            const diffText = diff.length > maxDiffLenAllRepos ? diff.slice(0, maxDiffLenAllRepos) + "\n\n… (truncated)" : diff;
            sections.push(
                `**${repo.name}** (${repo.path})\n\n` +
                (statusShort ? `Status: ${statusShort}\n\n` : "") +
                "```\n" + diffText + "\n```\n",
            );
        }
        return { text: "Summary for all git repos:\n\n" + sections.join("\n---\n\n") };
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
    async execute({ folder_path }: { folder_path: string }): Promise<unknown> {
        try {
            return await callWorker("list_folder_contents", { folder_path });
        } catch (e) {
            return { error: e instanceof Error ? e.message : "Worker error listing folder." };
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
        try {
            const payload: Record<string, unknown> = { path: path.trim() };
            if (start_line != null) payload.start_line = start_line;
            if (end_line != null) payload.end_line = end_line;
            const data = (await callWorker("read_file", payload)) as {
                success?: boolean;
                content?: string;
                total_lines?: number;
                error?: string;
            };
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
            return { success: false, error: e instanceof Error ? e.message : "Worker error reading file." };
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
        try {
            return await callWorker("run_command", { workspace_path: workspace_path.trim(), command_line: command_line.trim() });
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error running command." };
        }
    },
});

/**
 * Tool: run_scaffold
 * ------------------
 * Run a project scaffolding/initialization command (e.g. npx create-next-app).
 * Longer timeout (300s), shell=True. Only allowed scaffold commands are accepted.
 */
export const runScaffold = tool({
    description:
        "Run a project scaffolding or initialization command. Commands run HEADLESS (no input). Use non-interactive flags only: " +
        "create-next-app: add --yes; create-vite: add --template <name> (e.g. react-ts); create-react-app: add --template typescript; " +
        "npm/yarn init: use -y. Allowed: npx create-next-app, npx create-react-app, npx create-vite, django-admin startproject, " +
        "npm init, yarn init, pnpm init, pnpm create, python -m venv, pip install, git init. 300s timeout.",
    inputSchema: z.object({
        workspace_path: z
            .string()
            .min(1, "workspace_path must not be empty.")
            .describe("Absolute path to the directory where the command should run."),
        command_line: z
            .string()
            .min(1, "command_line must not be empty.")
            .describe("Scaffold command to run (must be non-interactive). E.g. 'npx create-next-app@latest my-app --yes --typescript --tailwind --eslint --app --use-npm'."),
    }),
    async execute({
        workspace_path,
        command_line,
    }: {
        workspace_path: string;
        command_line: string;
    }): Promise<unknown> {
        try {
            return await callWorker("run_scaffold", { workspace_path: workspace_path.trim(), command_line: command_line.trim() }, 300_000);
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error running scaffold." };
        }
    },
});

/**
 * Tool: create_file
 * -----------------
 * Create a new file in the workspace. Parent directories are created automatically.
 */
export const createFile = tool({
    description:
        "Create a new file in the workspace. Use when you need to add a completely new file (e.g. a new module, config, test file). " +
        "Provide file_path (relative to repo root, e.g. 'src/auth.py') and the full content. " +
        "Fails if the file already exists — use edit_file or insert_code for existing files.",
    inputSchema: z.object({
        repo_name: z
            .string()
            .optional()
            .describe("Optional repo name (e.g. 'Eureka')."),
        workspace_path: z
            .string()
            .optional()
            .describe("Optional absolute path to the git repo root."),
        file_path: z
            .string()
            .min(1, "file_path must not be empty.")
            .describe("Path for the new file relative to the repo root (e.g. 'src/auth.py')."),
        content: z
            .string()
            .min(1, "content must not be empty.")
            .describe("Full content of the new file."),
    }),
    async execute({
        repo_name,
        workspace_path,
        file_path,
        content,
    }: {
        repo_name?: string;
        workspace_path?: string;
        file_path: string;
        content: string;
    }): Promise<unknown> {
        const DEFAULT_WORKSPACE = process.env.LOCAL_DAEMON_WORKSPACE_PATH ?? "";
        let workspace = workspace_path?.trim();
        if (!workspace && repo_name) workspace = (await resolveRepoName(repo_name)) ?? undefined;
        if (!workspace) workspace = DEFAULT_WORKSPACE;
        if (!workspace) return { success: false, error: "Could not determine workspace." };
        try {
            return await callWorker("create_file", { workspace_path: workspace, file_path, content });
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error creating file." };
        }
    },
});

/**
 * Tool: batch_create_files
 * ------------------------
 * Create multiple files in one call. Useful when building a new feature that spans several files.
 */
export const batchCreateFiles = tool({
    description:
        "Create multiple new files in a single call. Use when building a feature that requires several new files " +
        "(e.g. a new module with routes, models, and tests). Each entry needs file_path and content.",
    inputSchema: z.object({
        repo_name: z
            .string()
            .optional()
            .describe("Optional repo name (e.g. 'Eureka')."),
        workspace_path: z
            .string()
            .optional()
            .describe("Optional absolute path to the workspace root."),
        files: z
            .array(
                z.object({
                    file_path: z.string().min(1).describe("Relative path (e.g. 'src/auth/routes.py')."),
                    content: z.string().min(1).describe("Full content of the file."),
                }),
            )
            .min(1, "Provide at least one file."),
    }),
    async execute({
        repo_name,
        workspace_path,
        files,
    }: {
        repo_name?: string;
        workspace_path?: string;
        files: Array<{ file_path: string; content: string }>;
    }): Promise<unknown> {
        const DEFAULT_WORKSPACE = process.env.LOCAL_DAEMON_WORKSPACE_PATH ?? "";
        let workspace = workspace_path?.trim();
        if (!workspace && repo_name) workspace = (await resolveRepoName(repo_name)) ?? undefined;
        if (!workspace) workspace = DEFAULT_WORKSPACE;
        if (!workspace) return { success: false, error: "Could not determine workspace." };
        try {
            return await callWorker("create_files", { workspace_path: workspace, files });
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error creating files." };
        }
    },
});

/**
 * Tool: github_create_repo
 * ------------------------
 * Create a new GitHub repository via the REST API.
 */
export const githubCreateRepo = tool({
    description:
        "Create a new GitHub repository. Returns the repo URL and clone URL. " +
        "Use this as the first step when the user asks to create a new project from scratch.",
    inputSchema: z.object({
        name: z
            .string()
            .min(1)
            .describe("Repository name (e.g. 'my-todo-app'). Letters, numbers, hyphens, dots, underscores only."),
        description: z
            .string()
            .optional()
            .describe("Short description for the repo."),
        private: z
            .boolean()
            .optional()
            .describe("Whether the repo should be private. Defaults to false (public)."),
    }),
    async execute({
        name,
        description,
        private: isPrivate,
    }: {
        name: string;
        description?: string;
        private?: boolean;
    }): Promise<unknown> {
        try {
            return await callWorker("github_create_repo", { name, description: description ?? "", private: isPrivate ?? false });
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error creating GitHub repo." };
        }
    },
});

/**
 * Tool: git_clone
 * ---------------
 * Clone a GitHub repository into an allowed workspace directory.
 */
export const gitClone = tool({
    description:
        "Clone a GitHub repository into a local directory. " +
        "Use after github_create_repo to clone the newly created repo. " +
        "Returns local_path which you should use as workspace_path for subsequent tools (batch_create_files, run_tests, etc.). " +
        "Do NOT pass parent_directory — the default is already configured correctly.",
    inputSchema: z.object({
        clone_url: z
            .string()
            .min(1)
            .describe("HTTPS clone URL from github_create_repo (e.g. 'https://github.com/user/repo.git')."),
        parent_directory: z
            .string()
            .optional()
            .describe("DO NOT SET THIS. The default parent directory is pre-configured. Only override if explicitly told to clone somewhere specific."),
        folder_name: z
            .string()
            .optional()
            .describe("Override the cloned folder name. Defaults to the repo name."),
    }),
    async execute({
        clone_url,
        parent_directory,
        folder_name,
    }: {
        clone_url: string;
        parent_directory?: string;
        folder_name?: string;
    }): Promise<unknown> {
        const parentDir = parent_directory?.trim() || process.env.LOCAL_DAEMON_WORKSPACE_PATH?.trim();
        if (!parentDir) return { success: false, error: "Could not determine parent directory. Set LOCAL_DAEMON_WORKSPACE_PATH." };
        try {
            const payload: Record<string, string> = { clone_url, parent_directory: parentDir };
            if (folder_name) payload.folder_name = folder_name;
            return await callWorker("git_clone", payload);
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error cloning repo." };
        }
    },
});

/**
 * Tool: delete_code
 * -----------------
 * Delete an entire function, class, or endpoint block from a file.
 * Provide a search_term that uniquely identifies the block (e.g. '/testing',
 * 'def ping', 'async def health'). The daemon finds the full block
 * (including decorators and docstring) and generates a deletion patch.
 */
export const deleteCode = tool({
    description:
        "Delete an entire function, class, or endpoint block from a Python file. " +
        "Provide a search_term that uniquely identifies the block to delete (e.g. '/testing', 'def ping', 'class UserModel'). " +
        "The daemon finds the full block including decorators and docstring and generates a deletion patch. " +
        "The user will get an 'Approve & Apply' button to confirm.",
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
        search_term: z
            .string()
            .min(1, "search_term must not be empty.")
            .describe(
                "A string that uniquely identifies the block to delete. " +
                "Examples: '/testing' for a GET /testing endpoint, 'def ping' for a ping function, 'class UserModel' for a class.",
            ),
    }),
    async execute({
        repo_name,
        workspace_path,
        file_path,
        search_term,
    }: {
        repo_name?: string;
        workspace_path?: string;
        file_path: string;
        search_term: string;
    }): Promise<StandardResponse> {
        const DEFAULT_WORKSPACE = process.env.LOCAL_DAEMON_WORKSPACE_PATH ?? "";
        let workspace = workspace_path?.trim();
        if (!workspace && repo_name) workspace = (await resolveRepoName(repo_name)) ?? undefined;
        if (!workspace) workspace = DEFAULT_WORKSPACE;
        if (!workspace) return { text: "Could not determine workspace." };

        try {
            const data = (await callWorker("edit_delete_block", { workspace_path: workspace, file_path, search_term })) as {
                success?: boolean; message?: string; patch?: string; deleted_lines?: string;
            };
            if (data.success === false) return { text: `Delete failed: ${data.message ?? "Unknown error."}` };
            const patch = data.patch ?? "";
            if (!patch) return { text: "Worker reported success but returned no patch." };
            console.info("[tools.delete_code] Block to delete:\n", (data.deleted_lines ?? "").slice(0, 500));
            const patchId = await stagePatch(patch, workspace);
            const preview = (data.deleted_lines ?? "").slice(0, 500);
            return {
                text: `I found the block to delete:\n\n\`\`\`\n${preview}\n\`\`\`\n\nPress **Approve & Apply** to remove it.`,
                interactiveButtons: [{ action: `apply_patch:${patchId}`, label: "Approve & Apply" }],
            };
        } catch (e) {
            return { text: `Worker error: ${e instanceof Error ? e.message : String(e)}` };
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
        const DEFAULT_WORKSPACE = process.env.LOCAL_DAEMON_WORKSPACE_PATH ?? "";
        let workspace = workspace_path?.trim();
        if (!workspace && repo_name) workspace = (await resolveRepoName(repo_name)) ?? undefined;
        if (!workspace) workspace = DEFAULT_WORKSPACE;
        if (!workspace) return { text: "Could not determine workspace. Set LOCAL_DAEMON_WORKSPACE_PATH or pass workspace_path." };

        try {
            const data = (await callWorker("edit_insert_code", { workspace_path: workspace, file_path, after_line, new_code })) as {
                success?: boolean; message?: string; patch?: string;
            };
            if (data.success === false) return { text: `Patch computation failed: ${data.message ?? "Unknown error."}` };
            const patch = data.patch ?? "";
            if (!patch) return { text: "Worker reported success but returned no patch." };
            console.info("[tools.insert_code] Patch computed:\n", patch.slice(0, 2000));
            const patchId = await stagePatch(patch, workspace);
            return {
                text: "I've prepared new code to add. Review and press **Approve & Apply** to apply it.\n\n```\n" + patch.slice(0, 1500) + "\n```",
                interactiveButtons: [{ action: `apply_patch:${patchId}`, label: "Approve & Apply" }],
            };
        } catch (e) {
            return { text: `Worker error: ${e instanceof Error ? e.message : String(e)}` };
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
        const DEFAULT_WORKSPACE = process.env.LOCAL_DAEMON_WORKSPACE_PATH ?? "";
        let workspace = workspace_path?.trim();
        if (!workspace && repo_name) workspace = (await resolveRepoName(repo_name)) ?? undefined;
        if (!workspace) workspace = DEFAULT_WORKSPACE;
        if (!workspace) return { text: "Could not determine workspace. Set LOCAL_DAEMON_WORKSPACE_PATH or pass workspace_path." };

        try {
            const data = (await callWorker("edit_replace", { workspace_path: workspace, file_path, search_string, replace_string })) as {
                success?: boolean; message?: string; patch?: string;
            };
            if (data.success === false) return { text: `Patch computation failed: ${data.message ?? "Unknown error."}` };
            const patch = data.patch ?? "";
            if (!patch) return { text: "Worker reported success but returned no patch." };
            console.info("[tools.edit_file] Patch computed:\n", patch.slice(0, 2000));
            const patchId = await stagePatch(patch, workspace);
            return {
                text: "I've prepared a code change. Review and press **Approve & Apply** to apply it.\n\n```\n" + patch.slice(0, 1500) + "\n```",
                interactiveButtons: [{ action: `apply_patch:${patchId}`, label: "Approve & Apply" }],
            };
        } catch (e) {
            return { text: `Worker error: ${e instanceof Error ? e.message : String(e)}` };
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
        let resolvedPath = workspace_path?.trim();
        if (repo_name?.trim() && !resolvedPath) {
            const path = await resolveRepoName(repo_name);
            if (!path) return { success: false, error: `Repository "${repo_name}" not found.` };
            resolvedPath = path;
        }
        if (!resolvedPath) {
            return { success: false, error: "Provide either workspace_path or repo_name." };
        }
        let data: { success?: boolean; message?: string; patch?: string };
        try {
            data = (await callWorker("edit_remove_line", {
                workspace_path: resolvedPath,
                file_path: file_path.replace(/\\/g, "/"),
                line_number,
                ...(line_end != null ? { line_end } : {}),
            })) as typeof data;
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error computing remove-line patch." };
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
        let resolvedPath = workspace_path?.trim();
        if (repo_name?.trim() && !resolvedPath) {
            const path = await resolveRepoName(repo_name);
            if (!path) return { success: false, error: `Repository "${repo_name}" not found.` };
            resolvedPath = path;
        }
        if (!resolvedPath) {
            return { success: false, error: "Provide either workspace_path or repo_name." };
        }
        let data: { success?: boolean; message?: string; patch?: string; files_affected?: number };
        try {
            data = (await callWorker("edit_remove_lines_matching", {
                workspace_path: resolvedPath,
                pattern,
                ...(file_path != null && file_path !== "" ? { file_path: file_path.replace(/\\/g, "/") } : {}),
                ...(path_glob != null && path_glob !== "" ? { path_glob } : {}),
            })) as typeof data;
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error computing remove-lines patch." };
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
        let data: {
            success?: boolean;
            message?: string;
            has_changes?: boolean;
            diff?: string;
            status_short?: string;
        };
        try {
            data = (await callWorker("uncommitted_diff", { workspace_path })) as typeof data;
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error getting diff." };
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
    async execute({ path: pathToDelete }: { path: string }): Promise<unknown> {
        try {
            return await callWorker("delete_path", { path: pathToDelete });
        } catch (e) {
            return { success: false, error: e instanceof Error ? e.message : "Worker error deleting path." };
        }
    },
});

// ---------------------------------------------------------------------------
// Spotify control (via daemon — requires daemon to have SPOTIFY_* configured)
// ---------------------------------------------------------------------------

export const spotifyPlay = tool({
    description: "Start or resume Spotify playback on the user's active device.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("spotify_play"); },
});

export const spotifyPause = tool({
    description: "Pause Spotify playback.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("spotify_pause"); },
});

export const spotifyNext = tool({
    description: "Skip to the next Spotify track.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("spotify_next"); },
});

export const spotifyPrevious = tool({
    description: "Skip to the previous Spotify track.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("spotify_previous"); },
});

export const spotifyVolume = tool({
    description: "Set Spotify app volume (not system volume). Only use when the user explicitly says 'Spotify volume'.",
    inputSchema: z.object({
        volume_percent: z.number().min(0).max(100).describe("Volume level from 0 to 100."),
    }),
    async execute({ volume_percent }: { volume_percent: number }): Promise<unknown> {
        return workerCall("spotify_volume", { volume_percent });
    },
});

export const spotifyStatus = tool({
    description: "Get current Spotify playback status (what's playing, device, etc.).",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("spotify_status"); },
});

export const spotifyClose = tool({
    description:
        "Close or quit the Spotify desktop app on the user's PC. Use when the user asks to close Spotify, exit Spotify, or stop the Spotify app.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("spotify_close"); },
});

// ---------------------------------------------------------------------------
// System volume — via worker on user's PC (pycaw/WASAPI)
// ---------------------------------------------------------------------------

export const systemVolume = tool({
    description:
        "Set or adjust the system master volume on the user's PC. " +
        "Use when the user says 'change volume', 'turn up/down volume', 'set volume to X', 'volume up/down', etc. " +
        "Prefer this over spotify_volume for all generic volume requests. " +
        "Pass absolute_level (0–100) to set an exact level, or step (e.g. '+10' or '-20') for relative adjustment.",
    inputSchema: z.object({
        absolute_level: z.number().min(0).max(100).optional().describe("Set volume to this exact percentage (0–100)."),
        step: z.string().optional().describe("Relative adjustment, e.g. '+10' or '-20'."),
    }),
    async execute({ absolute_level, step }: { absolute_level?: number; step?: string }): Promise<unknown> {
        return workerCall("adjust_volume", { absolute_level, step });
    },
});

// ---------------------------------------------------------------------------
// System brightness — via worker on user's PC (screen-brightness-control)
// ---------------------------------------------------------------------------

export const systemBrightness = tool({
    description:
        "Set or adjust the display brightness on the user's PC. " +
        "Use when the user says 'change brightness', 'turn up/down brightness', 'set brightness to X', 'brightness up/down', 'dim/brighten screen', etc. " +
        "Pass absolute_level (0–100) to set an exact level, or step (e.g. '+10' or '-20') for relative adjustment.",
    inputSchema: z.object({
        absolute_level: z.number().min(0).max(100).optional().describe("Set brightness to this exact percentage (0–100)."),
        step: z.string().optional().describe("Relative adjustment, e.g. '+10' or '-20'."),
    }),
    async execute({ absolute_level, step }: { absolute_level?: number; step?: string }): Promise<unknown> {
        return workerCall("adjust_brightness", { absolute_level, step });
    },
});

// ---------------------------------------------------------------------------
// System control (shutdown, sleep, restart, lock) — via worker on user's PC
// ---------------------------------------------------------------------------

export const systemShutdown = tool({
    description: "Shut down the user's PC. Use when the user asks to shut down, turn off, or power off their computer.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("system_shutdown"); },
});

export const systemRestart = tool({
    description: "Restart the user's PC. Use when the user asks to restart or reboot their computer.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("system_restart"); },
});

export const systemSleep = tool({
    description: "Put the user's PC to sleep (suspend). Use when the user asks to put the computer to sleep.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("system_sleep"); },
});

export const systemLock = tool({
    description: "Lock the user's PC (lock screen, like Win+L). Use when the user asks to lock their computer or lock the screen.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("system_lock"); },
});

export const systemUnlock = tool({
    description:
        "Unlock the user's locked PC by waking the display and auto-typing the stored PIN. " +
        "Use when the user asks to unlock their computer or unlock the screen. " +
        "Requires UNLOCK_PIN to be configured in the daemon's .env file.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> { return workerCall("system_unlock"); },
});

export const openUrl = tool({
    description:
        "Open a URL in the default browser on the user's local machine. " +
        "Use for any request to open a website, link, or URL. " +
        "For 'open antigravity', open https://xkcd.com/353/ — the Python antigravity Easter egg.",
    inputSchema: z.object({
        url: z.string().describe("The fully-qualified URL to open, e.g. 'https://xkcd.com/353/'."),
    }),
    async execute({ url }: { url: string }): Promise<unknown> {
        return workerCall("open_url", { url });
    },
});

export const openApp = tool({
    description:
        "Launch a Windows application by name on the user's PC. " +
        "Use when the user asks to open, start, or launch any app — e.g. 'open Spotify', 'launch VS Code', 'open Chrome', 'start Notepad'. " +
        "Accepts partial or full app names; the worker resolves them via Start Menu shortcuts and Windows App Paths.",
    inputSchema: z.object({
        app_name: z.string().describe("Partial or full application name, e.g. 'spotify', 'VS Code', 'chrome', 'notepad'."),
    }),
    async execute({ app_name }: { app_name: string }): Promise<unknown> {
        return workerCall("open_app", { app_name });
    },
});

export const closeApp = tool({
    description:
        "Close or kill a running Windows application by process name. " +
        "Use when the user asks to close, quit, exit, or kill any app — e.g. 'close Chrome', 'kill Spotify', 'quit Notepad'. " +
        "Pass the executable process name (e.g. 'chrome.exe', 'spotify.exe', 'notepad.exe'). " +
        "Do NOT use open_app for this — only use close_app when the user wants to close/quit/exit an app.",
    inputSchema: z.object({
        process_name: z.string().describe("Exact process executable name, e.g. 'chrome.exe', 'spotify.exe', 'code.exe'."),
    }),
    async execute({ process_name }: { process_name: string }): Promise<unknown> {
        return workerCall("kill_app", { process_name });
    },
});

// ---------------------------------------------------------------------------
// List running processes / apps
// ---------------------------------------------------------------------------

export const listRunningApps = tool({
    description:
        "List the processes currently running on the user's PC. " +
        "Returns windowed (visible) apps tagged [W] and top resource hogs tagged [R], " +
        "with their PID, RAM, and CPU usage. Use when the user asks what's running, " +
        "what apps are open, or wants to find a process before killing it.",
    inputSchema: z.object({}),
    async execute(): Promise<unknown> {
        return workerCall("list_apps", {});
    },
});

// ---------------------------------------------------------------------------
// Claude Code — delegate a task to the local Claude Code CLI
// (sender-aware: created inside makeAiTools so sender_id / bot_token are available)
// ---------------------------------------------------------------------------

/**
 * Tool: find_file
 * ---------------
 * Search for files by name pattern and/or recency across allowed workspaces.
 */
export const findFile = tool({
    description:
        "Search for files on the local PC by name or recency. " +
        "Use name_pattern to match filenames (substring). " +
        "Use modified_within_days to find recent files (e.g. 1 for yesterday's files). " +
        "Use folder_path to narrow the search to a specific directory. " +
        "IMPORTANT: This tool only FINDS the file. If the user wants to send/share the file, " +
        "you MUST immediately follow up by calling send_file_to_telegram with the returned path. " +
        "Never stop after find_file when the user's intent is to receive the file.",
    inputSchema: z.object({
        name_pattern: z.string().optional().describe("Filename substring to match (case-insensitive). Leave empty to match all files."),
        folder_path: z.string().optional().describe("Absolute path of folder to search in. Leave empty to search all allowed workspaces."),
        modified_within_days: z.number().optional().describe("Only return files modified within this many days (e.g. 1 = last 24 hours)."),
        max_results: z.number().optional().describe("Max number of results to return. Defaults to 20."),
    }),
    async execute(args): Promise<unknown> {
        try {
            return await callWorker("find_file", args);
        } catch (e) {
            return { error: e instanceof Error ? e.message : "Worker error finding file." };
        }
    },
});

/**
 * Aggregate export — base tools that don't need caller context.
 */
const baseAiTools = {
    search_local_codebase: searchLocalCodebase,
    create_file: createFile,
    batch_create_files: batchCreateFiles,
    github_create_repo: githubCreateRepo,
    git_clone: gitClone,
    delete_code: deleteCode,
    insert_code: insertCode,
    edit_file: editFile,
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
    run_scaffold: runScaffold,
    delete_path: deletePath,
    find_file: findFile,
    spotify_play: spotifyPlay,
    spotify_pause: spotifyPause,
    spotify_next: spotifyNext,
    spotify_previous: spotifyPrevious,
    spotify_volume: spotifyVolume,
    spotify_status: spotifyStatus,
    spotify_close: spotifyClose,
    system_volume: systemVolume,
    system_brightness: systemBrightness,
    system_shutdown: systemShutdown,
    system_restart: systemRestart,
    system_sleep: systemSleep,
    system_lock: systemLock,
    system_unlock: systemUnlock,
    open_url: openUrl,
    open_app: openApp,
    close_app: closeApp,
    list_running_apps: listRunningApps,
};

/**
 * Factory: returns all tools, including sender-aware tools like send_file_to_telegram.
 * Pass the caller's senderId (Telegram chat ID) to enable file delivery.
 */
export function makeAiTools(senderId: string) {
    const sendFileToTelegram = tool({
        description:
            "Send a file from the local PC to the user's Telegram chat. " +
            "Use find_file first to get the exact file path. " +
            "Supports images (jpg, png, gif, webp) and any other file type.",
        inputSchema: z.object({
            file_path: z.string().describe("Absolute path to the file to send."),
            caption: z.string().optional().describe("Optional caption displayed under the file."),
        }),
        async execute({ file_path, caption }: { file_path: string; caption?: string }): Promise<unknown> {
            const botToken = process.env.TELEGRAM_BOT_TOKEN;
            if (!botToken) return { error: "TELEGRAM_BOT_TOKEN not set in environment." };
            try {
                return await callWorker("send_file_to_telegram", {
                    file_path,
                    chat_id: senderId,
                    bot_token: botToken,
                    caption: caption ?? "",
                }, 90_000); // 90s: covers large files + slow upload to Telegram
            } catch (e) {
                const msg = e instanceof Error ? e.message : "Worker error sending file.";
                return { success: false, error: `Failed to send file: ${msg}` };
            }
        },
    });

    const captureWebcam = tool({
        description:
            "Snap a photo from the PC's webcam and send it instantly to this Telegram chat. " +
            "Use when the user asks to take a photo, capture a webcam image, or see what's in front of the camera.",
        inputSchema: z.object({}),
        async execute(): Promise<unknown> {
            const botToken = process.env.TELEGRAM_BOT_TOKEN;
            if (!botToken) return { error: "TELEGRAM_BOT_TOKEN not set in environment." };
            try {
                return await callWorker("capture_webcam", {
                    chat_id: senderId,
                    bot_token: botToken,
                }, 90_000); // 90s: camera init + 5-frame warmup + Telegram upload
            } catch (e) {
                return { error: e instanceof Error ? e.message : "Worker error capturing webcam." };
            }
        },
    });

    const rescueFile = tool({
        description:
            "Find a file by name on the home PC and upload it directly to this Telegram chat. " +
            "Searches under %USERPROFILE% (Desktop, Documents, Downloads, and coding folders). " +
            "Use when the user asks to send, fetch, or retrieve a specific file from their PC (e.g. 'send me Resume.pdf', 'get AWS_keys.env').",
        inputSchema: z.object({
            file_name: z
                .string()
                .min(1, "file_name must not be empty.")
                .describe("Exact filename to search for, e.g. 'Resume.pdf' or 'AWS_keys.env'."),
        }),
        async execute({ file_name }: { file_name: string }): Promise<unknown> {
            const botToken = process.env.TELEGRAM_BOT_TOKEN;
            if (!botToken) return { error: "TELEGRAM_BOT_TOKEN not set in environment." };
            try {
                return await callWorker("rescue_file", {
                    file_name,
                    chat_id: senderId,
                    bot_token: botToken,
                });
            } catch (e) {
                return { error: e instanceof Error ? e.message : "Worker error rescuing file." };
            }
        },
    });

    const remoteDownload = tool({
        description:
            "Download a URL on the home PC in the background and notify when complete. " +
            "Supports YouTube / Twitter / Vimeo video links (uses yt-dlp for best quality), " +
            "direct file URLs ending in .zip / .pdf / .csv / etc. (streamed to disk), " +
            "and magnet: links (passed to aria2c). " +
            "Sends a confirmation immediately and a 'Download Complete' message when finished. " +
            "Use when the user says 'download this for me', 'save this video', 'grab this file', etc.",
        inputSchema: z.object({
            url: z
                .string()
                .min(1, "url must not be empty.")
                .describe("The URL to download — a YouTube link, direct file URL, or magnet: URI."),
        }),
        async execute({ url }: { url: string }): Promise<unknown> {
            const botToken = process.env.TELEGRAM_BOT_TOKEN;
            if (!botToken) return { error: "TELEGRAM_BOT_TOKEN not set in environment." };
            try {
                // Short timeout: the worker starts a background thread and returns immediately.
                return await callWorker("remote_download", {
                    url,
                    chat_id: senderId,
                    bot_token: botToken,
                }, 10_000);
            } catch (e) {
                return { error: e instanceof Error ? e.message : "Worker error starting download." };
            }
        },
    });

    // Combined find + send — use this when the user wants to receive a file.
    // Avoids relying on the LLM to chain find_file → send_file_to_telegram.
    const sendFile = tool({
        description:
            "Find a file on the local PC by name and send it to this Telegram chat in one step. " +
            "Use this whenever the user asks to send, share, or receive a file. " +
            "Pass the filename (or partial name) and optionally a folder_path to narrow the search.",
        inputSchema: z.object({
            name_pattern: z.string().describe("Filename substring to search for (case-insensitive)."),
            folder_path: z.string().optional().describe("Absolute path to search in. Leave empty to search all workspaces."),
            caption: z.string().optional().describe("Optional caption to include with the file."),
        }),
        async execute({ name_pattern, folder_path, caption }: { name_pattern: string; folder_path?: string; caption?: string }): Promise<unknown> {
            const botToken = process.env.TELEGRAM_BOT_TOKEN;
            if (!botToken) return { success: false, error: "TELEGRAM_BOT_TOKEN not set." };
            // Step 1: find the file
            let found: { success?: boolean; files?: Array<{ path: string; name: string }> };
            try {
                found = await callWorker("find_file", { name_pattern, folder_path, max_results: 5 }) as typeof found;
            } catch (e) {
                return { success: false, error: `find_file failed: ${e instanceof Error ? e.message : e}` };
            }
            if (!found.files?.length) return { success: false, error: `No file matching "${name_pattern}" found.` };
            const file = found.files[0];
            // Step 2: send it
            try {
                return await callWorker("send_file_to_telegram", {
                    file_path: file.path,
                    chat_id: senderId,
                    bot_token: botToken,
                    caption: caption ?? "",
                }, 90_000);
            } catch (e) {
                return { success: false, error: `send failed: ${e instanceof Error ? e.message : e}` };
            }
        },
    });

    const runClaudeCode = tool({
        description:
            "Delegate a coding task to the Claude Code CLI running on the user's local machine. " +
            "Use this for large builds, multi-file refactors, or any task where Claude Code's " +
            "native file-editing tools (Write, Edit, Bash, Glob, Grep) are better suited than " +
            "manual patch application. Output is streamed back to Telegram in real time. " +
            "Use when the user says things like 'use Claude Code to build', 'let Claude Code handle it', " +
            "'build this with Claude Code', or for any complex project that benefits from an autonomous coding agent.",
        inputSchema: z.object({
            prompt: z.string().describe("Full task description / instructions for Claude Code."),
            working_directory: z.string().optional().describe("Absolute path to the project directory (must be in ALLOWED_WORKSPACES). Defaults to the first configured workspace."),
            timeout: z.number().optional().describe("Maximum seconds to wait before giving up (default 1200 = 20 min)."),
            allowed_tools: z.string().optional().describe("Comma-separated list of Claude Code tools to allow. Defaults to 'Bash,Write,Edit,Read,Glob,Grep,LS,MultiEdit'."),
        }),
        async execute({ prompt, working_directory, timeout, allowed_tools }: {
            prompt: string;
            working_directory?: string;
            timeout?: number;
            allowed_tools?: string;
        }): Promise<unknown> {
            const botToken = process.env.TELEGRAM_BOT_TOKEN;
            if (!botToken) return { error: "TELEGRAM_BOT_TOKEN not set in environment." };
            await pushHostCommand(
                "run_claude_code",
                { prompt, working_directory, timeout, allowed_tools },
                { senderId, botToken, async: true },
            );
            return { success: true, message: "Claude Code task started — output will stream to Telegram." };
        },
    });

    const takeScreenshot = tool({
        description: "Capture the user's entire screen and send the photo to Telegram.",
        inputSchema: z.object({
            caption: z.string().optional().describe("Optional caption shown under the screenshot."),
        }),
        async execute({ caption }: { caption?: string }): Promise<unknown> {
            const botToken = process.env.TELEGRAM_BOT_TOKEN;
            if (!botToken) return { error: "TELEGRAM_BOT_TOKEN not set." };
            try {
                return await callWorker("take_screenshot", {
                    chat_id: senderId,
                    bot_token: botToken,
                    caption: caption ?? "",
                }, 30_000);
            } catch (e) {
                return { success: false, error: e instanceof Error ? e.message : "Screenshot failed." };
            }
        },
    });

    const analyzeScreen = tool({
        description:
            "Capture the screen, send it to Telegram, then analyze what is visible using vision AI. " +
            "Use this when the user asks what is on their screen or wants you to read/describe screen content.",
        inputSchema: z.object({
            question: z.string().optional().describe("Specific question about what to look for on screen."),
        }),
        async execute({ question }: { question?: string }): Promise<unknown> {
            const botToken = process.env.TELEGRAM_BOT_TOKEN;
            if (!botToken) return { error: "TELEGRAM_BOT_TOKEN not set." };
            try {
                const raw = await callWorker("take_screenshot", {
                    chat_id: senderId,
                    bot_token: botToken,
                    return_base64: true,
                }, 30_000) as { base64_image?: string; success?: boolean; error?: string };
                if (!raw.base64_image) return { success: false, error: raw.error ?? "No image returned." };
                const { text } = await generateText({
                    model: openai(process.env.LLM_MODEL_NAME ?? "gpt-4o"),
                    messages: [{
                        role: "user",
                        content: [
                            { type: "image", image: Buffer.from(raw.base64_image, "base64") },
                            { type: "text", text: question ?? "Describe everything visible on this screen in detail." },
                        ],
                    }],
                });
                return { analysis: text };
            } catch (e) {
                return { success: false, error: e instanceof Error ? e.message : "Screen analysis failed." };
            }
        },
    });

    const browserAction = tool({
        description:
            "Control a Chromium browser: open URLs, navigate, click elements, fill forms, extract page text, scroll, or take a page screenshot. " +
            "Always call with action='open' before using other actions on a fresh session.",
        inputSchema: z.object({
            action: z.enum(["open", "navigate", "screenshot", "click", "type", "extract", "scroll", "fill_form", "close"])
                .describe("Browser action to perform."),
            url: z.string().optional().describe("URL for open/navigate actions."),
            selector: z.string().optional().describe("CSS selector for click/type/extract actions."),
            text: z.string().optional().describe("Visible text to click (alternative to selector)."),
            value: z.string().optional().describe("Text to fill into a field (type action)."),
            fields: z.record(z.string(), z.string()).optional().describe("Map of CSS selector to value for fill_form action."),
            dx: z.number().optional().describe("Horizontal scroll amount in pixels."),
            dy: z.number().optional().describe("Vertical scroll amount in pixels (positive = down)."),
            headless: z.boolean().optional().describe("Run browser headless (invisible). Defaults to false."),
            caption: z.string().optional().describe("Caption for screenshot action."),
        }),
        async execute({ action, url, selector, text, value, fields, dx, dy, headless, caption }): Promise<unknown> {
            const botToken = process.env.TELEGRAM_BOT_TOKEN;
            try {
                return await callWorker("browser", {
                    action, url, selector, text, value, fields, dx, dy, headless, caption,
                    chat_id: senderId,
                    bot_token: botToken ?? "",
                }, 45_000);
            } catch (e) {
                return { success: false, error: e instanceof Error ? e.message : "Browser action failed." };
            }
        },
    });

    const guiAction = tool({
        description:
            "Control the mouse and keyboard: click, double-click, right-click, type text, press hotkeys, scroll, move mouse, or drag. " +
            "Use take_screenshot first to find coordinates if needed.",
        inputSchema: z.object({
            action: z.enum(["click", "double_click", "right_click", "type", "hotkey", "press", "scroll", "move", "drag", "get_position"])
                .describe("GUI action to perform."),
            x: z.number().optional().describe("X coordinate (pixels from left)."),
            y: z.number().optional().describe("Y coordinate (pixels from top)."),
            x2: z.number().optional().describe("Destination X for drag action."),
            y2: z.number().optional().describe("Destination Y for drag action."),
            text: z.string().optional().describe("Text to type."),
            keys: z.string().optional().describe("Hotkey combination, e.g. 'ctrl+c' or 'alt+tab'."),
            key: z.string().optional().describe("Single key to press, e.g. 'enter', 'escape', 'f5'."),
            clicks: z.number().optional().describe("Scroll amount (positive = up, negative = down)."),
            button: z.string().optional().describe("Mouse button: 'left', 'right', or 'middle'."),
            duration: z.number().optional().describe("Duration in seconds for move/drag."),
            interval: z.number().optional().describe("Delay between keystrokes in seconds for type action."),
        }),
        async execute({ action, x, y, x2, y2, text, keys, key, clicks, button, duration, interval }): Promise<unknown> {
            try {
                return await callWorker("gui_action", { action, x, y, x2, y2, text, keys, key, clicks, button, duration, interval }, 15_000);
            } catch (e) {
                return { success: false, error: e instanceof Error ? e.message : "GUI action failed." };
            }
        },
    });

    const manageWindows = tool({
        description: "List all open windows or bring a specific window to the foreground by its title.",
        inputSchema: z.object({
            action: z.enum(["list", "focus"]).describe("'list' returns all window titles; 'focus' brings a window to front."),
            title_contains: z.string().optional().describe("Case-insensitive substring of the window title to focus."),
        }),
        async execute({ action, title_contains }: { action: string; title_contains?: string }): Promise<unknown> {
            try {
                if (action === "list") return await callWorker("get_windows", {}, 10_000);
                return await callWorker("focus_window", { title_contains }, 10_000);
            } catch (e) {
                return { success: false, error: e instanceof Error ? e.message : "Window management failed." };
            }
        },
    });

    // Remove find_file from the sender-aware tool set so the LLM uses
    // send_file (which chains find + send) instead of stopping after find_file.
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { find_file: _findFileOmitted, ...senderBaseTools } = baseAiTools;

    return {
        ...senderBaseTools,
        send_file: sendFile,
        send_file_to_telegram: sendFileToTelegram,
        capture_webcam: captureWebcam,
        rescue_file: rescueFile,
        remote_download: remoteDownload,
        run_claude_code: runClaudeCode,
        take_screenshot: takeScreenshot,
        analyze_screen: analyzeScreen,
        browser_action: browserAction,
        gui_action: guiAction,
        manage_windows: manageWindows,
    };
}

/** Convenience alias for call sites that don't need sender-aware tools. */
export const aiTools = baseAiTools;

export type AiTools = ReturnType<typeof makeAiTools>;

