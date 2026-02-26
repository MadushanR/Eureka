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
        "Find git repositories under the allowed workspaces on the local machine. " +
        "Use this to resolve project names like 'Eureka' to absolute repo paths before running git operations.",
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
 * Use when the user asks "any uncommitted changes?", "uncommitted changes in Eureka",
 * "show me changes in all repos", etc. For a named repo, call list_git_repos first to
 * resolve the path, then call this with that workspace_path.
 */
export const getUncommittedChanges = tool({
    description:
        "Get uncommitted changes (diff and status) for a git repo or for all repos. " +
        "Use when the user asks about uncommitted changes, e.g. 'any uncommitted changes?', 'uncommitted changes in Eureka', 'show changes in all repos'. " +
        "If workspace_path is provided, returns changes for that repo only. If omitted, returns a summary for every git repo under the allowed workspaces.",
    inputSchema: z.object({
        workspace_path: z
            .string()
            .optional()
            .describe(
                "Absolute path to one repo. Omit to get uncommitted changes for all repos.",
            ),
    }),
    async execute({
        workspace_path,
    }: {
        workspace_path?: string;
    }): Promise<{ text: string } | { error: string }> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { error: "LOCAL_DAEMON_URL is not configured." };
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

        if (workspace_path && workspace_path.trim()) {
            const data = await fetchDiff(workspace_path.trim());
            if (!data.success) {
                return { error: data.message ?? "Could not get repo status." };
            }
            if (!data.has_changes) {
                return {
                    text: `**${workspace_path}**\n\nNo uncommitted changes. Working tree clean.`,
                };
            }
            const diff = (data.diff ?? "").trim();
            const statusShort = (data.status_short ?? "").trim();
            const truncated = diff.length > maxDiffLen;
            const diffText = truncated ? diff.slice(0, maxDiffLen) + "\n\n… (truncated)" : diff;
            return {
                text:
                    `**${workspace_path}**\n\n` +
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
 * Tool: request_patch_approval
 * ----------------------------
 * When the model has prepared a *code* patch (unified diff) that will be applied
 * via git apply, it calls this tool to surface an approval UI. Use ONLY for
 * editing file contents (add/change/delete lines in code files). Do NOT use
 * for: deleting folders, deleting whole files, moving/renaming files, or any
 * other filesystem operation that is not a git patch.
 */
export const requestPatchApproval = tool({
    description:
        "Ask the user to approve a generated *code* patch (unified diff) before it is applied with git. " +
        "Use ONLY when you have produced a valid unified diff that edits file contents (e.g. add/change/remove lines in source files). " +
        "Do NOT use for: deleting folders, deleting entire files, moving files, or any task that is not applying a code patch. " +
        "For those tasks, reply in text that you cannot do them and suggest the user do it manually (e.g. delete the folder in File Explorer).",
    inputSchema: z.object({
        patch_string: z
            .string()
            .min(1, "Patch string must not be empty.")
            .describe(
                "Unified diff or patch representation of the proposed code changes (git apply format). " +
                    "Only use when the change is a valid code patch, not for folder/file deletion or other filesystem operations.",
            ),
        workspace_path: z
            .string()
            .optional()
            .describe(
                "Optional absolute path to the git repository root where the patch should be applied (e.g. C:\\Users\\you\\project). " +
                    "If omitted, the default from the environment is used.",
            ),
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
 * Then stages the patch for user approval (Approve & Apply). Use when the user says
 * e.g. "Remove line 10 from Eureka/src/main.py".
 */
export const removeLine = tool({
    description:
        "Remove a specific line (or range of lines) from a file in a repo. " +
        "Use when the user asks to remove line N, or lines N–M, from a file (e.g. 'Remove line 10 from Eureka/src/main.py'). " +
        "Call list_git_repos first to get workspace_path, then call this with the repo path and file path relative to the repo.",
    inputSchema: z.object({
        workspace_path: z
            .string()
            .min(1, "workspace_path must not be empty.")
            .describe("Absolute path to the git repository root."),
        file_path: z
            .string()
            .min(1, "file_path must not be empty.")
            .describe("Path to the file relative to the repo root (e.g. src/main.py)."),
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
        file_path,
        line_number,
        line_end,
    }: {
        workspace_path: string;
        file_path: string;
        line_number: number;
        line_end?: number;
    }): Promise<StandardResponse | { success: boolean; error: string }> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { success: false, error: "LOCAL_DAEMON_URL is not configured." };
        }
        const url = `${base}/edit/compute-remove-line-patch`;
        let data: { success?: boolean; message?: string; patch?: string };
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    workspace_path,
                    file_path,
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
        const patchId = await stagePatch(patch, workspace_path);
        return {
            text: `I have prepared a patch to remove line ${line_number}${line_end != null ? `–${line_end}` : ""} from \`${file_path}\`. Do you want to apply it?`,
            interactiveButtons: [{ action: `apply_patch:${patchId}`, label: "Approve & Apply" }],
        };
    },
});

/**
 * Tool: remove_lines_matching
 * ---------------------------
 * Ask the daemon to compute a patch that removes all lines containing a pattern
 * (in one file or all files in the repo). Then stages the patch for approval.
 * Use for e.g. "Remove all lines that say '#testing'" in one file or all files.
 */
export const removeLinesMatching = tool({
    description:
        "Remove all lines that contain a given pattern (literal text) from a file or from all files in a repo. " +
        "Use when the user asks to remove lines containing some text (e.g. 'Remove #testing from Eureka', 'Remove all lines that say #testing from all files in Eureka'). " +
        "For a single file, pass file_path. For all files, omit file_path. Call list_git_repos first to get workspace_path.",
    inputSchema: z.object({
        workspace_path: z
            .string()
            .min(1, "workspace_path must not be empty.")
            .describe("Absolute path to the git repository root."),
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
        pattern,
        file_path,
        path_glob,
    }: {
        workspace_path: string;
        pattern: string;
        file_path?: string;
        path_glob?: string;
    }): Promise<StandardResponse | { success: boolean; error: string }> {
        const base = LOCAL_DAEMON_BASE();
        if (!base) {
            return { success: false, error: "LOCAL_DAEMON_URL is not configured." };
        }
        const url = `${base}/edit/compute-remove-lines-matching-patch`;
        let data: { success?: boolean; message?: string; patch?: string; files_affected?: number };
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    workspace_path,
                    pattern,
                    ...(file_path != null && file_path !== "" ? { file_path } : {}),
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
        const patchId = await stagePatch(patch, workspace_path);
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
    request_patch_approval: requestPatchApproval,
    remove_line: removeLine,
    remove_lines_matching: removeLinesMatching,
    prepare_push_approval: preparePushApproval,
    prepare_push_only_approval: preparePushOnlyApproval,
    list_git_repos: listGitRepos,
    get_uncommitted_changes: getUncommittedChanges,
    list_workspace_folders: listWorkspaceFolders,
    list_folder_contents: listFolderContents,
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

