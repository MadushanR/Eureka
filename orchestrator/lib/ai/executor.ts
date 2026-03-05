/**
 * lib/ai/executor.ts — DevMode Execution Engine
 * ===================================================
 * Iterates through a ProjectPlan's BuildSteps, delegates each to the local
 * Python daemon via POST /execute-build-step, and finalises with a
 * POST /git-push.  Sends Telegram progress updates after each phase.
 *
 * Designed to run inside `waitUntil()` so the webhook returns 200 immediately
 * while this loop runs in the background.
 *
 * Public API:
 *   executeProjectRoadmap(plan, senderId, sendUpdate)
 */

import type { ProjectPlan, BuildStep } from "./planner";
import { createJob, updateJob, getJob, clearActiveJob, pushHostCommand, pollResult, type DevJob, type DevJobPhase } from "@/lib/redis";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ProgressFn = (message: string) => Promise<void>;

interface StepResult {
    success: boolean;
    stdout: string;
    stderr: string;
    exit_code: number;
    error: string;
}

interface GitPushResult {
    success: boolean;
    message: string;
    stdout: string;
    stderr: string;
}

interface GitHubCreateRepoResult {
    success: boolean;
    message: string;
    html_url: string;
    clone_url: string;
    name: string;
}

// ---------------------------------------------------------------------------
// Worker queue helpers
// ---------------------------------------------------------------------------

const workspacePath = (): string =>
    process.env.LOCAL_DAEMON_WORKSPACE_PATH?.replace(/\/+$/, "") ?? "C:/Users/madus/Desktop";

/** Timeout for execute_build_step. Worker allows 600 s per command; accommodate multiple long commands. */
const BUILD_STEP_TIMEOUT_MS = (() => {
    const raw = process.env.BUILD_STEP_TIMEOUT_MS;
    if (raw === undefined || raw === "") return 1_800_000; // 30 min default
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : 1_800_000;
})();

/**
 * Push an action onto the Redis host-command queue and wait for the result.
 * Replaces the old HTTP `callDaemon()` — no tunnel required.
 */
async function callWorker<T>(
    action: string,
    payload: Record<string, unknown>,
    timeoutMs: number = 30_000,
): Promise<T> {
    const taskId = await pushHostCommand(action, payload);
    return (await pollResult(taskId, timeoutMs)) as T;
}

// ---------------------------------------------------------------------------
// Execute a single build step (with one retry on failure)
// ---------------------------------------------------------------------------

async function executeBuildStep(
    step: BuildStep,
    projectRoot: string,
): Promise<StepResult> {
    const targetDir =
        step.targetDirectory === "." || !step.targetDirectory
            ? projectRoot
            : `${projectRoot}/${step.targetDirectory.replace(/^\/+/, "")}`;

    return callWorker<StepResult>(
        "execute_build_step",
        {
            workspace_path: targetDir,
            step_name: step.stepName,
            description: step.description,
            terminal_commands: step.terminalCommands,
        },
        BUILD_STEP_TIMEOUT_MS,
    );
}

async function executeBuildStepWithRetry(
    step: BuildStep,
    projectRoot: string,
    sendUpdate: ProgressFn,
    stepLabel: string,
): Promise<StepResult> {
    const result = await executeBuildStep(step, projectRoot);
    if (!result.success) {
        console.warn(`[devmode] ${stepLabel} failed on attempt 1, retrying in 10s...`);
        await sendUpdate(`Retrying ${stepLabel}...`);
        await new Promise((r) => setTimeout(r, 10_000));
        return executeBuildStep(step, projectRoot);
    }
    return result;
}

// ---------------------------------------------------------------------------
// GitHub repo creation + Git push finalisation
// ---------------------------------------------------------------------------

async function createGitHubRepo(
    name: string,
    description: string,
    privateRepo: boolean = false,
): Promise<GitHubCreateRepoResult> {
    return callWorker<GitHubCreateRepoResult>("github_create_repo", {
        name,
        description: description.slice(0, 350),
        private: privateRepo,
    });
}

async function gitPush(
    projectRoot: string,
    commitMessage: string,
    cloneUrl?: string,
): Promise<GitPushResult> {
    return callWorker<GitPushResult>("git_push", {
        workspace_path: projectRoot,
        commit_message: commitMessage,
        ...(cloneUrl ? { clone_url: cloneUrl } : {}),
    }, 300_000); // 5-minute timeout for git push
}

// ---------------------------------------------------------------------------
// Main execution loop
// ---------------------------------------------------------------------------

export async function executeProjectRoadmap(
    plan: ProjectPlan,
    senderId: string,
    sendUpdate: ProgressFn,
): Promise<void> {
    const projectRoot = `${workspacePath()}/${plan.projectName}`;
    const totalSteps = plan.steps.length;
    const job = await createJob(senderId, `DevMode: ${plan.projectName}`);

    const jobPhases: DevJobPhase[] = plan.steps.map((s) => ({
        name: s.stepName,
        description: s.description,
        files: [],
        status: "pending",
    }));

    await updateJob(job.id, {
        status: "executing",
        total_steps: totalSteps,
        phases: jobPhases,
        total_phases: totalSteps,
    });

    const completedSteps: string[] = [];

    for (let i = 0; i < totalSteps; i++) {
        // Check cancellation
        const current = await getJob(job.id);
        if (current?.status === "cancelled") {
            await sendUpdate(`DevMode build cancelled at step ${i + 1}/${totalSteps}.`);
            return;
        }

        const step = plan.steps[i];
        const phaseLabel = `[${i + 1}/${totalSteps}] ${step.stepName}`;

        jobPhases[i] = { ...jobPhases[i], status: "executing" };
        await updateJob(job.id, {
            current_step: i + 1,
            current_phase: i + 1,
            current_action: step.stepName,
            phases: jobPhases,
        });

        console.info(`[devmode] ${phaseLabel} — executing`);

        try {
            const result = await executeBuildStepWithRetry(step, projectRoot, sendUpdate, phaseLabel);

            if (!result.success) {
                let errDetail = result.error || result.stderr || "Unknown error";
                const lowerErr = errDetail.toLowerCase();
                if (lowerErr.includes("aider") && (lowerErr.includes("not found") || lowerErr.includes("command not found"))) {
                    errDetail += "\n\nTip: Install aider where the daemon runs (e.g. pip install aider-chat). Ensure the same PATH is used when the daemon runs build steps.";
                }

                if (step.optional) {
                    jobPhases[i] = { ...jobPhases[i], status: "failed" };
                    await updateJob(job.id, { phases: jobPhases });
                    await sendUpdate(`Skipped optional step ${phaseLabel} (non-fatal, continuing build).`);
                    console.warn(`[devmode] ${phaseLabel} SKIPPED (optional): ${errDetail.slice(0, 200)}`);
                    continue;
                }

                jobPhases[i] = { ...jobPhases[i], status: "failed" };
                await updateJob(job.id, {
                    status: "failed",
                    phases: jobPhases,
                    errors: [`Step ${i + 1} failed: ${errDetail}`],
                    finished_at: Date.now(),
                });

                await sendUpdate(
                    `HALTED at ${phaseLabel}\n\nError:\n${errDetail.slice(0, 600)}\n\n` +
                    `${completedSteps.length} of ${totalSteps} steps completed before failure.`,
                );
                console.error(`[devmode] ${phaseLabel} FAILED: ${errDetail.slice(0, 200)}`);

                await clearActiveJob(senderId);
                return;
            }

            jobPhases[i] = { ...jobPhases[i], status: "complete" };
            completedSteps.push(step.stepName);

            await updateJob(job.id, {
                phases: jobPhases,
                steps_completed: completedSteps,
            });

            await sendUpdate(`Phase ${i + 1}/${totalSteps} complete: ${step.stepName}`);
            console.info(`[devmode] ${phaseLabel} — done (exit ${result.exit_code})`);
        } catch (err) {
            const errMsg = err instanceof Error ? err.message : String(err);

            if (step.optional) {
                jobPhases[i] = { ...jobPhases[i], status: "failed" };
                await updateJob(job.id, { phases: jobPhases });
                await sendUpdate(`Skipped optional step ${phaseLabel} (exception, non-fatal, continuing build).`);
                console.warn(`[devmode] ${phaseLabel} SKIPPED (optional exception):`, err);
                continue;
            }

            jobPhases[i] = { ...jobPhases[i], status: "failed" };
            await updateJob(job.id, {
                status: "failed",
                phases: jobPhases,
                errors: [`Step ${i + 1} exception: ${errMsg}`],
                finished_at: Date.now(),
            });

            await sendUpdate(
                `HALTED at ${phaseLabel}\n\nException:\n${errMsg.slice(0, 500)}\n\n` +
                `${completedSteps.length} of ${totalSteps} steps completed before failure.`,
            );
            console.error(`[devmode] ${phaseLabel} EXCEPTION:`, err);

            await clearActiveJob(senderId);
            return;
        }
    }

    // ---------------------------------------------------------------------------
    // All steps passed — create GitHub repo (if daemon supports it) then push
    // ---------------------------------------------------------------------------
    await sendUpdate("All build steps complete. Creating GitHub repo...");
    console.info(`[build] All ${totalSteps} steps done. Creating repo for ${plan.projectName}...`);

    let cloneUrl: string | undefined;
    let repoHtmlUrl: string | undefined;

    try {
        const repoResult = await createGitHubRepo(plan.projectName, plan.techStack, false);
        if (repoResult.success && repoResult.clone_url) {
            cloneUrl = repoResult.clone_url;
            repoHtmlUrl = repoResult.html_url;
            await sendUpdate(`Repo created. Pushing to GitHub...`);
        } else {
            await sendUpdate(
                `GitHub repo not created (${repoResult.message}). Committing locally and attempting push...`,
            );
        }
    } catch (err) {
        console.warn(`[build] createGitHubRepo failed:`, err);
        await sendUpdate("Could not create GitHub repo (is GITHUB_TOKEN set in the daemon?). Pushing if remote exists...");
    }

    try {
        const pushResult = await gitPush(
            projectRoot,
            `feat: ${plan.projectName} — built by Eureka`,
            cloneUrl,
        );

        if (pushResult.success) {
            await updateJob(job.id, {
                status: "complete",
                current_action: "Pushed to Git",
                finished_at: Date.now(),
                phases: jobPhases,
            });

            const repoLine = repoHtmlUrl ? `\n\nRepo: ${repoHtmlUrl}` : "";
            await sendUpdate(
                `Build complete!\n\n` +
                `Project: ${plan.projectName}\n` +
                `Stack: ${plan.techStack}\n` +
                `Steps: ${totalSteps}\n\n` +
                completedSteps.map((s, i) => `  ${i + 1}. ${s}`).join("\n") +
                `\n\n${pushResult.message}` +
                repoLine,
            );
        } else {
            await updateJob(job.id, {
                status: "failed",
                errors: [`Git push failed: ${pushResult.message}`],
                finished_at: Date.now(),
            });

            const hint = !cloneUrl
                ? "\n\nTip: Set GITHUB_TOKEN in the daemon .env to auto-create a repo and push."
                : "";
            await sendUpdate(
                `Build complete but git push failed:\n${pushResult.message}\n\n` +
                `The project is at: ${projectRoot}\n` +
                `You can push manually.` +
                hint,
            );
        }
    } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        await updateJob(job.id, {
            status: "failed",
            errors: [`Git push exception: ${errMsg}`],
            finished_at: Date.now(),
        });

        await sendUpdate(
            `Build complete but git push threw an error:\n${errMsg}\n\n` +
            `The project is at: ${projectRoot}`,
        );
    }

    await clearActiveJob(senderId);
}
