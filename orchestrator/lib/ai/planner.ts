/**
 * lib/ai/planner.ts — DevMode Project Planner
 * =================================================
 * Uses the Vercel AI SDK `generateObject` with a strict Zod schema to break
 * a user's single-prompt project idea into a sequenced array of BuildSteps.
 *
 * Each BuildStep is an isolated build phase that the executor can hand off
 * to the local Python daemon for headless execution (scaffold commands,
 * aider code-gen, tests, etc.).
 *
 * Public API:
 *   generateProjectRoadmap(prompt)  — returns a validated ProjectPlan
 */

import { generateObject } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import { getUserProfile, formatProfileForPrompt } from "./memory";
import { withTelemetry } from "./telemetry";

// ---------------------------------------------------------------------------
// Zod schemas
// ---------------------------------------------------------------------------

export const BuildStepSchema = z.object({
    stepName: z.string().describe("Short human-readable phase name (e.g. 'Scaffold Next.js App')."),
    description: z.string().describe(
        "Detailed instructions for this phase. If the step requires AI code generation, " +
        "write the full aider prompt here. Otherwise describe what terminal commands accomplish.",
    ),
    targetDirectory: z.string().describe(
        "Relative directory inside the project root where this step operates " +
        "(e.g. '.' for root, 'backend/', 'frontend/').",
    ),
    terminalCommands: z.array(z.string()).describe(
        "Ordered list of shell commands to run for this step. " +
        "Use 'aider --message \"<INSTRUCTIONS>\" --yes --auto-commits' for AI code-gen steps. " +
        "Use standard CLI commands (npm init, pip install, npx create-next-app, etc.) for scaffolding. " +
        "Leave empty ONLY if the step is purely an aider code-gen step with no other commands.",
    ),
    dependsOnStep: z.number().nullable().describe(
        "0-based index of a prior step this one depends on. Use null if this step has no dependency.",
    ),
});

export const ProjectPlanSchema = z.object({
    projectName: z.string().describe(
        "A short, kebab-case name for the project (e.g. 'food-ordering-app'). " +
        "Used as the repo name and root directory.",
    ),
    techStack: z.string().describe(
        "One-line summary of the chosen tech stack (e.g. 'Next.js 14 + TailwindCSS + Prisma + PostgreSQL').",
    ),
    steps: z.array(BuildStepSchema).min(2).describe(
        "Ordered list of build phases. Must start with scaffolding/setup and end with " +
        "a testing/verification step. Typical count: 4-8 steps.",
    ),
});

export type BuildStep = z.infer<typeof BuildStepSchema>;
export type ProjectPlan = z.infer<typeof ProjectPlanSchema>;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

const PLANNER_MODEL_NAME = process.env.LLM_MODEL_NAME_DEV || process.env.LLM_MODEL_NAME || "gpt-4.1-mini";

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const PLANNER_SYSTEM_PROMPT =
    "You are a Senior Solutions Architect building a complete project from a single user prompt.\n\n" +
    "Your job:\n" +
    "1. Choose an appropriate, modern tech stack for the request.\n" +
    "2. Break the project into sequential, isolated BUILD STEPS.\n" +
    "3. Each step should be independently executable on a local machine.\n\n" +
    "Step ordering rules:\n" +
    "- Step 0: ALWAYS scaffolding / repo init (npx create-next-app, django-admin, etc.).\n" +
    "- Early steps: install dependencies, configure tooling (ESLint, Tailwind, etc.).\n" +
    "- Middle steps: backend → data models → API routes → frontend pages/components.\n" +
    "- Late steps: integration wiring, environment config.\n" +
    "- Final step: ALWAYS a test / build verification step (npm run build, pytest, etc.).\n\n" +
    "Terminal command rules:\n" +
    "- For scaffolding: use real CLI commands (npx create-next-app@latest <name> --ts --tailwind --app --use-npm, etc.).\n" +
    "- For AI code generation: use exactly this format:\n" +
    '  aider --message "<DETAILED INSTRUCTIONS>" --yes --auto-commits\n' +
    "  The message must be self-contained and describe ALL files to create/edit and their full contents.\n" +
    "- For installs: npm install <pkg>, pip install <pkg>, etc.\n" +
    "- For tests/builds: npm run build, npm test, pytest, etc.\n\n" +
    "CRITICAL:\n" +
    "- Every command must be a real, runnable shell command — no pseudocode.\n" +
    "- targetDirectory is relative to the project root. Use '.' for root.\n" +
    "- The plan must be complete: after all steps, the user should have a working, buildable project.\n" +
    "- Be specific in aider instructions — describe exact file paths, component names, API routes, DB schemas.\n" +
    "- Aim for 4-8 steps. Don't over-fragment.";

// ---------------------------------------------------------------------------
// Public function
// ---------------------------------------------------------------------------

export interface GenerateProjectRoadmapOptions {
    /** If set, the planner will use this exact project/repo name (kebab-case). */
    preferredProjectName?: string;
}

export async function generateProjectRoadmap(
    prompt: string,
    senderId?: string,
    options?: GenerateProjectRoadmapOptions,
): Promise<ProjectPlan> {
    let profileSnippet = "";
    if (senderId) {
        const profile = await getUserProfile(senderId);
        profileSnippet = formatProfileForPrompt(profile);
    }

    const nameInstruction =
        options?.preferredProjectName?.trim()
            ? `\n\nUse this exact project name (for repo and root directory): ${options.preferredProjectName.trim().replace(/\s+/g, "-")}\n`
            : "";

    const { object: plan } = await generateObject({
        model: openai(PLANNER_MODEL_NAME),
        schema: ProjectPlanSchema,
        messages: [
            { role: "system", content: PLANNER_SYSTEM_PROMPT + profileSnippet },
            {
                role: "user",
                content:
                    `Build the following project from scratch. Return a complete, sequenced build plan.${nameInstruction}\n` +
                    `USER PROMPT:\n${prompt}`,
            },
        ],
        ...withTelemetry("devmode.planner"),
    } as Parameters<typeof generateObject>[0]);

    console.info(
        `[planner] Generated roadmap for "${prompt.slice(0, 60)}": ` +
        `${plan.steps.length} steps, stack: ${plan.techStack}`,
    );

    return plan;
}
