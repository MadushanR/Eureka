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
        "Use 'aider --message \"<INSTRUCTIONS>\" --yes --auto-commits --no-pretty' for AI code-gen steps. " +
        "Use standard CLI commands (npm init, pip install, npx create-next-app, etc.) for scaffolding. " +
        "Leave empty ONLY if the step is purely an aider code-gen step with no other commands.",
    ),
    dependsOnStep: z.number().nullable().describe(
        "0-based index of a prior step this one depends on. Use null if this step has no dependency.",
    ),
    optional: z.boolean().describe(
        "If true, failure of this step is non-fatal — the build continues. " +
        "Mark test/build-verification and lint steps as optional: true. " +
        "All scaffolding, database, auth, and core API steps must be optional: false.",
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
        "Ordered list of build phases. Must start with scaffolding/setup. " +
        "Simple apps (landing page, CRUD app, CLI tool): 6-10 steps. " +
        "Complex apps (food delivery, marketplace, SaaS, multi-role, real-time): 12-20 steps. " +
        "End with an optional build/test verification step.",
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
    "You are a Senior Solutions Architect building a complete, production-grade project from a single user prompt.\n\n" +

    "YOUR JOB:\n" +
    "1. Choose a modern, appropriate tech stack.\n" +
    "2. Break the project into sequential, independently-executable BUILD STEPS.\n" +
    "3. Every step must be runnable on a local machine with no human input.\n\n" +

    "STEP COUNT RULES:\n" +
    "- Simple apps (landing page, basic CRUD, CLI tool, portfolio): 6–10 steps.\n" +
    "- Complex apps (food delivery, marketplace, SaaS, e-commerce, multi-role platforms, real-time apps): 12–20 steps.\n" +
    "  Complex apps REQUIRE: multi-role auth, database with migrations, separate backend+frontend, business logic API, and integration wiring.\n" +
    "  DO NOT compress a complex app into fewer steps — it will produce incomplete code.\n\n" +

    "CANONICAL STEP ORDER FOR COMPLEX APPS (monorepo backend/ + frontend/):\n" +
    "  0. Create root directory structure: mkdir -p backend frontend (targetDirectory '.')\n" +
    "  1. Scaffold backend: npm init -y in backend/ and install core runtime deps\n" +
    "  2. Scaffold frontend: Vite or Next.js in frontend/\n" +
    "  3. Set up Docker Compose + database service at root (targetDirectory '.')\n" +
    "  4. Install all backend dependencies (express/fastify, prisma, bcrypt, jsonwebtoken, socket.io, etc.) in backend/\n" +
    "  5. Create backend .env file and prisma/schema.prisma with ALL models\n" +
    "  6. Run: npx prisma migrate dev --name init in backend/\n" +
    "  7. aider: auth system — register/login routes, JWT signing, RBAC middleware\n" +
    "  8. aider: core domain models + CRUD API routes (restaurants, menus, products, etc.)\n" +
    "  9. aider: transactional API routes (orders, cart, checkout, status transitions)\n" +
    " 10. aider: secondary roles API (driver assignment, tracking, notifications)\n" +
    " 11. Install all frontend dependencies in frontend/\n" +
    " 12. aider: frontend auth pages + API client + auth context/store\n" +
    " 13. aider: frontend main pages (per user role — customer, restaurant owner, driver)\n" +
    " 14. aider: real-time features (socket.io client, live order tracking, WebSocket events)\n" +
    " 15. aider: integration wiring — connect frontend env vars to backend URL, CORS config\n" +
    " 16. [optional] Build verification: cd backend && npm run build; cd frontend && npm run build\n\n" +

    "AIDER PROMPT QUALITY RULES (CRITICAL):\n" +
    "- Every aider --message must be 200–1000 words. Short aider messages produce incomplete code.\n" +
    "- ALWAYS specify: exact file paths, complete function signatures, every API route (method + path + body + response), full database schema field names and types, exact import statements.\n" +
    "- Be 100% self-contained. Do not write 'etc.', 'similar to above', or 'add more routes as needed'.\n" +
    "- Example of a GOOD aider message for an orders route:\n" +
    '  "Create backend/routes/orders.js. Export an Express Router. Import prisma from \'../lib/prisma.js\' and requireAuth from \'../middleware/auth.js\'.\n' +
    "  POST /orders: body={restaurantId,items:[{menuItemId,quantity}]}. Validate restaurantId exists. Validate each menuItemId exists and belongs to restaurantId. Create order with status='PENDING' and total=sum(price*qty). Return 201 with full order including items.\n" +
    "  GET /orders/:id: return order with restaurant,items,driver. 403 if requesting user is not the customer, restaurant owner, or assigned driver.\n" +
    "  PATCH /orders/:id/status: body={status}. Allowed transitions: PENDING→ACCEPTED (restaurant), ACCEPTED→PREPARING (restaurant), PREPARING→READY (restaurant), READY→PICKED_UP (driver), PICKED_UP→DELIVERED (driver). Return updated order.\n" +
    '  All routes require requireAuth middleware. Return 404 if not found, 400 for invalid input."\n' +
    "- Example of a BAD aider message: 'Create order routes with CRUD operations and status updates.'\n\n" +

    "TERMINAL COMMAND RULES:\n" +
    "- All commands run HEADLESS — never prompt for input.\n" +
    "- aider format: aider --message \"<DETAILED INSTRUCTIONS>\" --yes --auto-commits --no-pretty\n" +
    "  --no-pretty prevents ANSI codes from polluting logs. Always include it.\n" +
    "- create-next-app: npx create-next-app@latest . --yes --ts --tailwind --eslint --app --use-npm\n" +
    "  Directory MUST be '.' — never pass project name as directory (creates nested folders).\n" +
    "- create-vite: npm create vite@latest <name> -- --template react-ts\n" +
    "- npm init / yarn init: always use -y flag.\n" +
    "- Each terminal command runs with targetDirectory as cwd. No shared shell state between commands.\n" +
    "  If you need to run in a subdir: use 'cd subdir && command' as a single command.\n" +
    "- Prisma: prisma init and prisma migrate must run from the directory containing prisma/schema.prisma.\n" +
    "- .env files: double-quote any value with =, #, or spaces: DATABASE_URL=\"postgresql://user:pass@host/db\"\n" +
    "- Docker Compose: use 'docker compose up -d' (not 'docker-compose') to start DB services.\n" +
    "- mkdir: on Windows Git Bash use 'mkdir -p dir1 dir2' — supported in Git Bash.\n\n" +

    "DOCKER COMPOSE FOR DATABASES:\n" +
    "- Whenever the app needs PostgreSQL, MySQL, MongoDB, or Redis: include a step to create docker-compose.yml at the project root.\n" +
    "- The step's terminal commands: echo the docker-compose.yml content then run 'docker compose up -d'.\n" +
    "- Example docker-compose.yml for Postgres:\n" +
    "  version: '3.8'\\nservices:\\n  db:\\n    image: postgres:16\\n    environment:\\n      POSTGRES_USER: app\\n      POSTGRES_PASSWORD: secret\\n      POSTGRES_DB: appdb\\n    ports:\\n      - '5432:5432'\n\n" +

    "ENVIRONMENT VARIABLES:\n" +
    "- Create .env files early (step 1–3). Include ALL variables the app will need, even if some are placeholders.\n" +
    "- Backend .env should include: DATABASE_URL, JWT_SECRET, PORT, CORS_ORIGIN, and any API keys.\n" +
    "- Frontend .env should include: VITE_API_URL or NEXT_PUBLIC_API_URL pointing to backend.\n\n" +

    "OPTIONAL STEPS:\n" +
    "- Mark test, build verification, and lint steps as optional: true.\n" +
    "- All scaffolding, database setup, auth, and core API steps must be optional: false.\n\n" +

    "CRITICAL RULES:\n" +
    "- Every command must be a real, runnable shell command — no pseudocode, no placeholders.\n" +
    "- targetDirectory is relative to the project root. Use '.' for root.\n" +
    "- The final project must be complete and functional: working auth, real data flows, no stub implementations.\n" +
    "- Do not skip steps to save time. A missing auth step or DB migration step will break everything downstream.";

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
    } as Parameters<typeof generateObject>[0]) as { object: ProjectPlan };

    console.info(
        `[planner] Generated roadmap for "${prompt.slice(0, 60)}": ` +
        `${plan.steps.length} steps, stack: ${plan.techStack}`,
    );

    return plan;
}
