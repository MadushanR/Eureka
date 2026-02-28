/**
 * lib/ai/researchAgents.ts — Multi-Agent Research Pipeline
 * =========================================================
 * Three-agent loop:
 *   A) Researcher  — gathers raw facts and citations on a topic
 *   B) Writer      — turns facts into a structured Markdown paper
 *   C) Reviewer    — critiques the draft; Writer revises (max 2 rounds)
 *
 * Entry point: runResearchPipeline(topic)
 */

import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

const DEFAULT_MODEL = process.env.LLM_MODEL_NAME_DEV || process.env.LLM_MODEL_NAME || "gpt-4.1-mini";

/** Parse LLM_MODELS_RESEARCH (comma-separated) from env. Falls back to [DEFAULT_MODEL] if unset or empty. */
function getResearchModelNames(): string[] {
    const raw = process.env.LLM_MODELS_RESEARCH?.trim();
    if (!raw) return [DEFAULT_MODEL];
    const list = raw.split(",").map((s) => s.trim()).filter(Boolean);
    return list.length > 0 ? list : [DEFAULT_MODEL];
}

const researchModelNames = getResearchModelNames();

/** Get the model for a given research agent. Index: 0=Researcher, 1=Writer, 2=Reviewer, 3=Summariser. */
function getResearchModel(agentIndex: 0 | 1 | 2 | 3): ReturnType<typeof openai> {
    const name = researchModelNames[agentIndex] ?? researchModelNames[1] ?? researchModelNames[0] ?? DEFAULT_MODEL;
    return openai(name);
}

const MAX_REVIEW_ROUNDS = 2;

/** Min/max target words allowed when user says "N words" (avoid abuse or uselessly short). */
const MIN_TARGET_WORDS = 200;
const MAX_TARGET_WORDS = 15_000;

/**
 * Parse user message for research topic and optional target word count.
 * Handles phrases like "500 words", "in 1000 words", "approximately 2000 words".
 */
export function parseResearchMessage(text: string): { topic: string; targetWords?: number } {
    const trimmed = text.trim();
    const wordCountMatch = trimmed.match(/\b(?:in|of|about|approximately|~|around)?\s*(\d{1,5})\s*words?\b/i);
    const targetWords = wordCountMatch ? parseInt(wordCountMatch[1], 10) : undefined;
    const clamped =
        targetWords != null
            ? Math.min(MAX_TARGET_WORDS, Math.max(MIN_TARGET_WORDS, targetWords))
            : undefined;
    const topic = wordCountMatch
        ? trimmed
              .replace(/\b(?:in|of|about|approximately|~|around)?\s*\d{1,5}\s*words?\b/gi, "")
              .replace(/\s+/g, " ")
              .trim()
        : trimmed;
    return { topic: topic || trimmed, targetWords: clamped };
}

/** Count words in a string (for display). */
export function countWords(s: string): number {
    return s.split(/\s+/).filter(Boolean).length;
}

// ---------------------------------------------------------------------------
// Agent A: Researcher
// ---------------------------------------------------------------------------

const RESEARCHER_PROMPT =
    "You are a meticulous Data Gatherer. Given a research topic, produce a comprehensive set of raw facts, statistics, and findings.\n\n" +
    "Rules:\n" +
    "- Cover at least 5 distinct subtopics or angles.\n" +
    "- For each fact, note a plausible source (e.g. 'WHO 2024 report', 'Nature, 2023').\n" +
    "- Use bullet points. Be thorough but concise — no fluff.\n" +
    "- Include both mainstream consensus and emerging/contrarian perspectives.\n" +
    "- End with a list of 'Key open questions' that remain unanswered.\n" +
    "- Output raw facts ONLY — do not write prose or a paper.";

export async function runResearcher(topic: string): Promise<string> {
    const result = await generateText({
        model: getResearchModel(0),
        messages: [
            { role: "system", content: RESEARCHER_PROMPT },
            { role: "user", content: `Research topic: "${topic}"\n\nGather comprehensive facts and cited data points on this topic.` },
        ],
    } as Parameters<typeof generateText>[0]);
    return (typeof result.text === "string" ? result.text.trim() : "") || "(no research output)";
}

// ---------------------------------------------------------------------------
// Agent B: Writer
// ---------------------------------------------------------------------------

const WRITER_PROMPT_BASE =
    "You are an Academic Writer. You receive raw research facts and must produce a well-structured Markdown research paper.\n\n" +
    "Paper structure:\n" +
    "1. **Title** — descriptive, specific\n" +
    "2. **Abstract** — 150-word summary (or proportionally shorter if total length is under 500 words)\n" +
    "3. **Introduction** — context and importance\n" +
    "4. **Findings** — organized into logical sections with ## headings\n" +
    "5. **Discussion** — analysis, implications, limitations\n" +
    "6. **Conclusion** — key takeaways and future directions\n" +
    "7. **References** — list all cited sources\n\n" +
    "Rules:\n" +
    "- Write in formal but accessible academic tone.\n" +
    "- Use Markdown formatting (headings, bold, lists, blockquotes for key stats).\n" +
    "- Integrate all provided facts — do not invent new data.\n" +
    "- Cite sources inline using [Author, Year] format.\n" +
    "- **Length**: ";

export async function runWriter(
    topic: string,
    researchFacts: string,
    feedback?: string,
    targetWords?: number,
): Promise<string> {
    const lengthInstruction =
        targetWords != null
            ? `Aim for approximately ${targetWords} words (stay within about ±15% of this).`
            : "Aim for 1500-3000 words.";
    const systemContent = WRITER_PROMPT_BASE + lengthInstruction;

    const userContent = feedback
        ? `Topic: "${topic}"\n\nResearch Facts:\n${researchFacts}\n\n---\nREVIEWER FEEDBACK (address all points):\n${feedback}\n\nPlease revise the paper addressing every piece of feedback above.`
        : `Topic: "${topic}"\n\nResearch Facts:\n${researchFacts}\n\nWrite a comprehensive research paper based on these facts.`;

    const result = await generateText({
        model: getResearchModel(1),
        messages: [
            { role: "system", content: systemContent },
            { role: "user", content: userContent },
        ],
    } as Parameters<typeof generateText>[0]);
    return (typeof result.text === "string" ? result.text.trim() : "") || "(no paper output)";
}

// ---------------------------------------------------------------------------
// Agent C: Reviewer
// ---------------------------------------------------------------------------

const REVIEWER_PROMPT =
    "You are a strict Academic Editor. Review the research paper draft below and provide feedback.\n\n" +
    "Evaluate on:\n" +
    "1. **Accuracy** — Are claims supported by the cited research facts?\n" +
    "2. **Structure** — Does it follow proper academic paper structure?\n" +
    "3. **Completeness** — Are all research facts incorporated?\n" +
    "4. **Clarity** — Is the writing clear and well-organized?\n" +
    "5. **Citations** — Are all sources properly referenced?\n\n" +
    "Reply with EXACTLY one of:\n" +
    "- APPROVED: <one sentence praising the paper>\n" +
    "- REVISE: <numbered list of specific issues that MUST be fixed>\n\n" +
    "Be strict but fair. Only approve if the paper is genuinely publication-ready.";

export async function runReviewer(draft: string, researchFacts: string): Promise<{ approved: boolean; feedback: string }> {
    const result = await generateText({
        model: getResearchModel(2),
        messages: [
            { role: "system", content: REVIEWER_PROMPT },
            {
                role: "user",
                content: `Original research facts:\n${researchFacts.slice(0, 3000)}\n\n---\n\nDraft paper to review:\n${draft}`,
            },
        ],
    } as Parameters<typeof generateText>[0]);
    const verdict = (typeof result.text === "string" ? result.text.trim() : "") || "";

    if (verdict.startsWith("APPROVED")) {
        return { approved: true, feedback: verdict };
    }
    return { approved: false, feedback: verdict.replace(/^REVISE:\s*/i, "") };
}

// ---------------------------------------------------------------------------
// Summariser — extract 3 bullet points for Telegram notification
// ---------------------------------------------------------------------------

export async function summarisePaper(paper: string): Promise<string> {
    const result = await generateText({
        model: getResearchModel(3),
        messages: [
            { role: "system", content: "Summarise the following research paper into exactly 3 concise bullet points. Each bullet should capture a key finding. Output ONLY the 3 bullets, no other text." },
            { role: "user", content: paper.slice(0, 6000) },
        ],
    } as Parameters<typeof generateText>[0]);
    return (typeof result.text === "string" ? result.text.trim() : "") || "- Research complete.";
}

// ---------------------------------------------------------------------------
// Full pipeline: Researcher → Writer → Reviewer loop → final paper
// ---------------------------------------------------------------------------

export interface ResearchResult {
    paper: string;
    summary: string;
    reviewRounds: number;
}

export type ResearchProgressCallback = (update: string) => Promise<void>;

/** Thrown when the user cancels research. Route should catch and send "Research cancelled." */
export class ResearchCancelledError extends Error {
    constructor() {
        super("Research was cancelled by the user.");
        this.name = "ResearchCancelledError";
    }
}

export interface ResearchPipelineOptions {
    targetWords?: number;
}

export async function runResearchPipeline(
    topic: string,
    onProgress?: ResearchProgressCallback,
    getIsCancelled?: () => Promise<boolean>,
    options?: ResearchPipelineOptions,
): Promise<ResearchResult> {
    const { targetWords } = options ?? {};
    const progress = async (msg: string) => {
        if (onProgress) {
            try { await onProgress(msg); } catch {}
        }
    };

    const checkCancelled = async (): Promise<void> => {
        if (getIsCancelled && (await getIsCancelled())) {
            throw new ResearchCancelledError();
        }
    };

    await checkCancelled();
    await progress("Researcher agent is gathering facts...");
    console.info(`[research] Step A: Researching "${topic.slice(0, 60)}"${targetWords != null ? ` (target: ${targetWords} words)` : ""}`);
    const facts = await runResearcher(topic);
    await checkCancelled();
    console.info(`[research] Researcher produced ${facts.length} chars of facts.`);
    await progress("Research complete. Writer is drafting the paper...");

    await checkCancelled();
    console.info(`[research] Step B: Writing initial draft`);
    let draft = await runWriter(topic, facts, undefined, targetWords);
    console.info(`[research] Initial draft: ${draft.length} chars.`);

    let reviewRounds = 0;
    for (let round = 0; round < MAX_REVIEW_ROUNDS; round++) {
        await checkCancelled();
        await progress(`Reviewer is checking draft (round ${round + 1}/${MAX_REVIEW_ROUNDS})...`);
        console.info(`[research] Step C: Review round ${round + 1}`);

        const review = await runReviewer(draft, facts);
        reviewRounds = round + 1;

        if (review.approved) {
            console.info(`[research] Reviewer approved on round ${round + 1}.`);
            await progress("Reviewer approved the paper!");
            break;
        }

        console.info(`[research] Reviewer requested revisions: ${review.feedback.slice(0, 120)}`);
        await progress(`Revisions requested. Writer is revising (round ${round + 1})...`);
        await checkCancelled();
        draft = await runWriter(topic, facts, review.feedback, targetWords);
        console.info(`[research] Revised draft: ${draft.length} chars.`);
    }

    await checkCancelled();
    await progress("Generating summary...");
    const summary = await summarisePaper(draft);

    return { paper: draft, summary, reviewRounds };
}
