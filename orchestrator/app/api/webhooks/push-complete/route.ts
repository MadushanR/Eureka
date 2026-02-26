/**
 * app/api/webhooks/push-complete/route.ts — Daemon push completion callback
 * ==========================================================================
 * The daemon POSTs here when a background push finishes. We verify the
 * signed token, then send a Telegram message to the user.
 *
 * Query: token (signed payload with senderId, pushType, exp)
 * Body: { success: boolean, message: string, stdout?: string, stderr?: string }
 *
 * Requires: PUSH_CALLBACK_SECRET, TELEGRAM_BOT_TOKEN, ALLOWED_TELEGRAM_CHAT_IDS.
 */

import { NextRequest, NextResponse } from "next/server";
import { TelegramAdapter } from "@/lib/adapters/TelegramAdapter";
import { verifyPushCompletionToken } from "@/lib/pushCallback";

const ALLOWED_CHAT_IDS = new Set(
    (process.env.ALLOWED_TELEGRAM_CHAT_IDS ?? "")
        .split(",")
        .map((id) => id.trim())
        .filter(Boolean)
);

export async function POST(request: NextRequest): Promise<NextResponse> {
    const token = request.nextUrl.searchParams.get("token");
    if (!token) {
        return NextResponse.json({ error: "Missing token" }, { status: 400 });
    }

    const payload = verifyPushCompletionToken(token);
    if (!payload) {
        return NextResponse.json({ error: "Invalid or expired token" }, { status: 401 });
    }

    if (ALLOWED_CHAT_IDS.size > 0 && !ALLOWED_CHAT_IDS.has(payload.senderId)) {
        return NextResponse.json({ error: "Unauthorised" }, { status: 403 });
    }

    let body: { success?: boolean; message?: string; stdout?: string; stderr?: string };
    try {
        body = await request.json();
    } catch {
        return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
    }

    const success = body.success === true;
    const message = body.message ?? (success ? "Done." : "Failed.");
    const detail = [body.stdout, body.stderr].filter(Boolean).join("\n").trim();
    const text = success
        ? `Push completed. ${message}${detail ? `\n\n${detail}` : ""}`
        : `Push failed. ${message}${detail ? `\n\n${detail}` : ""}`;

    const botToken = process.env.TELEGRAM_BOT_TOKEN;
    if (!botToken) {
        console.error("[push-complete] TELEGRAM_BOT_TOKEN not set");
        return NextResponse.json({ error: "Server misconfiguration" }, { status: 500 });
    }

    try {
        const adapter = new TelegramAdapter(botToken);
        await adapter.sendResponse({ text }, payload.senderId);
    } catch (err) {
        console.error("[push-complete] Failed to send Telegram message:", err);
        return NextResponse.json({ error: "Failed to send notification" }, { status: 500 });
    }

    return NextResponse.json({ ok: true });
}
