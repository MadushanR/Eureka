/**
 * lib/pushCallback.ts — Signed callback token for background push completion
 * ==========================================================================
 * Builds a completion webhook URL with a signed token so the daemon can
 * POST push results back; the push-complete route verifies the token and
 * sends a Telegram message to the right user.
 *
 * Token payload: { senderId: string, pushType: 'commit-and-push' | 'push-only', exp: number }
 * Signing: HMAC-SHA256(secret, base64url(payload)); token = base64url(payload).signature
 *
 * Requires: PUSH_CALLBACK_SECRET, and VERCEL_URL or PUSH_CALLBACK_BASE_URL for the base URL.
 */

import { createHmac } from "crypto";

const ALG = "sha256";
const EXPIRY_MS = 30 * 60 * 1000; // 30 minutes

export type PushType = "commit-and-push" | "push-only";

export interface PushCallbackPayload {
    senderId: string;
    pushType: PushType;
    exp: number;
}

function base64UrlEncode(buf: Buffer): string {
    return buf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlDecode(str: string): Buffer {
    const base64 = str.replace(/-/g, "+").replace(/_/g, "/");
    const pad = base64.length % 4;
    return Buffer.from(base64 + (pad ? "=".repeat(4 - pad) : ""), "base64");
}

/**
 * Create a signed token encoding senderId, pushType, and expiry.
 * Returns the full callback URL including the token as a query param.
 */
export function buildPushCompletionUrl(senderId: string, pushType: PushType): string | null {
    const secret = process.env.PUSH_CALLBACK_SECRET;
    const baseUrl = process.env.PUSH_CALLBACK_BASE_URL ?? (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : null);
    if (!secret || !baseUrl) return null;

    const payload: PushCallbackPayload = {
        senderId,
        pushType,
        exp: Date.now() + EXPIRY_MS,
    };
    const payloadJson = JSON.stringify(payload);
    const payloadB64 = base64UrlEncode(Buffer.from(payloadJson, "utf8"));
    const sig = createHmac(ALG, secret).update(payloadB64).digest();
    const sigB64 = base64UrlEncode(sig);
    const token = `${payloadB64}.${sigB64}`;
    const url = new URL("/api/webhooks/push-complete", baseUrl);
    url.searchParams.set("token", token);
    return url.toString();
}

/**
 * Verify the token and decode the payload. Returns null if invalid or expired.
 */
export function verifyPushCompletionToken(token: string): PushCallbackPayload | null {
    const secret = process.env.PUSH_CALLBACK_SECRET;
    if (!secret) return null;

    const dot = token.indexOf(".");
    if (dot <= 0) return null;
    const payloadB64 = token.slice(0, dot);
    const sigB64 = token.slice(dot + 1);

    const expectedSig = createHmac(ALG, secret).update(payloadB64).digest();
    const expectedB64 = base64UrlEncode(expectedSig);
    if (sigB64 !== expectedB64) return null;

    let payload: PushCallbackPayload;
    try {
        payload = JSON.parse(base64UrlDecode(payloadB64).toString("utf8")) as PushCallbackPayload;
    } catch {
        return null;
    }
    if (!payload.senderId || !payload.pushType || typeof payload.exp !== "number") return null;
    if (Date.now() > payload.exp) return null;
    return payload;
}
