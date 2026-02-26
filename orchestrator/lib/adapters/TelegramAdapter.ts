/**
 * lib/adapters/TelegramAdapter.ts — Telegram Platform Adapter
 * ============================================================
 * Implements the `MessageAdapter` interface for the Telegram Bot API.
 *
 * Telegram API reference: https://core.telegram.org/bots/api
 *
 * Responsibilities
 * ----------------
 * 1. `parseWebhook`   — Extract text + chat ID from a Telegram `Update`.
 * 2. `formatResponse` — Build a Telegram `sendMessage` payload, translating
 *                       `interactiveButtons` → `InlineKeyboardMarkup`.
 * 3. `sendResponse`   — Actually POST the payload to the Telegram Bot API.
 *
 * Environment variables required
 * ------------------------------
 *   TELEGRAM_BOT_TOKEN — The secret token from @BotFather.
 *
 * Security note
 * -------------
 * This adapter does NOT perform the allowed-chat-ID check — that is the
 * responsibility of the API route handler so that the policy is enforced in
 * one place and cannot be bypassed by swapping adapters.
 */

import type {
    MessageAdapter,
    StandardMessage,
    StandardResponse,
    InteractiveButton,
} from "@/types/messaging";

// ---------------------------------------------------------------------------
// Telegram-specific type definitions
// (We define only the fields we actually use to keep the types tight.)
// ---------------------------------------------------------------------------

/** A Telegram User object (partial). */
interface TelegramUser {
    readonly id: number;
    readonly is_bot: boolean;
    readonly first_name: string;
    readonly username?: string;
}

/** A Telegram Chat object (partial). */
interface TelegramChat {
    readonly id: number;
    readonly type: "private" | "group" | "supergroup" | "channel";
    readonly title?: string;
    readonly username?: string;
}

/** A Telegram Message object (partial). */
interface TelegramMessage {
    readonly message_id: number;
    readonly from?: TelegramUser;
    readonly chat: TelegramChat;
    readonly date: number;
    readonly text?: string;
    readonly caption?: string; // text for media messages
}

/** A Telegram callback query object (partial). */
interface TelegramCallbackQuery {
    readonly id: string;
    readonly from: TelegramUser;
    readonly data?: string;
    readonly message?: TelegramMessage;
}

/**
 * A Telegram Update object — the root of every webhook payload.
 * https://core.telegram.org/bots/api#update
 */
export interface TelegramUpdate {
    readonly update_id: number;
    readonly message?: TelegramMessage;
    readonly edited_message?: TelegramMessage;
    readonly channel_post?: TelegramMessage;
    readonly edited_channel_post?: TelegramMessage;
    readonly callback_query?: TelegramCallbackQuery;
}

// ---------------------------------------------------------------------------
// Telegram outbound payload types
// ---------------------------------------------------------------------------

/** A single button in an inline keyboard row. */
interface TelegramInlineKeyboardButton {
    readonly text: string;
    /** Callback data sent to the bot when the button is tapped (≤ 64 bytes). */
    readonly callback_data: string;
}

/**
 * Telegram's InlineKeyboardMarkup.
 * `inline_keyboard` is an array of rows; each row is an array of buttons.
 * https://core.telegram.org/bots/api#inlinekeyboardmarkup
 */
interface TelegramInlineKeyboardMarkup {
    readonly inline_keyboard: ReadonlyArray<ReadonlyArray<TelegramInlineKeyboardButton>>;
}

/** The body of a Telegram sendMessage API call. */
export interface TelegramSendMessagePayload {
    readonly chat_id: number | string;
    readonly text: string;
    /** "HTML" | "Markdown" | "MarkdownV2" */
    readonly parse_mode?: string;
    readonly reply_markup?: TelegramInlineKeyboardMarkup;
}

/** The shape of a successful Telegram API response. */
interface TelegramApiResponse<T = unknown> {
    readonly ok: boolean;
    readonly result?: T;
    readonly error_code?: number;
    readonly description?: string;
}

// ---------------------------------------------------------------------------
// Adapter implementation
// ---------------------------------------------------------------------------

/**
 * `TelegramAdapter` translates between Telegram's webhook format and the
 * orchestrator's platform-agnostic `StandardMessage` / `StandardResponse`.
 */
export class TelegramAdapter
    implements MessageAdapter<TelegramUpdate, TelegramSendMessagePayload> {
    /** The full Telegram Bot API base URL, pre-filled with the bot token. */
    private readonly apiBaseUrl: string;

    /**
     * @param botToken - The Telegram bot token from `process.env.TELEGRAM_BOT_TOKEN`.
     *                   Injected so the adapter is testable in isolation.
     */
    constructor(botToken: string) {
        if (!botToken) {
            throw new Error(
                "TelegramAdapter: botToken must be a non-empty string. " +
                "Set TELEGRAM_BOT_TOKEN in your .env.local file."
            );
        }
        this.apiBaseUrl = `https://api.telegram.org/bot${botToken}`;
    }

    // -------------------------------------------------------------------------
    // MessageAdapter.parseWebhook
    // -------------------------------------------------------------------------

    /**
     * Parse a raw Telegram `Update` webhook payload into a `StandardMessage`.
     *
     * Supports:
     *   - `message`, `edited_message`, `channel_post`, `edited_channel_post`
     *     (normal text messages)
     *   - `callback_query` (inline button presses)
     *
     * For normal messages we extract:
     *   - `text`        — from `message.text` or `message.caption`
     *   - `senderId`    — from `message.chat.id`
     *
     * For callback queries we extract:
     *   - `text`            — from `callback_query.data` (action identifier)
     *   - `senderId`        — from `callback_query.from.id`
     *   - `isAction`        — `true`
     *   - `callbackQueryId` — from `callback_query.id`
     *
     * @throws {Error} If the payload contains no recognisable message with text.
     */
    public parseWebhook(payload: TelegramUpdate): StandardMessage {
        // 1) Callback query (inline button press).
        if (payload.callback_query) {
            const { callback_query } = payload;
            const actionData = callback_query.data;

            if (!actionData || actionData.trim() === "") {
                throw new Error(
                    "TelegramAdapter.parseWebhook: callback_query contains no data. " +
                    `(update_id=${payload.update_id}).`
                );
            }

            return {
                text: actionData.trim(),
                // For actions we treat the user ID as the sender identifier;
                // replies still go to the associated chat ID via the route.
                senderId: String(callback_query.from.id),
                platform: "telegram",
                rawPayload: payload,
                isAction: true,
                callbackQueryId: callback_query.id,
            };
        }

        // 2) Normal text message variants.
        // Telegram sends one of several message variants per update.
        // We prefer `message` but fall back to the others in a consistent order.
        const message =
            payload.message ??
            payload.edited_message ??
            payload.channel_post ??
            payload.edited_channel_post;

        if (!message) {
            throw new Error(
                "TelegramAdapter.parseWebhook: Update contains no recognisable message " +
                `(update_id=${payload.update_id}). ` +
                "Non-message updates (inline queries, polls, etc.) are not supported."
            );
        }

        // `text` is set for plain text messages; `caption` is set when the user
        // sends a photo/video with a description.
        const text = message.text ?? message.caption;

        if (text === undefined || text.trim() === "") {
            throw new Error(
                "TelegramAdapter.parseWebhook: Message contains no text content " +
                `(message_id=${message.message_id}). ` +
                "Voice, sticker, and other media-only messages are not yet supported."
            );
        }

        return {
            text: text.trim(),
            // The chat ID is the correct identifier for replies — using `from.id`
            // would break group and channel conversations.
            senderId: String(message.chat.id),
            platform: "telegram",
            rawPayload: payload,
        };
    }

    // -------------------------------------------------------------------------
    // MessageAdapter.formatResponse
    // -------------------------------------------------------------------------

    /**
     * Translate a `StandardResponse` into a Telegram `sendMessage` payload.
     *
     * `interactiveButtons` are mapped to an `InlineKeyboardMarkup`, laying out
     * one button per row for readability.  The `action` string is used as the
     * `callback_data` (must be ≤ 64 bytes per Telegram's spec — longer values
     * are silently truncated here with a warning).
     *
     * @param response   - The orchestrator's reply.
     * @param receiverId - The Telegram chat ID to send the reply to.
     */
    public formatResponse(
        response: StandardResponse,
        receiverId: string
    ): TelegramSendMessagePayload {
        const payload: TelegramSendMessagePayload = {
            chat_id: receiverId,
            // Use MarkdownV2 so the LLM can include code blocks, bold text, etc.
            // Note: MarkdownV2 requires escaping certain characters — for now we
            // send as plain text to avoid unexpected rendering failures.
            text: response.text,
        };

        if (response.interactiveButtons && response.interactiveButtons.length > 0) {
            const keyboard = this.buildInlineKeyboard(response.interactiveButtons);
            return { ...payload, reply_markup: keyboard };
        }

        return payload;
    }

    // -------------------------------------------------------------------------
    // MessageAdapter.sendResponse
    // -------------------------------------------------------------------------

    /**
     * Send a `StandardResponse` to a Telegram chat by calling the Bot API.
     *
     * Uses the native `fetch` API (available in Node.js 18+ and the Next.js
     * Edge / Node runtimes).
     *
     * @throws {Error} If the Telegram API returns `ok: false` or the network
     *                 request fails.
     */
    public async sendResponse(
        response: StandardResponse,
        receiverId: string
    ): Promise<void> {
        const payload = this.formatResponse(response, receiverId);
        await this.callTelegramApi("sendMessage", payload);
    }

    /**
     * Answer a callback query to stop the loading spinner on the user's client.
     *
     * This should be called as soon as possible after receiving a callback
     * query update, ideally before performing any heavy processing.
     *
     * https://core.telegram.org/bots/api#answercallbackquery
     */
    public async answerCallbackQuery(callbackQueryId: string): Promise<void> {
        if (!callbackQueryId) return;

        await this.callTelegramApi("answerCallbackQuery", {
            callback_query_id: callbackQueryId,
        });
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Map a flat array of `InteractiveButton` objects to Telegram's
     * `InlineKeyboardMarkup` structure (one button per row).
     *
     * Telegram limits `callback_data` to 64 bytes.  If an `action` string
     * exceeds this limit it is truncated and a warning is emitted.
     */
    private buildInlineKeyboard(
        buttons: ReadonlyArray<InteractiveButton>
    ): TelegramInlineKeyboardMarkup {
        const MAX_CALLBACK_DATA_BYTES = 64;

        const rows = buttons.map((btn): ReadonlyArray<TelegramInlineKeyboardButton> => {
            let callbackData = btn.action;

            if (Buffer.byteLength(callbackData, "utf8") > MAX_CALLBACK_DATA_BYTES) {
                console.warn(
                    `[TelegramAdapter] callback_data for action "${btn.action}" exceeds ` +
                    `${MAX_CALLBACK_DATA_BYTES} bytes and will be truncated. ` +
                    "Consider using a shorter action identifier."
                );
                // Truncate to a safe length (conservative — ASCII only, so 1 byte/char).
                callbackData = callbackData.slice(0, MAX_CALLBACK_DATA_BYTES);
            }

            return [
                {
                    text: btn.label,
                    callback_data: callbackData,
                },
            ];
        });

        return { inline_keyboard: rows };
    }

    /**
     * Make a POST request to a Telegram Bot API method.
     *
     * @param method  - The API method name (e.g. "sendMessage").
     * @param body    - The JSON-serialisable request body.
     * @throws {Error} If the HTTP request fails or Telegram returns `ok: false`.
     */
    private async callTelegramApi<TResult>(
        method: string,
        body: unknown
    ): Promise<TelegramApiResponse<TResult>> {
        const url = `${this.apiBaseUrl}/${method}`;

        let httpResponse: Response;
        try {
            httpResponse = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                // Prevent the request from hanging indefinitely.
                // next: { revalidate: 0 } is used to opt out of Next.js caching.
                cache: "no-store",
            });
        } catch (networkError) {
            throw new Error(
                `[TelegramAdapter] Network error while calling ${method}: ${networkError instanceof Error ? networkError.message : String(networkError)
                }`
            );
        }

        // Parse the response body regardless of HTTP status so we can surface
        // Telegram's error description in the thrown error message.
        const apiResponse = (await httpResponse.json()) as TelegramApiResponse<TResult>;

        if (!apiResponse.ok) {
            throw new Error(
                `[TelegramAdapter] Telegram API error on ${method}: ` +
                `HTTP ${httpResponse.status} — ` +
                `code=${apiResponse.error_code ?? "unknown"} ` +
                `description="${apiResponse.description ?? "no description"}"`
            );
        }

        return apiResponse;
    }
}
