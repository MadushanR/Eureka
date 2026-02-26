/**
 * types/messaging.ts — Platform-Agnostic Messaging Interfaces
 * ============================================================
 * This file defines the canonical, platform-agnostic types used throughout
 * the orchestrator.  Every messaging platform adapter (Telegram, Slack,
 * Discord, etc.) MUST translate to and from these types.
 *
 * Design rationale — Omni-Channel Adapter Pattern
 * -----------------------------------------------
 * By normalising all incoming webhooks into a single `StandardMessage` and
 * all outgoing replies into a single `StandardResponse`, the core application
 * logic never needs to know which platform sent the message.  New platforms
 * can be added simply by implementing the `MessageAdapter` interface.
 */

// ---------------------------------------------------------------------------
// Inbound — normalised representation of a message from any platform
// ---------------------------------------------------------------------------

/**
 * A platform-agnostic representation of an inbound user message.
 *
 * All platform adapters must produce this shape from their raw webhook
 * payloads via `MessageAdapter.parseWebhook`.
 */
export interface StandardMessage {
    /**
     * The plain-text body of the message.
     * Multi-line content is preserved; markdown formatting is stripped by the
     * adapter so the LLM receives clean text.
     */
    readonly text: string;

    /**
     * A stable, platform-scoped identifier for the conversation or user.
     *
     * For Telegram this is the numeric chat ID (cast to string).
     * For Slack this is the channel ID.
     * For Discord this is the channel ID.
     *
     * This value is used for both security gating (allowlist check) and
     * routing replies back to the correct conversation.
     */
    readonly senderId: string;

    /**
     * A human-readable identifier for the originating platform.
     * Used for logging, metrics, and routing decisions.
     * Conventionally lowercase (e.g. "telegram", "slack", "discord").
     */
    readonly platform: string;

    /**
     * Optional raw payload preserved from the original webhook.
     * Useful for platform-specific processing that doesn't fit the
     * normalised shape (e.g. voice messages, file attachments).
     * NOT forwarded to the LLM.
     */
    readonly rawPayload?: unknown;

    /**
     * Optional flag indicating that this message represents a user action
     * (e.g. a Telegram callback query from an inline button) rather than a
     * free-form text message.
     *
     * When `true`, `text` typically contains a machine-readable action
     * identifier such as "action:apply_patch:patch_123".
     */
    readonly isAction?: boolean;

    /**
     * Optional identifier for the underlying platform interaction.
     *
     * For Telegram callback queries this is `callback_query.id` and is used
     * when calling the `answerCallbackQuery` API to stop the client's loading
     * spinner.
     */
    readonly callbackQueryId?: string;
}

// ---------------------------------------------------------------------------
// Outbound — normalised representation of a reply to any platform
// ---------------------------------------------------------------------------

/**
 * A single interactive button that the orchestrator may include in a reply.
 * Platform adapters translate this into the platform's native button format.
 */
export interface InteractiveButton {
    /**
     * The machine-readable action identifier sent back to the orchestrator
     * when the user taps this button.  Should be a stable, URL-safe string
     * (e.g. "confirm_patch", "reject_diff").
     */
    readonly action: string;

    /**
     * The human-readable label displayed on the button.
     */
    readonly label: string;
}

/**
 * A platform-agnostic representation of a reply to be sent to the user.
 *
 * Adapters translate this into the platform's native sendMessage format
 * via `MessageAdapter.formatResponse`.
 */
export interface StandardResponse {
    /**
     * The main text body of the reply.
     * Adapters may apply platform-specific formatting (e.g. Telegram Markdown).
     */
    readonly text: string;

    /**
     * Optional set of interactive buttons to present below the message.
     * If the platform does not support interactive buttons the adapter should
     * fall back to appending the options as plain text.
     */
    readonly interactiveButtons?: ReadonlyArray<InteractiveButton>;
}

// ---------------------------------------------------------------------------
// Adapter contract
// ---------------------------------------------------------------------------

/**
 * The `MessageAdapter` interface is the core contract of the Omni-Channel
 * Adapter Pattern.  Every platform integration MUST implement both methods.
 *
 * @template TWebhookPayload  – The shape of the raw webhook body from the
 *                              platform (e.g. a Telegram `Update` object).
 * @template TPlatformMessage – The shape of the platform's native outbound
 *                              message payload (e.g. a Telegram sendMessage
 *                              request body).
 */
export interface MessageAdapter<TWebhookPayload = unknown, TPlatformMessage = unknown> {
    /**
     * Parse a raw webhook payload into a normalised `StandardMessage`.
     *
     * @param payload - The raw, unvalidated JSON body received from the platform.
     * @returns A `StandardMessage` ready for processing by the orchestrator.
     * @throws {Error} If the payload is missing required fields or is malformed.
     */
    parseWebhook(payload: TWebhookPayload): StandardMessage;

    /**
     * Translate a normalised `StandardResponse` into a platform-native message
     * payload, ready to be serialised and sent to the platform's API.
     *
     * @param response   - The orchestrator's reply.
     * @param receiverId - The platform-specific conversation / channel ID to
     *                     send the reply to (typically `StandardMessage.senderId`).
     * @returns A platform-native message payload object.
     */
    formatResponse(response: StandardResponse, receiverId: string): TPlatformMessage;

    /**
     * Deliver a `StandardResponse` to the platform by sending the formatted
     * message payload to the platform's HTTP API.
     *
     * This method combines `formatResponse` + the actual HTTP call so callers
     * don't need to know the API endpoint or auth mechanism.
     *
     * @param response   - The orchestrator's reply.
     * @param receiverId - The conversation / chat to send the reply to.
     * @returns A promise that resolves when the platform acknowledges the send.
     * @throws {Error} If the platform API returns a non-OK response.
     */
    sendResponse(response: StandardResponse, receiverId: string): Promise<void>;
}
