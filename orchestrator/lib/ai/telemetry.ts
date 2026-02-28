/**
 * lib/ai/telemetry.ts — Shared telemetry config for Vercel AI SDK calls.
 *
 * Import `withTelemetry` and spread it into every generateText / generateObject
 * options object so spans are automatically exported to the configured OTEL
 * backend (LangSmith, Datadog, etc.).
 */

export function withTelemetry(functionId: string, metadata?: Record<string, string>) {
    return {
        experimental_telemetry: {
            isEnabled: true,
            functionId,
            metadata: metadata ?? {},
        },
    };
}
