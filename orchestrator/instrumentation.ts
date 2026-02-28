/**
 * instrumentation.ts — OpenTelemetry bootstrap for LLMOps tracing.
 *
 * Next.js calls `register()` once at server startup (both Node.js and Edge).
 * We use @vercel/otel with a LangSmith OTLP exporter so every Vercel AI SDK
 * span (generateText, generateObject, tool calls) is forwarded to LangSmith.
 *
 * Required env vars (add to .env.local):
 *   LANGSMITH_API_KEY      — your LangSmith API key
 *   LANGSMITH_PROJECT      — project name (e.g. "eureka-prod")
 *   LANGSMITH_ENDPOINT     — optional, defaults to https://api.smith.langchain.com
 */

import { registerOTel } from "@vercel/otel";

export function register() {
    const langsmithKey = process.env.LANGSMITH_API_KEY;

    if (!langsmithKey) {
        console.warn(
            "[otel] LANGSMITH_API_KEY not set — traces will use the default " +
            "OTLP exporter (or none). Set it to enable LangSmith tracing.",
        );
        registerOTel({ serviceName: "eureka-orchestrator" });
        return;
    }

    const endpoint =
        (process.env.LANGSMITH_ENDPOINT || "https://api.smith.langchain.com") +
        "/otel/v1/traces";

    process.env.OTEL_EXPORTER_OTLP_ENDPOINT = endpoint;
    process.env.OTEL_EXPORTER_OTLP_HEADERS = `x-api-key=${langsmithKey}`;
    process.env.OTEL_EXPORTER_OTLP_PROTOCOL = "http/protobuf";

    registerOTel({ serviceName: "eureka-orchestrator" });

    console.info(
        `[otel] LangSmith tracing enabled — project="${process.env.LANGSMITH_PROJECT || "(default)"}", endpoint="${endpoint}"`,
    );
}
