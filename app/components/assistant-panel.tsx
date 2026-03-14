"use client";

import { FormEvent, useMemo, useRef, useState } from "react";


type AssistantConfig = {
  universe: string[];
  discoverGlobalMovers: boolean;
  globalMoversLimit: number;
  benchmark: string;
  startDate: string;
  holdoutDays: number;
  topK: number;
  intradayPeriodDays: number;
  intradayInterval: string;
  enableLiveNews: boolean;
  enableLiveOptions: boolean;
  enableEarningsFeatures: boolean;
  startingCapital: number;
  maxNotionalPerTrade: number;
  executionMode: string;
};

type PredictionRow = {
  rank: number;
  symbol: string;
  side: string;
  opportunityScore: number;
  predictedReturn: number;
  predictedReturnLower: number;
  predictedReturnUpper: number;
  predictedMove: number;
  probabilityUp: number;
  confidence: number;
  finalScore: number;
  minutesToClose?: number | null;
};

type PredictionsResponse = {
  ok: boolean;
  generatedAt: string;
  request: Record<string, unknown>;
  summary: {
    predictionCount: number;
    topSymbol: string | null;
    topSide: string | null;
    avgTopKReturn: number | null;
    directionalAccuracy: number | null;
    signalBarIndex: number | null;
    minutesToClose: number | null;
    discoveredUniverse?: boolean;
  };
  predictions: PredictionRow[];
  metrics: Record<string, unknown>;
  executionPlan: Array<Record<string, unknown>>;
};

type AssistantMessage = {
  role: "assistant" | "user" | "system";
  content: string;
};

type WebLLMEngine = {
  chat: {
    completions: {
      create: (request: Record<string, unknown>) => Promise<any>;
    };
  };
  unload?: () => Promise<void>;
};


const DEFAULT_WEBLLM_MODEL = process.env.NEXT_PUBLIC_WEBLLM_MODEL || "Qwen3-8B-q4f32_1-MLC";

const MODEL_OPTIONS = [
  { id: "Qwen3-8B-q4f32_1-MLC", label: "Qwen3 8B q4f32" },
  { id: "Qwen3-4B-q4f32_1-MLC", label: "Qwen3 4B q4f32" },
  { id: "Qwen3-1.7B-q4f32_1-MLC", label: "Qwen3 1.7B q4f32" },
];


function percent(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${(value * 100).toFixed(2)}%`;
}


function number(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}


function buildToolSchema() {
  return [
    {
      type: "function",
      function: {
        name: "get_predictions",
        description: "Fetch fresh StockOracle predictions tailored to the user's interests and current market scan request.",
        parameters: {
          type: "object",
          properties: {
            universe: {
              type: "array",
              items: { type: "string" },
              description: "Optional symbol list to scan. Leave empty when the user wants a market-wide global mover scan.",
            },
            discoverGlobalMovers: {
              type: "boolean",
              description: "Set true when the user wants the tool to find top movers globally instead of using a manual symbol list.",
            },
            globalMoversLimit: { type: "integer", minimum: 10, maximum: 120 },
            benchmark: { type: "string" },
            topK: { type: "integer", minimum: 1, maximum: 10 },
            holdoutDays: { type: "integer", minimum: 15, maximum: 90 },
            intradayPeriodDays: { type: "integer", minimum: 10, maximum: 60 },
            intradayInterval: { type: "string", enum: ["5m", "15m", "30m", "60m"] },
            enableLiveNews: { type: "boolean" },
            enableLiveOptions: { type: "boolean" },
            enableEarningsFeatures: { type: "boolean" },
            startingCapital: { type: "number" },
            maxNotionalPerTrade: { type: "number" },
            executionMode: { type: "string", enum: ["paper", "alpaca"] },
          },
        },
      },
    },
  ];
}


function safeJsonParse(input: string | undefined): Record<string, unknown> {
  if (!input) {
    return {};
  }
  try {
    return JSON.parse(input) as Record<string, unknown>;
  } catch {
    return {};
  }
}


function reducePredictionPayload(payload: PredictionsResponse) {
  return {
    generatedAt: payload.generatedAt,
    summary: payload.summary,
    predictions: payload.predictions.slice(0, 6),
    metrics: {
      avgTopKReturn: payload.metrics.avg_top_k_return ?? null,
      directionalAccuracy: payload.metrics.directional_accuracy ?? null,
      backtestSharpe: payload.metrics.backtest_sharpe ?? null,
      minutesToClose: payload.metrics.median_minutes_to_close ?? null,
    },
  };
}


export function AssistantPanel({ config }: { config: AssistantConfig }) {
  const engineRef = useRef<WebLLMEngine | null>(null);
  const [modelId, setModelId] = useState(DEFAULT_WEBLLM_MODEL);
  const [interestPrompt, setInterestPrompt] = useState("Focus on same-day profit ideas in AI, semis, and strong momentum names. Prefer higher-confidence setups and explain the risk in plain language.");
  const [assistantMessages, setAssistantMessages] = useState<AssistantMessage[]>([
    {
      role: "assistant",
      content: "Load the browser model, describe your interests, and I will call the predictions API and translate the results into a trading-focused summary.",
    },
  ]);
  const [assistantError, setAssistantError] = useState<string | null>(null);
  const [assistantBusy, setAssistantBusy] = useState(false);
  const [modelBusy, setModelBusy] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [modelProgress, setModelProgress] = useState("Idle");
  const [lastPredictions, setLastPredictions] = useState<PredictionsResponse | null>(null);

  const browserSupportText = useMemo(() => {
    if (typeof window === "undefined") {
      return "Browser-only feature.";
    }
    if (!("gpu" in navigator)) {
      return "WebGPU is not available in this browser, so WebLLM cannot load here.";
    }
    return null;
  }, []);

  async function ensureEngine() {
    if (engineRef.current) {
      return engineRef.current;
    }
    if (browserSupportText) {
      throw new Error(browserSupportText);
    }

    setModelBusy(true);
    setModelProgress("Loading WebLLM runtime...");
    try {
      const webllm = await import("@mlc-ai/web-llm");
      const engine = await webllm.CreateMLCEngine(modelId, {
        initProgressCallback: (report: { progress?: number; text?: string }) => {
          const percentValue = typeof report.progress === "number" ? ` ${Math.round(report.progress * 100)}%` : "";
          setModelProgress(`${report.text || "Loading model..."}${percentValue}`);
        },
      });
      engineRef.current = engine as unknown as WebLLMEngine;
      setModelReady(true);
      setModelProgress(`Ready: ${modelId}`);
      return engineRef.current;
    } finally {
      setModelBusy(false);
    }
  }

  async function callPredictionsTool(overrides: Record<string, unknown>) {
    const overrideUniverse = Array.isArray(overrides.universe)
      ? overrides.universe.filter((value): value is string => typeof value === "string" && value.trim().length > 0)
      : [];
    const discoverGlobalMovers = typeof overrides.discoverGlobalMovers === "boolean"
      ? overrides.discoverGlobalMovers
      : config.discoverGlobalMovers;
    const payload = {
      universe: discoverGlobalMovers ? [] : (overrideUniverse.length ? overrideUniverse : config.universe),
      discoverGlobalMovers,
      globalMoversLimit: typeof overrides.globalMoversLimit === "number" ? overrides.globalMoversLimit : config.globalMoversLimit,
      benchmark: typeof overrides.benchmark === "string" ? overrides.benchmark : config.benchmark,
      startDate: config.startDate,
      holdoutDays: typeof overrides.holdoutDays === "number" ? overrides.holdoutDays : config.holdoutDays,
      topK: typeof overrides.topK === "number" ? overrides.topK : config.topK,
      intradayPeriodDays: typeof overrides.intradayPeriodDays === "number" ? overrides.intradayPeriodDays : config.intradayPeriodDays,
      intradayInterval: typeof overrides.intradayInterval === "string" ? overrides.intradayInterval : config.intradayInterval,
      enableLiveNews: typeof overrides.enableLiveNews === "boolean" ? overrides.enableLiveNews : config.enableLiveNews,
      enableLiveOptions: typeof overrides.enableLiveOptions === "boolean" ? overrides.enableLiveOptions : config.enableLiveOptions,
      enableEarningsFeatures: typeof overrides.enableEarningsFeatures === "boolean" ? overrides.enableEarningsFeatures : config.enableEarningsFeatures,
      startingCapital: typeof overrides.startingCapital === "number" ? overrides.startingCapital : config.startingCapital,
      maxNotionalPerTrade: typeof overrides.maxNotionalPerTrade === "number" ? overrides.maxNotionalPerTrade : config.maxNotionalPerTrade,
      executionMode: typeof overrides.executionMode === "string" ? overrides.executionMode : config.executionMode,
    };

    const response = await fetch("/api/predictions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    });
    const json = (await response.json()) as PredictionsResponse | { detail?: string; ok?: boolean };
    if (!response.ok || !("ok" in json && json.ok)) {
      throw new Error("detail" in json && json.detail ? json.detail : "Prediction tool call failed.");
    }
    setLastPredictions(json as PredictionsResponse);
    return reducePredictionPayload(json as PredictionsResponse);
  }

  async function onAnalyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAssistantBusy(true);
    setAssistantError(null);

    try {
      const engine = await ensureEngine();
      const tools = buildToolSchema();
      const currentUniverse = config.universe.join(", ");
      const baseMessages: Array<Record<string, unknown>> = [
        {
          role: "system",
          content:
            "You are the StockOracle browser analyst. Always call the get_predictions tool first before giving advice. Use the user's interests to refine the scan, stay grounded in the returned data, and explain the top setups, risks, and why the names fit the user's interests. Keep the answer concise and practical.",
        },
        {
          role: "user",
          content: `User interests: ${interestPrompt}\nCurrent scan defaults: universe=${config.discoverGlobalMovers ? "GLOBAL_MOVERS_DISCOVERY" : currentUniverse}; benchmark=${config.benchmark}; topK=${config.topK}; intradayInterval=${config.intradayInterval}; holdoutDays=${config.holdoutDays}; intradayPeriodDays=${config.intradayPeriodDays}; liveNews=${config.enableLiveNews}; liveOptions=${config.enableLiveOptions}; earningsFeatures=${config.enableEarningsFeatures}; globalMoversLimit=${config.globalMoversLimit}.`,
        },
      ];

      setAssistantMessages((messages) => [...messages, { role: "user", content: interestPrompt }]);

      const planning = await engine.chat.completions.create({
        messages: baseMessages,
        tools,
        tool_choice: { type: "function", function: { name: "get_predictions" } },
        temperature: 0.2,
        max_tokens: 300,
        extra_body: { enable_thinking: false },
      });

      const toolCall = planning?.choices?.[0]?.message?.tool_calls?.[0];
      const toolArgs = safeJsonParse(toolCall?.function?.arguments);
      const toolResult = await callPredictionsTool(toolArgs);

      const finalMessages = [
        ...baseMessages,
        {
          role: "assistant",
          content: planning?.choices?.[0]?.message?.content || "",
          tool_calls: toolCall ? [toolCall] : [],
        },
        {
          role: "tool",
          tool_call_id: toolCall?.id || "get_predictions",
          content: JSON.stringify(toolResult),
        },
      ];

      const finalResponse = await engine.chat.completions.create({
        messages: finalMessages,
        temperature: 0.35,
        max_tokens: 700,
        extra_body: { enable_thinking: false },
      });

      const content = finalResponse?.choices?.[0]?.message?.content || "I fetched the latest predictions, but the model did not return a natural-language summary.";
      setAssistantMessages((messages) => [...messages, { role: "assistant", content }]);
    } catch (error) {
      setAssistantError(error instanceof Error ? error.message : "WebLLM analysis failed.");
    } finally {
      setAssistantBusy(false);
    }
  }

  return (
    <section className="assistant-card">
      <div className="assistant-header">
        <div>
          <p className="eyebrow">WebLLM Analyst</p>
          <h2>Qwen assistant for your interests</h2>
        </div>
        <div className="assistant-status-group">
          <span className="assistant-chip">Browser model</span>
          <span className="assistant-chip assistant-chip-cool">{modelReady ? modelId : "Not loaded"}</span>
        </div>
      </div>

      <p className="empty-copy">
        Runs in the browser with WebLLM and calls the parser-friendly predictions API. The closest packaged browser model to Qwen3 9B is Qwen3 8B, which is used by default here.
      </p>

      <div className="assistant-toolbar">
        <label>
          WebLLM model
          <select value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={modelBusy || assistantBusy || modelReady}>
            {MODEL_OPTIONS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <button className="secondary-button" type="button" onClick={() => void ensureEngine()} disabled={modelBusy || modelReady || Boolean(browserSupportText)}>
          {modelBusy ? "Loading model..." : modelReady ? "Model ready" : "Load WebLLM model"}
        </button>
      </div>

      <p className="assistant-progress">{browserSupportText || modelProgress}</p>

      <form className="assistant-form" onSubmit={onAnalyze}>
        <label>
          What are you interested in?
          <textarea value={interestPrompt} onChange={(event) => setInterestPrompt(event.target.value)} rows={5} placeholder="Example: Find same-day long setups in semis with high confidence and avoid names with weak sentiment." />
        </label>
        <button className="primary-button" type="submit" disabled={assistantBusy || modelBusy || Boolean(browserSupportText)}>
          {assistantBusy ? "Analyzing with WebLLM..." : "Interpret predictions"}
        </button>
      </form>

      {assistantError ? <p className="error-copy">{assistantError}</p> : null}

      <div className="assistant-chat">
        {assistantMessages.map((message, index) => (
          <article key={`${message.role}-${index}`} className={`assistant-message assistant-message-${message.role}`}>
            <span>{message.role === "assistant" ? "Qwen" : "You"}</span>
            <p>{message.content}</p>
          </article>
        ))}
      </div>

      {lastPredictions ? (
        <div className="assistant-data-card">
          <div>
            <p className="eyebrow">API Snapshot</p>
            <h3>Latest structured predictions</h3>
          </div>
          <div className="assistant-summary-grid">
            <div className="feature-row">
              <span>Generated</span>
              <strong>{new Date(lastPredictions.generatedAt).toLocaleString()}</strong>
            </div>
            <div className="feature-row">
              <span>Top symbol</span>
              <strong>{lastPredictions.summary.topSymbol || "--"}</strong>
            </div>
            <div className="feature-row">
              <span>Top side</span>
              <strong>{lastPredictions.summary.topSide || "--"}</strong>
            </div>
            <div className="feature-row">
              <span>Avg top-k return</span>
              <strong>{percent(lastPredictions.summary.avgTopKReturn)}</strong>
            </div>
            <div className="feature-row">
              <span>Universe mode</span>
              <strong>{lastPredictions.summary.discoveredUniverse ? "Global movers" : "Manual list"}</strong>
            </div>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Pred Return</th>
                  <th>Range</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {lastPredictions.predictions.slice(0, 5).map((row) => (
                  <tr key={`${row.symbol}-${row.rank}`}>
                    <td>{row.rank}</td>
                    <td>{row.symbol}</td>
                    <td>{row.side}</td>
                    <td>{percent(row.predictedReturn)}</td>
                    <td>{`${percent(row.predictedReturnLower)} to ${percent(row.predictedReturnUpper)}`}</td>
                    <td>{percent(row.confidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <pre className="assistant-json">{JSON.stringify(reducePredictionPayload(lastPredictions), null, 2)}</pre>
        </div>
      ) : null}
    </section>
  );
}