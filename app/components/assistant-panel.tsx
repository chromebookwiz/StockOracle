"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";


type AssistantConfig = {
  universe: string[];
  discoverGlobalMovers: boolean;
  globalMoversLimit: number;
  operatorAuthenticated: boolean;
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

type MoversResponse = {
  symbols: string[];
  count: number;
  source?: string;
};

type PositionsResponse = {
  positions?: Array<{ symbol: string; quantity: number; avgPrice: number }>;
  orders?: Array<Record<string, unknown>>;
  detail?: string;
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

const WINDOW_WIDTH = 460;
const WINDOW_GAP = 20;


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
    {
      type: "function",
      function: {
        name: "get_global_movers",
        description: "Fetch the live discovered global mover universe before ranking so you can inspect what the scanner is considering.",
        parameters: {
          type: "object",
          properties: {
            limit: { type: "integer", minimum: 10, maximum: 120 },
          },
        },
      },
    },
    {
      type: "function",
      function: {
        name: "get_positions",
        description: "Fetch the current paper or Alpaca positions when portfolio context is relevant. This may require operator sign-in.",
        parameters: {
          type: "object",
          properties: {
            mode: { type: "string", enum: ["paper", "alpaca"] },
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
  const windowRef = useRef<HTMLDivElement | null>(null);
  const dragStateRef = useRef<{ offsetX: number; offsetY: number } | null>(null);
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
  const [assistantActivity, setAssistantActivity] = useState("Idle");
  const [lastPredictions, setLastPredictions] = useState<PredictionsResponse | null>(null);
  const [lastGlobalMovers, setLastGlobalMovers] = useState<MoversResponse | null>(null);
  const [lastPositions, setLastPositions] = useState<PositionsResponse | null>(null);
  const [isOpen, setIsOpen] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const toolSummary = useMemo(
    () => [
      "Predictions",
      "Global movers",
      config.operatorAuthenticated ? "Positions" : "Positions locked",
    ],
    [config.operatorAuthenticated],
  );

  const browserSupportText = useMemo(() => {
    if (typeof window === "undefined") {
      return "Browser-only feature.";
    }
    if (!("gpu" in navigator)) {
      return "WebGPU is not available in this browser, so WebLLM cannot load here.";
    }
    return null;
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const nextX = Math.max(WINDOW_GAP, window.innerWidth - WINDOW_WIDTH - WINDOW_GAP);
    setPosition({ x: nextX, y: 96 });
  }, []);

  useEffect(() => {
    function onPointerMove(event: PointerEvent) {
      if (!dragStateRef.current) {
        return;
      }
      const maxX = Math.max(WINDOW_GAP, window.innerWidth - WINDOW_WIDTH - WINDOW_GAP);
      const maxY = Math.max(WINDOW_GAP, window.innerHeight - 96);
      const nextX = Math.min(Math.max(WINDOW_GAP, event.clientX - dragStateRef.current.offsetX), maxX);
      const nextY = Math.min(Math.max(WINDOW_GAP, event.clientY - dragStateRef.current.offsetY), maxY);
      setPosition({ x: nextX, y: nextY });
    }

    function onPointerUp() {
      dragStateRef.current = null;
    }

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
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

  async function callGlobalMoversTool(overrides: Record<string, unknown>) {
    const limit = typeof overrides.limit === "number" ? overrides.limit : config.globalMoversLimit;
    const response = await fetch(`/api/universe/global-movers?limit=${encodeURIComponent(String(limit))}`, {
      cache: "no-store",
    });
    const json = (await response.json()) as MoversResponse | { detail?: string };
    if (!response.ok || !("symbols" in json)) {
      throw new Error("detail" in json && json.detail ? json.detail : "Global mover scan failed.");
    }
    setLastGlobalMovers(json as MoversResponse);
    return json;
  }

  async function callPositionsTool(overrides: Record<string, unknown>) {
    const mode = typeof overrides.mode === "string" ? overrides.mode : config.executionMode;
    const response = await fetch(`/api/trade/positions?mode=${encodeURIComponent(mode)}`, {
      cache: "no-store",
    });
    const json = (await response.json()) as PositionsResponse;
    setLastPositions(json);
    return {
      ok: response.ok,
      mode,
      positions: json.positions ?? [],
      orders: Array.isArray(json.orders) ? json.orders.slice(0, 10) : [],
      detail: json.detail ?? null,
    };
  }

  async function executeTool(name: string, args: Record<string, unknown>) {
    if (name === "get_predictions") {
      setAssistantActivity("Fetching predictions");
      return callPredictionsTool(args);
    }
    if (name === "get_global_movers") {
      setAssistantActivity("Scanning global movers");
      return callGlobalMoversTool(args);
    }
    if (name === "get_positions") {
      setAssistantActivity("Checking portfolio positions");
      return callPositionsTool(args);
    }
    throw new Error(`Unknown tool: ${name}`);
  }

  function startDrag(event: React.PointerEvent<HTMLDivElement>) {
    if (!windowRef.current) {
      return;
    }
    const target = event.target as HTMLElement;
    if (target.closest("button, input, select, textarea")) {
      return;
    }
    const rect = windowRef.current.getBoundingClientRect();
    dragStateRef.current = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
  }

  function openWindow() {
    setIsOpen(true);
    setIsMinimized(false);
  }

  async function onAnalyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAssistantBusy(true);
    setAssistantError(null);
    setAssistantActivity("Preparing advisor run");

    try {
      const engine = await ensureEngine();
      const tools = buildToolSchema();
      const currentUniverse = config.universe.join(", ");
      const baseMessages: Array<Record<string, unknown>> = [
        {
          role: "system",
          content:
            "You are the StockOracle browser advisor. Use tools whenever needed before giving advice. You can inspect global movers, fetch ranked predictions, and review current positions. Base recommendations only on returned tool data, explain risk and uncertainty clearly, and keep the answer concise and practical.",
        },
        {
          role: "user",
          content: `User interests: ${interestPrompt}\nCurrent scan defaults: universe=${config.discoverGlobalMovers ? "GLOBAL_MOVERS_DISCOVERY" : currentUniverse}; benchmark=${config.benchmark}; topK=${config.topK}; intradayInterval=${config.intradayInterval}; holdoutDays=${config.holdoutDays}; intradayPeriodDays=${config.intradayPeriodDays}; liveNews=${config.enableLiveNews}; liveOptions=${config.enableLiveOptions}; earningsFeatures=${config.enableEarningsFeatures}; globalMoversLimit=${config.globalMoversLimit}; positionsAvailable=${config.operatorAuthenticated}.`,
        },
      ];

      setAssistantMessages((messages) => [...messages, { role: "user", content: interestPrompt }]);
      const conversation: Array<Record<string, unknown>> = [...baseMessages];
      let content = "I gathered the available evidence, but the model did not return a final advisory note.";

      for (let round = 0; round < 4; round += 1) {
        const response = await engine.chat.completions.create({
          messages: conversation,
          tools,
          tool_choice: "auto",
          temperature: 0.25,
          max_tokens: 500,
          extra_body: { enable_thinking: false },
        });

        const message = response?.choices?.[0]?.message;
        const toolCalls = Array.isArray(message?.tool_calls) ? message.tool_calls : [];
        const messageContent = typeof message?.content === "string" ? message.content : "";

        conversation.push({
          role: "assistant",
          content: messageContent,
          tool_calls: toolCalls,
        });

        if (!toolCalls.length) {
          content = messageContent || content;
          break;
        }

        for (const toolCall of toolCalls) {
          const toolName = typeof toolCall?.function?.name === "string" ? toolCall.function.name : "";
          const toolArgs = safeJsonParse(toolCall?.function?.arguments);
          const toolResult = await executeTool(toolName, toolArgs);
          conversation.push({
            role: "tool",
            tool_call_id: toolCall?.id || toolName,
            content: JSON.stringify(toolResult),
          });
        }
      }

      setAssistantActivity("Advisory note ready");
      setAssistantMessages((messages) => [...messages, { role: "assistant", content }]);
    } catch (error) {
      setAssistantActivity("Advisor error");
      setAssistantError(error instanceof Error ? error.message : "WebLLM analysis failed.");
    } finally {
      setAssistantBusy(false);
    }
  }

  if (!isOpen) {
    return (
      <button className="assistant-launcher" type="button" onClick={openWindow}>
        Advisor
      </button>
    );
  }

  return (
    <section
      ref={windowRef}
      className={`assistant-window${isMinimized ? " assistant-window-minimized" : ""}`}
      style={{ left: `${position.x}px`, top: `${position.y}px` }}
    >
      <div className="assistant-window-header" onPointerDown={startDrag}>
        <div>
          <p className="eyebrow">WebLLM Analyst</p>
          <h2>Qwen assistant for your interests</h2>
        </div>
        <div className="assistant-window-actions">
          <button className="assistant-window-button" type="button" onClick={() => setIsMinimized((value) => !value)}>
            {isMinimized ? "Expand" : "Minimize"}
          </button>
          <button className="assistant-window-button" type="button" onClick={() => setIsOpen(false)}>
            Close
          </button>
        </div>
      </div>

      {isMinimized ? (
        <div className="assistant-window-minibar">
          <span className="assistant-chip">Browser model</span>
          <span className="assistant-chip assistant-chip-cool">{modelReady ? modelId : "Not loaded"}</span>
        </div>
      ) : (
        <div className="assistant-window-body">
          <p className="empty-copy">
            Runs in the browser with WebLLM and can inspect predictions, discovered movers, and portfolio positions before responding. The closest packaged browser model to Qwen3 9B is Qwen3 8B, which is used by default here.
          </p>

          <div className="assistant-status-row">
            <div className="assistant-status-group">
              {toolSummary.map((item) => (
                <span key={item} className="assistant-chip">{item}</span>
              ))}
              <span className="assistant-chip assistant-chip-cool">{modelReady ? modelId : "Not loaded"}</span>
            </div>
            <p className="assistant-progress">{browserSupportText || `${modelProgress} · ${assistantActivity}`}</p>
          </div>

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

          <div className="assistant-quick-actions">
            <button className="secondary-button" type="button" onClick={() => setInterestPrompt("Run a fresh market-wide same-day scan, identify the strongest long and short setups, and explain the best two opportunities.")}>Market scan</button>
            <button className="secondary-button" type="button" onClick={() => setInterestPrompt("Review my current positions, compare them with the latest signals, and tell me whether any holdings look weak or crowded.")}>Review positions</button>
            <button className="secondary-button" type="button" onClick={() => setInterestPrompt("Focus on risk management. Explain the highest-confidence setup, the main failure mode, and what would invalidate the trade.")}>Risk review</button>
          </div>

          <form className="assistant-form" onSubmit={onAnalyze}>
            <label>
              Advisor request
              <textarea value={interestPrompt} onChange={(event) => setInterestPrompt(event.target.value)} rows={5} placeholder="Example: Run a market-wide scan, compare the top setups, and explain which one has the cleanest same-day risk-reward." />
            </label>
            <button className="primary-button" type="submit" disabled={assistantBusy || modelBusy || Boolean(browserSupportText)}>
              {assistantBusy ? "Advising..." : "Run advisor"}
            </button>
          </form>

          {assistantError ? <p className="error-copy">{assistantError}</p> : null}

          <div className="assistant-chat assistant-scroll">
            {assistantMessages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`assistant-message assistant-message-${message.role}`}>
                <span>{message.role === "assistant" ? "Qwen" : "You"}</span>
                <p>{message.content}</p>
              </article>
            ))}
          </div>

          {(lastPredictions || lastGlobalMovers || lastPositions) ? (
            <div className="assistant-data-card">
              <div>
                <p className="eyebrow">Advisor Snapshot</p>
                <h3>Latest tool output</h3>
              </div>
              <div className="assistant-summary-grid">
                {lastPredictions ? (
                  <>
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
                  </>
                ) : null}
                {lastGlobalMovers ? (
                  <div className="feature-row">
                    <span>Global movers</span>
                    <strong>{lastGlobalMovers.count}</strong>
                  </div>
                ) : null}
                {lastPositions ? (
                  <div className="feature-row">
                    <span>Open positions</span>
                    <strong>{lastPositions.positions?.length ?? 0}</strong>
                  </div>
                ) : null}
              </div>

              {lastPredictions ? (
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
              ) : null}

              {lastGlobalMovers?.symbols?.length ? (
                <div className="assistant-inline-list">
                  {lastGlobalMovers.symbols.slice(0, 16).map((symbol) => (
                    <span key={symbol} className="assistant-chip">{symbol}</span>
                  ))}
                </div>
              ) : null}

              <pre className="assistant-json">{JSON.stringify({
                predictions: lastPredictions ? reducePredictionPayload(lastPredictions) : null,
                globalMovers: lastGlobalMovers,
                positions: lastPositions,
              }, null, 2)}</pre>
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}