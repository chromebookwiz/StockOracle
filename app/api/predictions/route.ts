import { NextRequest, NextResponse } from "next/server";

import { execFile } from "node:child_process";
import { promisify } from "node:util";


const execFileAsync = promisify(execFile);


type PredictionInput = {
  universe?: string[] | string;
  symbols?: string[] | string;
  discoverGlobalMovers?: boolean | string;
  globalMoversLimit?: number | string;
  benchmark?: string;
  startDate?: string;
  holdoutDays?: number | string;
  topK?: number | string;
  intradayPeriodDays?: number | string;
  intradayInterval?: string;
  signalBarIndex?: number | string | null;
  enableLiveNews?: boolean | string;
  enableLiveOptions?: boolean | string;
  enableEarningsFeatures?: boolean | string;
  startingCapital?: number | string;
  maxNotionalPerTrade?: number | string;
  executionMode?: string;
};


function parseUniverse(value: string[] | string | undefined): string[] {
  if (Array.isArray(value)) {
    return value.map((symbol) => symbol.trim().toUpperCase()).filter(Boolean);
  }
  if (typeof value === "string") {
    return value.split(/[\n,]/).map((symbol) => symbol.trim().toUpperCase()).filter(Boolean);
  }
  return [];
}


function parseBoolean(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) {
      return true;
    }
    if (["0", "false", "no", "off"].includes(normalized)) {
      return false;
    }
  }
  return fallback;
}


function parseNumber(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}


function buildPayload(input: PredictionInput): Record<string, unknown> {
  const universe = parseUniverse(input.universe ?? input.symbols);
  const discoverGlobalMovers = parseBoolean(input.discoverGlobalMovers, universe.length === 0);
  return {
    universe,
    discoverGlobalMovers,
    globalMoversLimit: parseNumber(input.globalMoversLimit, 60),
    benchmark: (input.benchmark || "SPY").trim().toUpperCase(),
    startDate: input.startDate || "2021-01-01",
    holdoutDays: parseNumber(input.holdoutDays, 45),
    topK: parseNumber(input.topK, 10),
    intradayPeriodDays: parseNumber(input.intradayPeriodDays, 45),
    intradayInterval: input.intradayInterval || "15m",
    signalBarIndex: input.signalBarIndex ?? null,
    enableLiveNews: parseBoolean(input.enableLiveNews, true),
    enableLiveOptions: parseBoolean(input.enableLiveOptions, true),
    enableEarningsFeatures: parseBoolean(input.enableEarningsFeatures, true),
    startingCapital: parseNumber(input.startingCapital, 25000),
    maxNotionalPerTrade: parseNumber(input.maxNotionalPerTrade, 5000),
    executionMode: input.executionMode || "paper",
  };
}


function buildGetPayload(request: NextRequest): Record<string, unknown> {
  const params = request.nextUrl.searchParams;
  return buildPayload({
    symbols: params.get("symbols") || params.get("universe") || undefined,
    discoverGlobalMovers: params.get("discoverGlobalMovers") || undefined,
    globalMoversLimit: params.get("globalMoversLimit") || undefined,
    benchmark: params.get("benchmark") || undefined,
    startDate: params.get("startDate") || undefined,
    holdoutDays: params.get("holdoutDays") || undefined,
    topK: params.get("topK") || undefined,
    intradayPeriodDays: params.get("intradayPeriodDays") || undefined,
    intradayInterval: params.get("intradayInterval") || undefined,
    signalBarIndex: params.get("signalBarIndex") || undefined,
    enableLiveNews: params.get("enableLiveNews") || undefined,
    enableLiveOptions: params.get("enableLiveOptions") || undefined,
    enableEarningsFeatures: params.get("enableEarningsFeatures") || undefined,
    startingCapital: params.get("startingCapital") || undefined,
    maxNotionalPerTrade: params.get("maxNotionalPerTrade") || undefined,
    executionMode: params.get("executionMode") || undefined,
  });
}


function parserFriendlyResponse(rankResponse: Record<string, unknown>, payload: Record<string, unknown>) {
  const ranking = Array.isArray(rankResponse.ranking) ? rankResponse.ranking as Array<Record<string, unknown>> : [];
  const predictions = ranking.map((row, index) => ({
    rank: index + 1,
    symbol: row.symbol,
    side: row.signal_side,
    opportunityScore: row.opportunity_score,
    predictedReturn: row.predicted_return,
    predictedReturnLower: row.predicted_return_lower,
    predictedReturnUpper: row.predicted_return_upper,
    predictedMove: row.predicted_move,
    probabilityUp: row.probability_up,
    confidence: row.confidence,
    finalScore: row.final_score,
    timingScore: row.timing_score,
    futureScore: row.future_score,
    futureConfidence: row.future_confidence,
    nextDayReturn: row.future_return_1d,
    next3DayReturn: row.future_return_3d,
    next5DayReturn: row.future_return_5d,
    directionAlignment: row.direction_alignment,
    minutesToClose: row.minutes_to_close,
    sessionReturnSoFar: row.session_return_so_far,
    newsSentiment: row.news_sentiment,
    optionsPutCallOi: row.options_put_call_oi,
  }));

  const metrics = typeof rankResponse.metrics === "object" && rankResponse.metrics !== null ? rankResponse.metrics : {};
  const timestamp = typeof (metrics as Record<string, unknown>).latest_ranking_timestamp === "string"
    ? (metrics as Record<string, string>).latest_ranking_timestamp
    : new Date().toISOString();
  const discoveredUniverse = Boolean(payload.discoverGlobalMovers) || (Array.isArray(payload.universe) && payload.universe.length === 0);

  return {
    ok: true,
    generatedAt: timestamp,
    request: payload,
    summary: {
      predictionCount: predictions.length,
      topSymbol: predictions[0]?.symbol || null,
      topSide: predictions[0]?.side || null,
      avgTopKReturn: (metrics as Record<string, unknown>).avg_top_k_return ?? null,
      directionalAccuracy: (metrics as Record<string, unknown>).directional_accuracy ?? null,
      signalBarIndex: (metrics as Record<string, unknown>).signal_bar_index ?? null,
      minutesToClose: (metrics as Record<string, unknown>).median_minutes_to_close ?? null,
      discoveredUniverse,
    },
    predictions,
    metrics,
    ranking: rankResponse.ranking ?? [],
    featureImportance: rankResponse.featureImportance ?? [],
    holdoutPredictions: rankResponse.holdoutPredictions ?? [],
    backtestCurve: rankResponse.backtestCurve ?? [],
    executionPlan: rankResponse.executionPlan ?? [],
    executionConfirmation: rankResponse.executionConfirmation ?? null,
  };
}


function localPythonExecutable(): string {
  if (process.env.STOCKORACLE_PYTHON_BIN) {
    return process.env.STOCKORACLE_PYTHON_BIN;
  }
  return process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";
}


async function runLocalPythonRank(payload: Record<string, unknown>) {
  const script = [
    "import json, math, sys",
    "from pathlib import Path",
    "root = Path.cwd()",
    "src = root / 'src'",
    "if str(src) not in sys.path: sys.path.insert(0, str(src))",
    "from api.rank import RankingRequest, rank",
    "def clean(value):",
    "    if isinstance(value, float) and not math.isfinite(value): return None",
    "    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}",
    "    if isinstance(value, list): return [clean(v) for v in value]",
    "    return value",
    "payload = json.loads(sys.argv[1])",
    "result = rank(RankingRequest(**payload))",
    "print(json.dumps(clean(result), default=str))",
  ].join("\n");

  const { stdout } = await execFileAsync(localPythonExecutable(), ["-c", script, JSON.stringify(payload)], {
    cwd: process.cwd(),
    maxBuffer: 32 * 1024 * 1024,
  });

  return JSON.parse(stdout) as Record<string, unknown>;
}


async function fetchRankResponse(request: NextRequest, payload: Record<string, unknown>) {
  const response = await fetch(`${request.nextUrl.origin}/api/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });

  if (response.status !== 404) {
    const json = await response.json();
    return { status: response.status, json };
  }

  if (process.env.NODE_ENV === "production") {
    const json = await response.json();
    return { status: response.status, json };
  }

  const json = await runLocalPythonRank(payload);
  return { status: 200, json };
}


async function fetchPredictions(request: NextRequest, payload: Record<string, unknown>) {
  const { status, json } = await fetchRankResponse(request, payload);
  if (status >= 400) {
    const errorResponse = NextResponse.json({ ok: false, detail: json.detail || "Prediction request failed." }, { status });
    errorResponse.headers.set("Cache-Control", "no-store");
    return errorResponse;
  }

  const parserResponse = NextResponse.json(parserFriendlyResponse(json as Record<string, unknown>, payload));
  parserResponse.headers.set("Cache-Control", "no-store");
  return parserResponse;
}


export async function GET(request: NextRequest) {
  const payload = buildGetPayload(request);
  return fetchPredictions(request, payload);
}


export async function POST(request: NextRequest) {
  const body = await request.json() as PredictionInput;
  const payload = buildPayload(body);
  return fetchPredictions(request, payload);
}