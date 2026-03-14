"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type RankingRow = {
  symbol: string;
  signal_side: string;
  opportunity_score: number;
  predicted_return: number;
  predicted_return_lower: number;
  predicted_return_upper: number;
  predicted_move: number;
  probability_up: number;
  confidence: number;
  model_score: number;
  overlay_score: number;
  final_score: number;
  minutes_to_close?: number | null;
  session_return_so_far?: number | null;
  news_sentiment?: number | null;
  options_put_call_oi?: number | null;
};

type FeatureRow = {
  feature: string;
  importance: number;
};

type BacktestRow = {
  date: string;
  equity: number;
  net_return: number;
};

type ApiResponse = {
  ranking: RankingRow[];
  metrics: Record<string, number | string>;
  featureImportance: FeatureRow[];
  backtestCurve: BacktestRow[];
  executionPlan: Array<{
    symbol: string;
    side: string;
    quantity: number;
    notional: number;
    reference_price: number;
    predicted_return: number;
    confidence: number;
    final_score: number;
  }>;
  executionConfirmation: {
    required: boolean;
    configured: boolean;
    mode: string;
    topK: number;
    confirmationToken: string | null;
  };
};

type SessionState = {
  configured: boolean;
  authenticated: boolean;
  username: string | null;
};

const defaultUniverse = [
  "AAPL",
  "MSFT",
  "NVDA",
  "AMZN",
  "META",
  "GOOGL",
  "TSLA",
  "AMD",
  "AVGO",
  "PLTR",
  "CRWD",
  "ANET",
  "UBER",
  "JPM",
  "LLY",
  "XOM",
].join(", ");

function percent(value: number | string | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function number(value: number | string | undefined, digits = 2): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function sparklinePath(points: BacktestRow[]): string {
  if (points.length === 0) {
    return "";
  }
  const values = points.map((point) => point.equity);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  return points
    .map((point, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * 100;
      const y = 100 - ((point.equity - min) / range) * 100;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export default function Home() {
  const [universe, setUniverse] = useState(defaultUniverse);
  const [benchmark, setBenchmark] = useState("SPY");
  const [startDate, setStartDate] = useState("2021-01-01");
  const [holdoutDays, setHoldoutDays] = useState(45);
  const [topK, setTopK] = useState(10);
  const [intradayPeriodDays, setIntradayPeriodDays] = useState(45);
  const [intradayInterval, setIntradayInterval] = useState("15m");
  const [liveNews, setLiveNews] = useState(true);
  const [liveOptions, setLiveOptions] = useState(true);
  const [earningsFeatures, setEarningsFeatures] = useState(true);
  const [executionMode, setExecutionMode] = useState("paper");
  const [confirmExecution, setConfirmExecution] = useState(false);
  const [startingCapital, setStartingCapital] = useState(25000);
  const [maxNotionalPerTrade, setMaxNotionalPerTrade] = useState(5000);
  const [positions, setPositions] = useState<Array<{ symbol: string; quantity: number; avgPrice: number }>>([]);
  const [session, setSession] = useState<SessionState>({ configured: false, authenticated: false, username: null });
  const [loginUsername, setLoginUsername] = useState("operator");
  const [loginPassword, setLoginPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(true);
  const [loading, setLoading] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ApiResponse | null>(null);

  const sparkline = useMemo(() => sparklinePath(data?.backtestCurve ?? []), [data]);

  async function refreshSession() {
    const response = await fetch("/api/auth/session", { cache: "no-store" });
    const json = (await response.json()) as SessionState;
    setSession(json);
    setAuthLoading(false);
    return json;
  }

  async function refreshPositions(mode: string) {
    const response = await fetch(`/api/trade/positions?mode=${encodeURIComponent(mode)}`, { cache: "no-store" });
    if (!response.ok) {
      setPositions([]);
      return;
    }
    const json = (await response.json()) as { positions?: Array<{ symbol: string; quantity: number; avgPrice: number }> };
    setPositions(json.positions ?? []);
  }

  useEffect(() => {
    void refreshSession().then((current) => {
      if (current.authenticated) {
        void refreshPositions(executionMode);
      }
    });
  }, []);

  useEffect(() => {
    if (session.authenticated) {
      void refreshPositions(executionMode);
    }
  }, [executionMode, session.authenticated]);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    const payload = {
      universe: universe
        .split(/[,\n]/)
        .map((symbol) => symbol.trim().toUpperCase())
        .filter(Boolean),
      benchmark,
      startDate,
      holdoutDays,
      topK,
      intradayPeriodDays,
      enableLiveNews: liveNews,
      intradayInterval,
      enableLiveOptions: liveOptions,
      enableEarningsFeatures: earningsFeatures,
      executionMode,
      startingCapital,
      maxNotionalPerTrade,
    };

    try {
      const response = await fetch("/api/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = (await response.json()) as ApiResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in json && json.detail ? json.detail : "Ranking request failed.");
      }
      setData(json as ApiResponse);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unknown error");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  async function executePlan() {
    setExecuting(true);
    setError(null);

    try {
      const response = await fetch("/api/trade/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          universe: universe
            .split(/[,\n]/)
            .map((symbol) => symbol.trim().toUpperCase())
            .filter(Boolean),
          benchmark,
          startDate,
          holdoutDays,
          topK,
          intradayPeriodDays,
          intradayInterval,
          enableLiveNews: liveNews,
          enableLiveOptions: liveOptions,
          enableEarningsFeatures: earningsFeatures,
          executionMode,
          startingCapital,
          maxNotionalPerTrade,
          confirmExecution,
          confirmationToken: data?.executionConfirmation?.confirmationToken,
        }),
      });
      const json = (await response.json()) as { positions?: Array<{ symbol: string; quantity: number; avgPrice: number }>; detail?: string };
      if (!response.ok) {
        throw new Error(json.detail || "Execution request failed.");
      }
      setPositions(json.positions ?? []);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unknown execution error");
    } finally {
      setExecuting(false);
    }
  }

  async function login() {
    setError(null);
    setAuthLoading(true);
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: loginUsername, password: loginPassword }),
      });
      const json = (await response.json()) as { detail?: string };
      if (!response.ok) {
        throw new Error(json.detail || "Login failed.");
      }
      const current = await refreshSession();
      if (current.authenticated) {
        await refreshPositions(executionMode);
      }
      setLoginPassword("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unknown auth error");
      setAuthLoading(false);
    }
  }

  async function logout() {
    setError(null);
    await fetch("/api/auth/logout", { method: "POST" });
    setSession({ configured: session.configured, authenticated: false, username: null });
    setPositions([]);
    setConfirmExecution(false);
  }

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Same-Day Close Targeting</p>
          <h1>StockOracle</h1>
          <p className="hero-copy">
            A same-day profit tool that ranks stocks by predicted return from the current intraday bar into the close,
            blending live bar structure, prior-day context, news sentiment, options positioning, earnings timing, and a
            time-aligned walk-forward backtest.
          </p>
        </div>
        <div className="hero-metrics">
          <span>Vercel-ready frontend</span>
          <span>Python serverless API</span>
          <span>Walk-forward evaluation</span>
        </div>
      </section>

      <section className="grid-layout">
        <form className="control-panel" onSubmit={onSubmit}>
          <label>
            Universe
            <textarea value={universe} onChange={(event) => setUniverse(event.target.value)} rows={8} />
          </label>

          <div className="two-up">
            <label>
              Benchmark
              <input value={benchmark} onChange={(event) => setBenchmark(event.target.value.toUpperCase())} />
            </label>
            <label>
              Start Date
              <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
            </label>
          </div>

          <div className="two-up">
            <label>
              Holdout Days
              <input type="range" min="15" max="90" value={holdoutDays} onChange={(event) => setHoldoutDays(Number(event.target.value))} />
              <span>{holdoutDays}</span>
            </label>
            <label>
              Intraday Interval
              <select value={intradayInterval} onChange={(event) => setIntradayInterval(event.target.value)}>
                <option value="5m">5m</option>
                <option value="15m">15m</option>
                <option value="30m">30m</option>
                <option value="60m">60m</option>
              </select>
            </label>
          </div>

          <div className="two-up">
            <label>
              Top Picks
              <input type="range" min="3" max="20" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
              <span>{topK}</span>
            </label>
            <label>
              Intraday Lookback
              <input
                type="range"
                min="10"
                max="60"
                value={intradayPeriodDays}
                onChange={(event) => setIntradayPeriodDays(Number(event.target.value))}
              />
              <span>{intradayPeriodDays}d</span>
            </label>
          </div>

          <div className="toggle-grid">
            <label>
              <input type="checkbox" checked={liveNews} onChange={(event) => setLiveNews(event.target.checked)} />
              Live news sentiment
            </label>
            <label>
              <input type="checkbox" checked={liveOptions} onChange={(event) => setLiveOptions(event.target.checked)} />
              Live options flow
            </label>
            <label>
              <input type="checkbox" checked={earningsFeatures} onChange={(event) => setEarningsFeatures(event.target.checked)} />
              Earnings timing
            </label>
          </div>

          <div className="two-up">
            <label>
              Execution Mode
              <select value={executionMode} onChange={(event) => setExecutionMode(event.target.value)}>
                <option value="paper">paper</option>
                <option value="alpaca">alpaca</option>
              </select>
            </label>
            <label>
              Starting Capital
              <input type="number" value={startingCapital} onChange={(event) => setStartingCapital(Number(event.target.value))} />
            </label>
            <label>
              Max Notional / Trade
              <input type="number" value={maxNotionalPerTrade} onChange={(event) => setMaxNotionalPerTrade(Number(event.target.value))} />
            </label>
          </div>

          <div className="auth-card">
            <p className="eyebrow">Operator Auth</p>
            {authLoading ? <p className="empty-copy">Checking session...</p> : null}
            {!authLoading && !session.configured ? <p className="empty-copy">Set STOCKORACLE_OPERATOR_PASSWORD and STOCKORACLE_SESSION_SECRET to enable operator login.</p> : null}
            {!authLoading && session.configured && !session.authenticated ? (
              <>
                <label>
                  Username
                  <input value={loginUsername} onChange={(event) => setLoginUsername(event.target.value)} />
                </label>
                <label>
                  Password
                  <input type="password" value={loginPassword} onChange={(event) => setLoginPassword(event.target.value)} />
                </label>
                <button className="secondary-button" type="button" onClick={login}>
                  Sign in operator
                </button>
              </>
            ) : null}
            {!authLoading && session.authenticated ? (
              <>
                <p className="empty-copy">Signed in as {session.username}.</p>
                <button className="secondary-button" type="button" onClick={logout}>
                  Sign out
                </button>
              </>
            ) : null}
          </div>

          <label className="checkbox-row">
            <input type="checkbox" checked={confirmExecution} onChange={(event) => setConfirmExecution(event.target.checked)} />
            I confirm the current execution plan and want to submit these orders.
          </label>

          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? "Running model..." : "Generate movers"}
          </button>
          <button
            className="secondary-button"
            type="button"
            disabled={!data || executing || !confirmExecution || !session.authenticated || (Boolean(data?.executionConfirmation.required) && !Boolean(data?.executionConfirmation.configured))}
            onClick={executePlan}
          >
            {executing ? "Submitting orders..." : `Submit top picks to ${executionMode}`}
          </button>
          {data?.executionConfirmation.required && !data?.executionConfirmation.configured ? (
            <p className="empty-copy">Trading is locked until STOCKORACLE_CONFIRMATION_SECRET is configured on the server.</p>
          ) : null}
          {error ? <p className="error-copy">{error}</p> : null}
        </form>

        <section className="results-panel">
          <div className="metric-grid">
            <article>
              <span>Avg top-k strategy return</span>
              <strong>{percent((data?.metrics.avg_top_k_return as number | undefined) ?? undefined)}</strong>
            </article>
            <article>
              <span>Hit rate</span>
              <strong>{percent((data?.metrics.top_k_hit_rate as number | undefined) ?? undefined)}</strong>
            </article>
            <article>
              <span>Minutes to close</span>
              <strong>{number((data?.metrics.median_minutes_to_close as number | undefined) ?? undefined, 0)}</strong>
            </article>
            <article>
              <span>Backtest Sharpe</span>
              <strong>{number((data?.metrics.backtest_sharpe as number | undefined) ?? undefined, 2)}</strong>
            </article>
            <article>
              <span>Directional accuracy</span>
              <strong>{percent((data?.metrics.directional_accuracy as number | undefined) ?? undefined)}</strong>
            </article>
          </div>

          <div className="chart-card">
            <div>
              <p className="eyebrow">Backtest Curve</p>
              <h2>Cost-aware equity path</h2>
            </div>
            {data?.backtestCurve?.length ? (
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="sparkline">
                <path d={sparkline} vectorEffect="non-scaling-stroke" />
              </svg>
            ) : (
              <p className="empty-copy">Run the model to render the holdout equity curve.</p>
            )}
          </div>

          <div className="table-card">
            <div>
              <p className="eyebrow">Ranked Movers</p>
              <h2>Latest session</h2>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Opportunity</th>
                    <th>To-Close Return</th>
                    <th>Return Range</th>
                    <th>Prob Up</th>
                    <th>Confidence</th>
                    <th>Model</th>
                    <th>Overlay</th>
                    <th>Final</th>
                    <th>Mins Left</th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.ranking ?? []).map((row) => (
                    <tr key={row.symbol}>
                      <td>{row.symbol}</td>
                      <td>{row.signal_side}</td>
                      <td>{number(row.opportunity_score)}</td>
                      <td>{percent(row.predicted_return)}</td>
                      <td>{`${percent(row.predicted_return_lower)} to ${percent(row.predicted_return_upper)}`}</td>
                      <td>{percent(row.probability_up)}</td>
                      <td>{percent(row.confidence)}</td>
                      <td>{number(row.model_score)}</td>
                      <td>{number(row.overlay_score)}</td>
                      <td>{number(row.final_score)}</td>
                      <td>{number(row.minutes_to_close ?? undefined, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="feature-card">
            <div>
              <p className="eyebrow">Driver Map</p>
              <h2>Top feature importance</h2>
            </div>
            <div className="feature-list">
              {(data?.featureImportance ?? []).slice(0, 10).map((feature) => (
                <div key={feature.feature} className="feature-row">
                  <span>{feature.feature}</span>
                  <strong>{number(feature.importance, 3)}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="table-card">
            <div>
              <p className="eyebrow">Execution</p>
              <h2>Paper trade plan</h2>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Notional</th>
                    <th>Ref Price</th>
                    <th>Pred Return</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.executionPlan ?? []).map((row) => (
                    <tr key={`${row.symbol}-${row.side}`}>
                      <td>{row.symbol}</td>
                      <td>{row.side}</td>
                      <td>{number(row.quantity, 0)}</td>
                      <td>${number(row.notional, 2)}</td>
                      <td>${number(row.reference_price, 2)}</td>
                      <td>{percent(row.predicted_return)}</td>
                      <td>{percent(row.confidence)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {positions.length ? (
              <div className="positions-block">
                <p className="eyebrow">Paper Broker Positions</p>
                {positions.map((position) => (
                  <div key={position.symbol} className="feature-row">
                    <span>{position.symbol}</span>
                    <strong>{`${position.quantity} @ $${number(position.avgPrice, 2)}`}</strong>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </section>
      </section>
    </main>
  );
}