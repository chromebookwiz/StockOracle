"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type RankingRow = {
  symbol: string;
  signal_side: string;
  opportunity_score: number;
  predicted_return: number;
  predicted_return_lower: number;
  predicted_return_upper: number;
  prediction_interval?: number | null;
  predicted_move: number;
  probability_up: number;
  confidence: number;
  model_score: number;
  model_disagreement?: number | null;
  overlay_score: number;
  final_score: number;
  expected_edge?: number | null;
  risk_budget?: number | null;
  reward_risk_ratio?: number | null;
  setup_quality?: number | null;
  minutes_to_close?: number | null;
  session_return_so_far?: number | null;
  news_sentiment?: number | null;
  options_put_call_oi?: number | null;
  timing_score?: number | null;
  future_score?: number | null;
  future_confidence?: number | null;
  future_return_blend?: number | null;
  future_return_1d?: number | null;
  future_return_3d?: number | null;
  future_return_5d?: number | null;
  direction_alignment?: number | null;
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

type ExecutionPlanRow = {
  symbol: string;
  side: string;
  quantity: number;
  notional: number;
  reference_price: number;
  predicted_return: number;
  confidence: number;
  final_score: number;
};

type ApiResponse = {
  ok?: boolean;
  ranking: RankingRow[];
  metrics: Record<string, number | string>;
  featureImportance: FeatureRow[];
  backtestCurve: BacktestRow[];
  executionPlan: ExecutionPlanRow[];
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

type PositionRow = {
  symbol: string;
  quantity: number;
  avgPrice: number;
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

function metricNumber(metrics: Record<string, number | string> | undefined, key: string): number | undefined {
  const value = metrics?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function percent(value: number | string | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function number(value: number | string | null | undefined, digits = 2): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function currency(value: number | string | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function alignmentLabel(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "flat";
  }
  if (value > 0) {
    return "aligned";
  }
  if (value < 0) {
    return "conflict";
  }
  return "flat";
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
  const [discoverGlobalMovers, setDiscoverGlobalMovers] = useState(false);
  const [globalMoversLimit, setGlobalMoversLimit] = useState(60);
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
  const [positions, setPositions] = useState<PositionRow[]>([]);
  const [session, setSession] = useState<SessionState>({ configured: false, authenticated: false, username: null });
  const [loginUsername, setLoginUsername] = useState("operator");
  const [loginPassword, setLoginPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(true);
  const [loading, setLoading] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ApiResponse | null>(null);

  const sparkline = useMemo(() => sparklinePath(data?.backtestCurve ?? []), [data]);
  const universeList = useMemo(
    () =>
      universe
        .split(/[\n,]/)
        .map((symbol) => symbol.trim().toUpperCase())
        .filter(Boolean),
    [universe],
  );
  const topSetups = useMemo(() => (data?.ranking ?? []).slice(0, 4), [data]);
  const featureLeaders = useMemo(() => (data?.featureImportance ?? []).slice(0, 8), [data]);

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
    const json = (await response.json()) as { positions?: PositionRow[] };
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

  async function runPredictions(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    const resolvedUniverse = discoverGlobalMovers
      ? []
      : universe
          .split(/[\n,]/)
          .map((symbol) => symbol.trim().toUpperCase())
          .filter(Boolean);

    try {
      const response = await fetch("/api/predictions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          universe: resolvedUniverse,
          discoverGlobalMovers,
          globalMoversLimit,
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
        }),
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
          universe: discoverGlobalMovers
            ? []
            : universe
                .split(/[\n,]/)
                .map((symbol) => symbol.trim().toUpperCase())
                .filter(Boolean),
          discoverGlobalMovers,
          globalMoversLimit,
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
      const json = (await response.json()) as { positions?: PositionRow[]; detail?: string };
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

  const summaryCards = [
    { label: "Holdout edge", value: percent(metricNumber(data?.metrics, "avg_top_k_return")) },
    { label: "Hit rate", value: percent(metricNumber(data?.metrics, "top_k_hit_rate")) },
    { label: "Sharpe", value: number(metricNumber(data?.metrics, "backtest_sharpe"), 2) },
    { label: "Top edge", value: percent(metricNumber(data?.metrics, "top_setup_expected_edge")) },
    { label: "Top R/R", value: number(metricNumber(data?.metrics, "top_setup_reward_risk"), 2) },
    { label: "Alignment", value: percent(metricNumber(data?.metrics, "signal_alignment_rate")) },
    { label: "Breadth", value: percent(metricNumber(data?.metrics, "breadth_balance")) },
    { label: "Minutes left", value: number(metricNumber(data?.metrics, "median_minutes_to_close"), 0) },
  ];

  return (
    <main className="page-shell">
      <header className="topbar panel">
        <div>
          <p className="eyebrow">StockOracle</p>
          <h1>Signal Desk</h1>
        </div>
        <div className="chip-row">
          <span className="chip">{benchmark}</span>
          <span className="chip">{intradayInterval}</span>
          <span className="chip">{discoverGlobalMovers ? `Global ${globalMoversLimit}` : `${universeList.length} names`}</span>
          <span className="chip">{session.authenticated ? `Operator ${session.username}` : "Research"}</span>
        </div>
      </header>

      <section className="workspace-grid">
        <form className="control-rail panel" onSubmit={runPredictions}>
          <div className="section-head">
            <div>
              <p className="eyebrow">Run Settings</p>
              <h2>Inputs</h2>
            </div>
            <span className="badge">{discoverGlobalMovers ? `Pool ${globalMoversLimit}` : `${universeList.length} tickers`}</span>
          </div>

          <label className="toggle-row">
            <span>Market-wide scan</span>
            <input type="checkbox" checked={discoverGlobalMovers} onChange={(event) => setDiscoverGlobalMovers(event.target.checked)} />
          </label>

          <label className="field">
            <span>Symbols</span>
            <textarea value={universe} onChange={(event) => setUniverse(event.target.value)} rows={7} disabled={discoverGlobalMovers} />
          </label>

          <section className="field-section">
            <div className="section-label">Model window</div>
            <div className="field-grid two-up">
              <label className="field">
                <span>Benchmark</span>
                <input value={benchmark} onChange={(event) => setBenchmark(event.target.value.toUpperCase())} />
              </label>
              <label className="field">
                <span>Start date</span>
                <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
              </label>
              <label className="field range-field">
                <span>Top picks</span>
                <input type="range" min="3" max="20" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
                <strong>{topK}</strong>
              </label>
              <label className="field range-field">
                <span>Holdout days</span>
                <input type="range" min="15" max="90" value={holdoutDays} onChange={(event) => setHoldoutDays(Number(event.target.value))} />
                <strong>{holdoutDays}</strong>
              </label>
            </div>
          </section>

          <section className="field-section">
            <div className="section-label">Timing</div>
            <div className="field-grid two-up">
              <label className="field range-field">
                <span>Global mover pool</span>
                <input type="range" min="20" max="120" step="10" value={globalMoversLimit} onChange={(event) => setGlobalMoversLimit(Number(event.target.value))} disabled={!discoverGlobalMovers} />
                <strong>{globalMoversLimit}</strong>
              </label>
              <label className="field">
                <span>Intraday interval</span>
                <select value={intradayInterval} onChange={(event) => setIntradayInterval(event.target.value)}>
                  <option value="5m">5m</option>
                  <option value="15m">15m</option>
                  <option value="30m">30m</option>
                  <option value="60m">60m</option>
                </select>
              </label>
              <label className="field range-field field-span-full">
                <span>Intraday lookback</span>
                <input type="range" min="10" max="60" value={intradayPeriodDays} onChange={(event) => setIntradayPeriodDays(Number(event.target.value))} />
                <strong>{intradayPeriodDays}d</strong>
              </label>
            </div>
          </section>

          <section className="field-section">
            <div className="section-label">Overlays</div>
            <div className="switch-grid">
              <label className="toggle-card">
                <span>News</span>
                <input type="checkbox" checked={liveNews} onChange={(event) => setLiveNews(event.target.checked)} />
              </label>
              <label className="toggle-card">
                <span>Options</span>
                <input type="checkbox" checked={liveOptions} onChange={(event) => setLiveOptions(event.target.checked)} />
              </label>
              <label className="toggle-card">
                <span>Earnings</span>
                <input type="checkbox" checked={earningsFeatures} onChange={(event) => setEarningsFeatures(event.target.checked)} />
              </label>
            </div>
          </section>

          <section className="field-section">
            <div className="section-label">Execution</div>
            <div className="field-grid two-up">
              <label className="field">
                <span>Mode</span>
                <select value={executionMode} onChange={(event) => setExecutionMode(event.target.value)}>
                  <option value="paper">paper</option>
                  <option value="alpaca">alpaca</option>
                </select>
              </label>
              <label className="field">
                <span>Capital</span>
                <input type="number" value={startingCapital} onChange={(event) => setStartingCapital(Number(event.target.value))} />
              </label>
              <label className="field field-span-full">
                <span>Max notional per trade</span>
                <input type="number" value={maxNotionalPerTrade} onChange={(event) => setMaxNotionalPerTrade(Number(event.target.value))} />
              </label>
            </div>
          </section>

          <section className="auth-panel">
            <div className="section-head">
              <div>
                <p className="eyebrow">Operator</p>
                <h3>Session</h3>
              </div>
              <span className="badge">{session.authenticated ? "Open" : "Locked"}</span>
            </div>
            {authLoading ? <p className="muted-copy">Checking session</p> : null}
            {!authLoading && !session.configured ? <p className="muted-copy">Auth secrets not configured</p> : null}
            {!authLoading && session.configured && !session.authenticated ? (
              <div className="field-grid">
                <label className="field">
                  <span>Username</span>
                  <input value={loginUsername} onChange={(event) => setLoginUsername(event.target.value)} />
                </label>
                <label className="field">
                  <span>Password</span>
                  <input type="password" value={loginPassword} onChange={(event) => setLoginPassword(event.target.value)} />
                </label>
                <button className="secondary-button" type="button" onClick={login}>
                  Sign in
                </button>
              </div>
            ) : null}
            {!authLoading && session.authenticated ? (
              <div className="session-actions">
                <strong>{session.username}</strong>
                <button className="secondary-button" type="button" onClick={logout}>
                  Sign out
                </button>
              </div>
            ) : null}
          </section>

          <label className="toggle-row confirm-row">
            <span>Confirm current orders</span>
            <input type="checkbox" checked={confirmExecution} onChange={(event) => setConfirmExecution(event.target.checked)} />
          </label>

          <div className="action-row">
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Running" : "Run scan"}
            </button>
            <button
              className="secondary-button"
              type="button"
              disabled={!data || executing || !confirmExecution || !session.authenticated || (Boolean(data?.executionConfirmation.required) && !Boolean(data?.executionConfirmation.configured))}
              onClick={executePlan}
            >
              {executing ? "Submitting" : "Stage orders"}
            </button>
          </div>

          {data?.executionConfirmation.required && !data?.executionConfirmation.configured ? <p className="muted-copy">Execution confirmation secret missing</p> : null}
          {error ? <p className="error-copy">{error}</p> : null}
        </form>

        <section className="dashboard">
          <section className="summary-grid">
            {summaryCards.map((card) => (
              <article key={card.label} className="metric-card panel">
                <span>{card.label}</span>
                <strong>{card.value}</strong>
              </article>
            ))}
          </section>

          <section className="signal-grid">
            {topSetups.length ? (
              topSetups.map((row, index) => (
                <article key={row.symbol} className="signal-card panel">
                  <div className="section-head">
                    <div>
                      <p className="eyebrow">Rank {index + 1}</p>
                      <h3>{row.symbol}</h3>
                    </div>
                    <span className={`side-pill ${row.signal_side}`}>{row.signal_side}</span>
                  </div>
                  <div className="signal-metrics">
                    <div>
                      <span>Expected edge</span>
                      <strong>{percent(row.expected_edge)}</strong>
                    </div>
                    <div>
                      <span>Risk budget</span>
                      <strong>{percent(row.risk_budget)}</strong>
                    </div>
                    <div>
                      <span>R/R</span>
                      <strong>{number(row.reward_risk_ratio, 2)}</strong>
                    </div>
                    <div>
                      <span>Confidence</span>
                      <strong>{percent(row.confidence)}</strong>
                    </div>
                    <div>
                      <span>Next 3d</span>
                      <strong>{percent(row.future_return_3d)}</strong>
                    </div>
                    <div>
                      <span>Alignment</span>
                      <strong>{alignmentLabel(row.direction_alignment)}</strong>
                    </div>
                  </div>
                </article>
              ))
            ) : (
              <article className="signal-card panel signal-empty">
                <p className="eyebrow">Signals</p>
                <h3>No ranking</h3>
                <p className="muted-copy">Run scan</p>
              </article>
            )}
          </section>

          <section className="detail-grid">
            <article className="panel chart-panel">
              <div className="section-head">
                <div>
                  <p className="eyebrow">Backtest</p>
                  <h2>Equity curve</h2>
                </div>
                <span className="badge">{currency(startingCapital)}</span>
              </div>
              {data?.backtestCurve?.length ? (
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="sparkline">
                  <path d={sparkline} vectorEffect="non-scaling-stroke" />
                </svg>
              ) : (
                <p className="muted-copy">No curve</p>
              )}
            </article>

            <article className="panel feature-panel">
              <div className="section-head">
                <div>
                  <p className="eyebrow">Drivers</p>
                  <h2>Top features</h2>
                </div>
                <span className="badge">{featureLeaders.length}</span>
              </div>
              <div className="stack-list">
                {featureLeaders.length ? (
                  featureLeaders.map((row) => (
                    <div key={row.feature} className="list-row">
                      <span>{row.feature}</span>
                      <strong>{number(row.importance, 3)}</strong>
                    </div>
                  ))
                ) : (
                  <p className="muted-copy">No feature importances</p>
                )}
              </div>
            </article>

            <article className="panel execution-panel">
              <div className="section-head">
                <div>
                  <p className="eyebrow">Execution</p>
                  <h2>Order staging</h2>
                </div>
                <span className="badge">{executionMode}</span>
              </div>
              <div className="stack-list">
                {(data?.executionPlan ?? []).slice(0, 6).map((row) => (
                  <div key={`${row.symbol}-${row.side}`} className="list-row">
                    <div>
                      <strong>{row.symbol}</strong>
                      <span className="row-subtitle">{row.side}</span>
                    </div>
                    <div className="row-right">
                      <strong>{currency(row.notional)}</strong>
                      <span className="row-subtitle">{percent(row.predicted_return)}</span>
                    </div>
                  </div>
                ))}
                {!data?.executionPlan?.length ? <p className="muted-copy">No orders</p> : null}
              </div>
              <div className="positions-block">
                <div className="section-label">Open positions</div>
                {positions.length ? positions.map((position) => (
                  <div key={position.symbol} className="list-row compact-row">
                    <span>{position.symbol}</span>
                    <strong>{`${position.quantity} @ $${number(position.avgPrice, 2)}`}</strong>
                  </div>
                )) : <p className="muted-copy">No positions</p>}
              </div>
            </article>
          </section>

          <article className="panel table-panel">
            <div className="section-head">
              <div>
                <p className="eyebrow">Ranking</p>
                <h2>Decision board</h2>
              </div>
              <span className="badge">{data?.ranking.length ?? 0} rows</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Symbol</th>
                    <th>Bias</th>
                    <th>Edge</th>
                    <th>Risk</th>
                    <th>R/R</th>
                    <th>Conf</th>
                    <th>Interval</th>
                    <th>Disagree</th>
                    <th>1d</th>
                    <th>3d</th>
                    <th>5d</th>
                    <th>Align</th>
                    <th>Mins</th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.ranking ?? []).map((row, index) => (
                    <tr key={row.symbol}>
                      <td>{index + 1}</td>
                      <td>{row.symbol}</td>
                      <td><span className={`side-pill ${row.signal_side}`}>{row.signal_side}</span></td>
                      <td>{percent(row.expected_edge)}</td>
                      <td>{percent(row.risk_budget)}</td>
                      <td>{number(row.reward_risk_ratio, 2)}</td>
                      <td>{percent(row.confidence)}</td>
                      <td>{percent(row.prediction_interval)}</td>
                      <td>{number(row.model_disagreement, 4)}</td>
                      <td>{percent(row.future_return_1d)}</td>
                      <td>{percent(row.future_return_3d)}</td>
                      <td>{percent(row.future_return_5d)}</td>
                      <td>{alignmentLabel(row.direction_alignment)}</td>
                      <td>{number(row.minutes_to_close, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}
