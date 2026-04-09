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
  timing_score?: number | null;
  future_score?: number | null;
  future_confidence?: number | null;
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

type ApiResponse = {
  ok?: boolean;
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

function currency(value: number | string | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
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
  const universeList = useMemo(
    () =>
      universe
        .split(/[,\n]/)
        .map((symbol) => symbol.trim().toUpperCase())
        .filter(Boolean),
    [universe],
  );
  const topSetups = useMemo(() => (data?.ranking ?? []).slice(0, 3), [data]);

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

  async function runPredictions(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    const payload = {
      universe: discoverGlobalMovers
        ? []
        : universe
            .split(/[,\n]/)
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
    };

    try {
      const response = await fetch("/api/predictions", {
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
          universe: discoverGlobalMovers
            ? []
            : universe
                .split(/[,\n]/)
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
      <section className="masthead">
        <div className="brand-lockup">
          <p className="eyebrow">StockOracle</p>
          <h1>Signal Desk</h1>
        </div>
        <div className="masthead-meta">
          <span>{benchmark}</span>
          <span>{intradayInterval}</span>
          <span>{discoverGlobalMovers ? `Global ${globalMoversLimit}` : `${universeList.length} names`}</span>
          <span>{session.authenticated ? `Operator ${session.username}` : "Research mode"}</span>
        </div>
      </section>

      <section className="grid-layout">
        <form className="control-panel" onSubmit={runPredictions}>
          <div className="panel-header">
            <div>
              <p className="eyebrow">Scan Setup</p>
              <h2>Universe</h2>
            </div>
            <div className="panel-stat">{discoverGlobalMovers ? `Live scan ${globalMoversLimit}` : `${universeList.length} tickers`}</div>
          </div>

          <div className="toggle-grid compact-grid">
            <label>
              <input type="checkbox" checked={discoverGlobalMovers} onChange={(event) => setDiscoverGlobalMovers(event.target.checked)} />
              Market-wide scan
            </label>
          </div>

          <label>
            Symbols
            <textarea value={universe} onChange={(event) => setUniverse(event.target.value)} rows={8} disabled={discoverGlobalMovers} />
          </label>

          <div className="control-section">
            <div className="section-title">Model Window</div>
            <div className="two-up">
              <label>
                Benchmark
                <input value={benchmark} onChange={(event) => setBenchmark(event.target.value.toUpperCase())} />
              </label>
              <label>
                Start Date
                <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
              </label>
              <label>
                Top Picks
                <input type="range" min="3" max="20" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
                <span>{topK}</span>
              </label>
              <label>
                Holdout Days
                <input type="range" min="15" max="90" value={holdoutDays} onChange={(event) => setHoldoutDays(Number(event.target.value))} />
                <span>{holdoutDays}</span>
              </label>
            </div>
          </div>

          <div className="control-section">
            <div className="section-title">Timing Inputs</div>
            <div className="two-up">
              <label>
                Global mover pool
                <input type="range" min="20" max="120" step="10" value={globalMoversLimit} onChange={(event) => setGlobalMoversLimit(Number(event.target.value))} disabled={!discoverGlobalMovers} />
                <span>{globalMoversLimit}</span>
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
              <label>
                Intraday Lookback
                <input type="range" min="10" max="60" value={intradayPeriodDays} onChange={(event) => setIntradayPeriodDays(Number(event.target.value))} />
                <span>{intradayPeriodDays}d</span>
              </label>
            </div>
          </div>

          <div className="control-section">
            <div className="section-title">Overlays</div>
            <div className="toggle-grid compact-grid">
              <label>
                <input type="checkbox" checked={liveNews} onChange={(event) => setLiveNews(event.target.checked)} />
                News
              </label>
              <label>
                <input type="checkbox" checked={liveOptions} onChange={(event) => setLiveOptions(event.target.checked)} />
                Options
              </label>
              <label>
                <input type="checkbox" checked={earningsFeatures} onChange={(event) => setEarningsFeatures(event.target.checked)} />
                Earnings
              </label>
            </div>
          </div>

          <div className="control-section">
            <div className="section-title">Execution</div>
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
          </div>

          <div className="auth-card">
            <div className="panel-header tight-header">
              <div>
                <p className="eyebrow">Operator</p>
                <h3>Session</h3>
              </div>
              <div className="panel-stat">{session.authenticated ? "Open" : "Locked"}</div>
            </div>
            {authLoading ? <p className="empty-copy">Checking session...</p> : null}
            {!authLoading && !session.configured ? <p className="empty-copy">Set operator auth secrets to unlock trade actions.</p> : null}
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
                  Sign in
                </button>
              </>
            ) : null}
            {!authLoading && session.authenticated ? (
              <>
                <p className="empty-copy">{session.username}</p>
                <button className="secondary-button" type="button" onClick={logout}>
                  Sign out
                </button>
              </>
            ) : null}
          </div>

          <label className="checkbox-row">
            <input type="checkbox" checked={confirmExecution} onChange={(event) => setConfirmExecution(event.target.checked)} />
            Confirm current orders
          </label>

          <div className="action-row">
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Running..." : "Refresh board"}
            </button>
            <button
              className="secondary-button"
              type="button"
              disabled={!data || executing || !confirmExecution || !session.authenticated || (Boolean(data?.executionConfirmation.required) && !Boolean(data?.executionConfirmation.configured))}
              onClick={executePlan}
            >
              {executing ? "Submitting..." : "Send top picks"}
            </button>
          </div>

          {data?.executionConfirmation.required && !data?.executionConfirmation.configured ? (
            <p className="empty-copy">Execution confirmation secrets are not configured.</p>
          ) : null}
          {error ? <p className="error-copy">{error}</p> : null}
        </form>

        <section className="results-panel">
          <div className="overview-strip">
            <article className="summary-card">
              <span>Session edge</span>
              <strong>{percent((data?.metrics.avg_top_k_return as number | undefined) ?? undefined)}</strong>
            </article>
            <article className="summary-card">
              <span>Hit rate</span>
              <strong>{percent((data?.metrics.top_k_hit_rate as number | undefined) ?? undefined)}</strong>
            </article>
            <article className="summary-card">
              <span>Sharpe</span>
              <strong>{number((data?.metrics.backtest_sharpe as number | undefined) ?? undefined, 2)}</strong>
            </article>
            <article className="summary-card">
              <span>Next 3d median</span>
              <strong>{percent((data?.metrics.median_future_return_3d as number | undefined) ?? undefined)}</strong>
            </article>
            <article className="summary-card">
              <span>Signal alignment</span>
              <strong>{percent((data?.metrics.signal_alignment_rate as number | undefined) ?? undefined)}</strong>
            </article>
            <article className="summary-card">
              <span>Minutes left</span>
              <strong>{number((data?.metrics.median_minutes_to_close as number | undefined) ?? undefined, 0)}</strong>
            </article>
          </div>

          <div className="leader-grid">
            {topSetups.length ? (
              topSetups.map((row, index) => (
                <article key={row.symbol} className="leader-card">
                  <div className="leader-header">
                    <div>
                      <p className="eyebrow">Setup {index + 1}</p>
                      <h3>{row.symbol}</h3>
                    </div>
                    <span className={`side-pill ${row.signal_side}`}>{row.signal_side}</span>
                  </div>
                  <div className="leader-metrics">
                    <div>
                      <span>To close</span>
                      <strong>{percent(row.predicted_return)}</strong>
                    </div>
                    <div>
                      <span>Next 1d</span>
                      <strong>{percent(row.future_return_1d ?? undefined)}</strong>
                    </div>
                    <div>
                      <span>Next 3d</span>
                      <strong>{percent(row.future_return_3d ?? undefined)}</strong>
                    </div>
                    <div>
                      <span>Next 5d</span>
                      <strong>{percent(row.future_return_5d ?? undefined)}</strong>
                    </div>
                  </div>
                  <div className="leader-footer">
                    <span>Confidence {percent(row.confidence)}</span>
                    <span>Timing {number(row.timing_score ?? undefined)}</span>
                    <span>Swing {number(row.future_score ?? undefined)}</span>
                  </div>
                </article>
              ))
            ) : (
              <article className="leader-card leader-empty">
                <p className="eyebrow">Board</p>
                <h3>No live ranking</h3>
                <p className="empty-copy">Run the scan to populate the mover board.</p>
              </article>
            )}
          </div>

          <div className="chart-card">
            <div className="panel-header tight-header">
              <div>
                <p className="eyebrow">Backtest</p>
                <h2>Equity path</h2>
              </div>
              <div className="panel-stat">{currency(startingCapital)}</div>
            </div>
            {data?.backtestCurve?.length ? (
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="sparkline">
                <path d={sparkline} vectorEffect="non-scaling-stroke" />
              </svg>
            ) : (
              <p className="empty-copy">No backtest curve yet.</p>
            )}
          </div>

          <div className="table-card">
            <div className="panel-header tight-header">
              <div>
                <p className="eyebrow">Mover Board</p>
                <h2>Current ranking</h2>
              </div>
              <div className="panel-stat">{data?.ranking.length ?? 0} rows</div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Bias</th>
                    <th>To Close</th>
                    <th>Next 1d</th>
                    <th>Next 3d</th>
                    <th>Next 5d</th>
                    <th>Conf</th>
                    <th>Timing</th>
                    <th>Swing</th>
                    <th>Final</th>
                    <th>Mins</th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.ranking ?? []).map((row) => (
                    <tr key={row.symbol}>
                      <td>{row.symbol}</td>
                      <td>
                        <span className={`side-pill ${row.signal_side}`}>{row.signal_side}</span>
                      </td>
                      <td>{percent(row.predicted_return)}</td>
                      <td>{percent(row.future_return_1d ?? undefined)}</td>
                      <td>{percent(row.future_return_3d ?? undefined)}</td>
                      <td>{percent(row.future_return_5d ?? undefined)}</td>
                      <td>{percent(row.confidence)}</td>
                      <td>{number(row.timing_score ?? undefined)}</td>
                      <td>{number(row.future_score ?? undefined)}</td>
                      <td>{number(row.final_score)}</td>
                      <td>{number(row.minutes_to_close ?? undefined, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="table-card">
            <div className="panel-header tight-header">
              <div>
                <p className="eyebrow">Execution</p>
                <h2>Order staging</h2>
              </div>
              <div className="panel-stat">{executionMode}</div>
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
                <p className="eyebrow">Open positions</p>
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
