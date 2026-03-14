from __future__ import annotations

from datetime import datetime

import plotly.express as px
import streamlit as st

from stockoracle import AppConfig, run_stock_oracle
from stockoracle.universe import DEFAULT_UNIVERSE


st.set_page_config(page_title="StockOracle", page_icon="SO", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 186, 73, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 122, 204, 0.14), transparent 25%),
            linear-gradient(180deg, #f6f2e8 0%, #f9fbfd 100%);
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(17, 24, 39, 0.12);
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        margin-bottom: 0.25rem;
    }
    .hero p {
        font-size: 1rem;
        margin-bottom: 0;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>StockOracle</h1>
        <p>Same-day intraday mover ranking using bar-level models, live overlays, and close-of-session targets.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Setup")
    tickers_text = st.text_area("Universe", value=", ".join(DEFAULT_UNIVERSE), height=180)
    benchmark = st.text_input("Benchmark", value="SPY")
    start_date = st.date_input("History start", value=datetime(2021, 1, 1))
    holdout_days = st.slider("Holdout sessions", min_value=15, max_value=90, value=45)
    top_k = st.slider("Top picks to evaluate", min_value=3, max_value=20, value=10)
    intraday_period_days = st.slider("Intraday lookback (days)", min_value=10, max_value=60, value=45)
    intraday_interval = st.selectbox("Intraday interval", options=["5m", "15m", "30m", "60m"], index=1)
    execution_mode = st.selectbox("Execution mode", options=["paper", "alpaca"], index=0)
    transaction_cost_bps = st.slider("Transaction cost (bps)", min_value=0, max_value=25, value=5)
    slippage_bps = st.slider("Slippage (bps)", min_value=0, max_value=25, value=5)
    starting_capital = st.number_input("Starting capital", min_value=1000.0, value=25000.0, step=1000.0)
    max_notional_per_trade = st.number_input("Max notional / trade", min_value=100.0, value=5000.0, step=100.0)
    enable_live_news = st.toggle("Use live news sentiment", value=True)
    enable_live_options = st.toggle("Use live options flow", value=True)
    enable_earnings_features = st.toggle("Use earnings timing", value=True)
    run_button = st.button("Run ranking", type="primary")

st.caption("Research tool only. Rankings are model outputs, not financial advice.")

if run_button:
    universe = [symbol.strip().upper() for symbol in tickers_text.replace("\n", ",").split(",") if symbol.strip()]
    config = AppConfig(
        universe=universe,
        benchmark=benchmark.strip().upper() or "SPY",
        start_date=start_date.isoformat(),
        holdout_days=holdout_days,
        top_k=top_k,
        intraday_period_days=intraday_period_days,
        intraday_interval=intraday_interval,
        execution_mode=execution_mode,
        enable_live_news=enable_live_news,
        enable_live_options=enable_live_options,
        enable_earnings_features=enable_earnings_features,
        starting_capital=starting_capital,
        max_notional_per_trade=max_notional_per_trade,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )

    with st.spinner("Fetching data, training ensemble, and ranking the latest session..."):
        try:
            output = run_stock_oracle(config)
        except Exception as exc:
            st.error(f"Run failed: {exc}")
        else:
            metrics = output.metrics
            metric_columns = st.columns(4)
            metric_columns[0].metric("Avg same-slot holdout return", f"{metrics.get('avg_top_k_return', 0.0):.2%}")
            metric_columns[1].metric("Top-k hit rate", f"{metrics.get('top_k_hit_rate', 0.0):.2%}")
            metric_columns[2].metric("Avg rank IC", f"{metrics.get('avg_rank_ic', 0.0):.3f}")
            metric_columns[3].metric("Minutes to close", f"{metrics.get('median_minutes_to_close', 0.0):.0f}")

            backtest_columns = st.columns(4)
            backtest_columns[0].metric("Backtest total return", f"{metrics.get('backtest_total_return', 0.0):.2%}")
            backtest_columns[1].metric("Backtest Sharpe", f"{metrics.get('backtest_sharpe', 0.0):.2f}")
            backtest_columns[2].metric("Max drawdown", f"{metrics.get('backtest_max_drawdown', 0.0):.2%}")
            backtest_columns[3].metric("Signal slot", f"{int(metrics.get('signal_bar_index', 0.0))}")

            st.subheader("Latest ranking")
            ranking = output.ranking.copy()
            ranking["predicted_return"] = ranking["predicted_return"].map(lambda value: f"{value:.2%}")
            ranking["predicted_move"] = ranking["predicted_move"].map(lambda value: f"{value:.2%}")
            ranking["confidence"] = ranking["confidence"].map(lambda value: f"{value:.2%}")
            ranking["session_return_so_far"] = ranking["session_return_so_far"].map(lambda value: f"{value:.2%}")
            display_columns = [
                "symbol",
                "predicted_return",
                "predicted_move",
                "confidence",
                "model_score",
                "overlay_score",
                "final_score",
                "minutes_to_close",
                "session_return_so_far",
                "news_sentiment",
                "options_put_call_oi",
                "close",
                "volume",
            ]
            st.dataframe(ranking[display_columns], use_container_width=True, hide_index=True)

            details_column, chart_column = st.columns([1, 1.2])

            with details_column:
                st.subheader("Feature importance")
                st.dataframe(output.feature_importance.head(12), use_container_width=True, hide_index=True)

            with chart_column:
                st.subheader("Holdout prediction map")
                holdout = output.holdout_predictions.copy()
                if holdout.empty:
                    st.info("Not enough history to build the holdout diagnostic chart.")
                else:
                    scatter = px.scatter(
                        holdout,
                        x="score",
                        y="target_return",
                        color="symbol",
                        hover_data=["symbol"],
                        trendline="ols",
                        color_continuous_scale="Turbo",
                    )
                    scatter.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=420)
                    st.plotly_chart(scatter, use_container_width=True)

            st.subheader("Backtest equity")
            if output.backtest_curve.empty:
                st.info("Not enough holdout rows to render the backtest curve.")
            else:
                equity_chart = px.line(output.backtest_curve, x="date", y="equity")
                equity_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=320)
                st.plotly_chart(equity_chart, use_container_width=True)

            st.subheader("Top names")
            top_symbols = output.ranking.head(top_k)[["symbol", "predicted_return", "confidence", "final_score", "minutes_to_close"]].copy()
            top_symbols["predicted_return"] = top_symbols["predicted_return"].map(lambda value: f"{value:.2%}")
            top_symbols["confidence"] = top_symbols["confidence"].map(lambda value: f"{value:.2%}")
            st.table(top_symbols)

            st.subheader("Execution plan")
            if output.execution_plan.empty:
                st.info("No orders were generated from the current ranking.")
            else:
                st.dataframe(output.execution_plan, use_container_width=True, hide_index=True)
else:
    st.info("Choose a universe and run the ranking to generate the current same-day mover list into the close.")