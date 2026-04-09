from __future__ import annotations

import numpy as np
import pandas as pd

from .alternative_data import fetch_earnings_calendar, fetch_live_alternative_data
from .backtest import run_backtest
from .config import AppConfig
from .data import download_intraday_data, download_market_data
from .execution import build_execution_plan, execution_plan_frame
from .features import FEATURE_COLUMNS, add_multi_horizon_targets, build_feature_frame
from .modeling import DailyHorizonRanker, EnsembleRanker, ModelOutput, build_recency_weights, evaluate_holdout
from .same_day import SAME_DAY_FEATURE_COLUMNS, build_same_day_dataset, select_current_same_day_frame


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean()) / std


def _apply_live_overlays(ranking: pd.DataFrame, live_features: pd.DataFrame) -> pd.DataFrame:
    if live_features.empty:
        ranking["overlay_score"] = 0.0
        ranking["final_score"] = ranking["score"]
        ranking["signal_side"] = ranking["final_score"].ge(0).map({True: "long", False: "short"})
        ranking["opportunity_score"] = ranking.get("opportunity_score", ranking["final_score"].abs())
        return ranking.sort_values(["opportunity_score", "confidence"], ascending=[False, False]).reset_index(drop=True)

    merged = ranking.merge(live_features, on="symbol", how="left")
    overlay_score = (
        0.35 * _zscore(merged.get("news_sentiment", pd.Series(0.0, index=merged.index)).fillna(0.0))
        + 0.15 * _zscore(merged.get("news_buzz", pd.Series(0.0, index=merged.index)).fillna(0.0))
        + 0.10 * _zscore(merged.get("recent_news_count", pd.Series(0.0, index=merged.index)).fillna(0.0))
        - 0.15 * _zscore(merged.get("options_put_call_oi", pd.Series(1.0, index=merged.index)).fillna(1.0))
        + 0.10 * _zscore(merged.get("options_call_put_volume", pd.Series(1.0, index=merged.index)).fillna(1.0))
        + 0.05 * _zscore(merged.get("options_iv_skew", pd.Series(0.0, index=merged.index)).fillna(0.0))
        + 0.10 * _zscore(merged.get("earnings_proximity", pd.Series(0.0, index=merged.index)).fillna(0.0))
    )
    merged["overlay_score"] = overlay_score
    merged["final_score"] = merged["score"] + overlay_score
    merged["signal_side"] = merged["final_score"].ge(0).map({True: "long", False: "short"})
    merged["opportunity_score"] = merged["final_score"].abs() + 0.10 * _zscore(merged["confidence"].fillna(0.0))
    return merged.sort_values(["opportunity_score", "confidence"], ascending=[False, False]).reset_index(drop=True)


def _blend_future_horizon_signals(ranking: pd.DataFrame, future_predictions: pd.DataFrame) -> pd.DataFrame:
    ranking = ranking.copy()
    ranking["timing_score"] = ranking["final_score"]

    if future_predictions.empty:
        ranking["future_score"] = 0.0
        ranking["future_confidence"] = ranking["confidence"].fillna(0.0)
        ranking["future_return_1d"] = pd.NA
        ranking["future_return_3d"] = pd.NA
        ranking["future_return_5d"] = pd.NA
        ranking["direction_alignment"] = 0.0
        return ranking.sort_values(["opportunity_score", "confidence"], ascending=[False, False]).reset_index(drop=True)

    merged = ranking.merge(
        future_predictions[
            [
                "symbol",
                "future_score",
                "future_confidence",
                "future_return_1d",
                "future_return_3d",
                "future_return_5d",
                "future_move_5d",
                "future_return_blend",
            ]
        ],
        on="symbol",
        how="left",
    )

    for column in [
        "future_score",
        "future_confidence",
        "future_return_1d",
        "future_return_3d",
        "future_return_5d",
        "future_move_5d",
        "future_return_blend",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    timing_direction = np.sign(pd.to_numeric(merged["timing_score"], errors="coerce").fillna(0.0))
    future_direction = np.sign(merged["future_score"].fillna(0.0))
    merged["direction_alignment"] = np.where(
        (timing_direction == 0) | (future_direction == 0),
        0.0,
        np.where(timing_direction == future_direction, 1.0, -1.0),
    )

    alignment_bonus = merged["direction_alignment"] * (0.5 + merged["future_confidence"].fillna(0.0))
    merged["final_score"] = (
        0.52 * _zscore(merged["timing_score"].fillna(0.0))
        + 0.28 * _zscore(merged["future_score"].fillna(0.0))
        + 0.12 * _zscore(merged["future_return_blend"].fillna(0.0))
        + 0.08 * _zscore(alignment_bonus.fillna(0.0))
    )
    base_confidence = 0.65 * merged["confidence"].fillna(0.0) + 0.35 * merged["future_confidence"].fillna(0.0)
    merged["confidence"] = np.clip(
        base_confidence
        + np.where(merged["direction_alignment"] > 0, 0.05, 0.0)
        - np.where(merged["direction_alignment"] < 0, 0.12, 0.0),
        0.0,
        1.0,
    )
    merged["signal_side"] = np.where(merged["final_score"] >= 0, "long", "short")
    merged["opportunity_score"] = (
        merged["final_score"].abs()
        + 0.10 * _zscore(merged["confidence"].fillna(0.0))
        + 0.08 * _zscore(merged["predicted_move"].fillna(0.0))
        + 0.10 * _zscore(merged["future_move_5d"].fillna(0.0))
        + 0.08 * _zscore(merged["future_return_3d"].abs().fillna(0.0))
    )
    return merged.sort_values(["opportunity_score", "confidence"], ascending=[False, False]).reset_index(drop=True)


def run_stock_oracle(config: AppConfig) -> ModelOutput:
    symbols = config.normalized_universe()
    tradable_symbols = config.tradable_universe()
    market_data = download_market_data(symbols=symbols, start_date=config.start_date)
    if market_data.empty:
        raise ValueError("No market data returned. Check the ticker universe or network access.")

    intraday_data = download_intraday_data(
        symbols=symbols,
        period_days=config.intraday_period_days,
        interval=config.intraday_interval,
    )
    earnings_calendar = (
        fetch_earnings_calendar(tradable_symbols)
        if config.enable_earnings_features
        else pd.DataFrame(columns=["symbol", "earnings_date"])
    )

    daily_feature_frame = build_feature_frame(
        market_data,
        benchmark_symbol=config.benchmark,
        earnings_calendar=earnings_calendar,
    )
    daily_feature_frame = daily_feature_frame.groupby("symbol", group_keys=False).filter(lambda frame: len(frame) >= config.min_history_days)
    if daily_feature_frame.empty:
        raise ValueError("Not enough history to build features for the selected universe.")

    same_day_frame = build_same_day_dataset(
        intraday_frame=intraday_data,
        daily_feature_frame=daily_feature_frame,
        benchmark_symbol=config.benchmark,
    )
    if same_day_frame.empty:
        raise ValueError("No intraday same-day training rows were produced for the selected universe.")

    current_frame, signal_bar_index, latest_timestamp = select_current_same_day_frame(
        same_day_frame,
        signal_bar_index=config.signal_bar_index,
    )
    current_date = pd.Timestamp(current_frame["date"].max())

    training_frame = same_day_frame.loc[
        (same_day_frame["date"] < current_date) & (same_day_frame["bar_index"] == signal_bar_index)
    ].dropna(subset=["target_return", "target_abs_move"])
    training_frame = training_frame.dropna(subset=SAME_DAY_FEATURE_COLUMNS, how="all")
    if training_frame.empty:
        raise ValueError("No training rows available after feature engineering.")

    active_feature_columns = [column for column in SAME_DAY_FEATURE_COLUMNS if training_frame[column].notna().any()]
    if not active_feature_columns:
        raise ValueError("No usable feature columns were produced for training.")

    current_frame = current_frame.dropna(subset=active_feature_columns, how="all")
    if current_frame.empty:
        raise ValueError("No latest-session rows available for ranking.")

    ranker = EnsembleRanker(random_state=config.random_state)
    ranker.fit(training_frame, active_feature_columns, sample_weight=build_recency_weights(training_frame))
    current_ranking = ranker.predict(current_frame, active_feature_columns)

    daily_horizon_frame = add_multi_horizon_targets(daily_feature_frame, horizons=(1, 3, 5))
    daily_horizon_frame = daily_horizon_frame.loc[daily_horizon_frame["symbol"] != config.benchmark].copy()
    future_predictions = pd.DataFrame()
    future_feature_importance = pd.DataFrame(columns=["feature", "importance"])
    available_daily_dates = daily_horizon_frame.loc[daily_horizon_frame["date"] <= current_date, "date"].dropna()
    daily_current_date = pd.Timestamp(available_daily_dates.max()) if not available_daily_dates.empty else current_date
    daily_training_frame = daily_horizon_frame.loc[daily_horizon_frame["date"] < daily_current_date].copy()
    current_daily_frame = daily_horizon_frame.loc[daily_horizon_frame["date"] == daily_current_date].copy()
    active_daily_feature_columns = [column for column in FEATURE_COLUMNS if column in daily_training_frame.columns and daily_training_frame[column].notna().any()]

    if not daily_training_frame.empty and not current_daily_frame.empty and active_daily_feature_columns:
        daily_ranker = DailyHorizonRanker(random_state=config.random_state)
        daily_ranker.fit(
            daily_training_frame,
            active_daily_feature_columns,
            sample_weight=build_recency_weights(daily_training_frame, half_life_sessions=25.0),
        )
        future_predictions = daily_ranker.predict(current_daily_frame)
        future_feature_importance = daily_ranker.feature_importance()

    live_features = pd.DataFrame(columns=["symbol"])
    if config.enable_live_news or config.enable_live_options or config.enable_earnings_features:
        live_features = fetch_live_alternative_data(tradable_symbols)
        if not config.enable_live_news:
            for column in ["news_sentiment", "news_buzz", "recent_news_count"]:
                if column in live_features.columns:
                    live_features[column] = 0.0
        if not config.enable_live_options:
            for column in ["options_put_call_oi", "options_call_put_volume", "options_iv_skew", "options_open_interest_total"]:
                if column in live_features.columns:
                    live_features[column] = 1.0 if "ratio" in column else 0.0

    current_ranking = _apply_live_overlays(current_ranking, live_features)
    current_ranking = _blend_future_horizon_signals(current_ranking, future_predictions)
    current_ranking = current_ranking.merge(
        current_frame[["symbol", "minutes_to_close", "session_return_so_far", "bar_index", "timestamp", "volume"]],
        on="symbol",
        how="left",
        suffixes=("", "_current"),
    )

    metrics, holdout_predictions = evaluate_holdout(
        full_frame=training_frame,
        feature_columns=active_feature_columns,
        holdout_days=config.holdout_days,
        top_k=config.top_k,
        random_state=config.random_state,
    )
    backtest_metrics, backtest_curve = run_backtest(
        holdout_predictions=holdout_predictions,
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
        slippage_bps=config.slippage_bps,
        max_position_weight=config.max_position_weight,
    )

    default_metrics = {
        "avg_top_k_return": 0.0,
        "top_k_hit_rate": 0.0,
        "avg_rank_ic": 0.0,
        "directional_accuracy": 0.0,
        "avg_prediction_interval": 0.0,
        "holdout_days_evaluated": 0.0,
        "backtest_total_return": 0.0,
        "backtest_annualized_return": 0.0,
        "backtest_annualized_volatility": 0.0,
        "backtest_sharpe": 0.0,
        "backtest_max_drawdown": 0.0,
        "backtest_win_rate": 0.0,
        "backtest_avg_turnover": 0.0,
    }
    default_metrics.update(metrics)
    default_metrics.update(backtest_metrics)
    metrics = default_metrics
    metrics["latest_ranking_date"] = current_date.isoformat()
    metrics["latest_ranking_timestamp"] = latest_timestamp.isoformat()
    metrics["signal_bar_index"] = float(signal_bar_index)
    metrics["median_minutes_to_close"] = float(current_frame["minutes_to_close"].median())
    metrics["long_candidates"] = float((current_ranking["signal_side"] == "long").sum())
    metrics["short_candidates"] = float((current_ranking["signal_side"] == "short").sum())
    metrics["avg_probability_up"] = float(current_ranking["probability_up"].mean())
    metrics["median_future_return_1d"] = float(pd.to_numeric(current_ranking.get("future_return_1d"), errors="coerce").median())
    metrics["median_future_return_3d"] = float(pd.to_numeric(current_ranking.get("future_return_3d"), errors="coerce").median())
    metrics["median_future_return_5d"] = float(pd.to_numeric(current_ranking.get("future_return_5d"), errors="coerce").median())
    metrics["signal_alignment_rate"] = float((pd.to_numeric(current_ranking.get("direction_alignment"), errors="coerce") > 0).mean())
    feature_importance = ranker.feature_importance(active_feature_columns)
    if not future_feature_importance.empty:
        feature_importance = (
            pd.concat([feature_importance, future_feature_importance], ignore_index=True)
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    execution_plan = execution_plan_frame(
        build_execution_plan(
            ranking=current_ranking,
            top_k=config.top_k,
            capital=config.starting_capital,
            max_position_weight=config.max_position_weight,
            max_notional_per_trade=config.max_notional_per_trade,
        )
    )
    return ModelOutput(
        ranking=current_ranking,
        metrics=metrics,
        feature_importance=feature_importance,
        holdout_predictions=holdout_predictions,
        backtest_curve=backtest_curve,
        execution_plan=execution_plan,
    )