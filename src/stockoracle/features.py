from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_8",
    "ret_10",
    "ret_20",
    "ret_60",
    "volume_z_20",
    "dollar_volume_z_20",
    "volatility_10",
    "volatility_20",
    "volatility_ratio_10_20",
    "range_pct",
    "gap_pct",
    "sma_10_gap",
    "sma_20_gap",
    "sma_50_gap",
    "ema_12_gap",
    "ema_26_gap",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "atr_pct_14",
    "atr_regime_14_50",
    "price_position_20",
    "drawdown_20",
    "breakout_distance_55",
    "momentum_spread_5_20",
    "momentum_spread_20_60",
    "mean_reversion_3_10",
    "trend_gap_20_50",
    "realized_skew_20",
    "realized_kurtosis_20",
    "benchmark_ret_5",
    "benchmark_ret_20",
    "relative_strength_5",
    "relative_strength_20",
    "relative_volume_strength",
    "intraday_trend",
    "intraday_last_bar_return",
    "intraday_range_pct",
    "intraday_vwap_gap",
    "intraday_close_to_high",
    "intraday_volume_ratio",
    "days_to_earnings",
    "days_since_earnings",
    "earnings_proximity",
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def build_intraday_feature_frame(intraday_frame: pd.DataFrame) -> pd.DataFrame:
    if intraday_frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "intraday_trend",
                "intraday_last_bar_return",
                "intraday_range_pct",
                "intraday_vwap_gap",
                "intraday_close_to_high",
                "intraday_volume_ratio",
            ]
        )

    frame = intraday_frame.copy()
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert(None)
    frame["timestamp"] = timestamps
    frame["date"] = frame["timestamp"].dt.normalize()

    records: list[dict[str, float | pd.Timestamp | str]] = []
    for (symbol, session_date), session_frame in frame.groupby(["symbol", "date"], sort=False):
        ordered = session_frame.sort_values("timestamp")
        first_open = ordered["open"].iloc[0]
        last_close = ordered["close"].iloc[-1]
        high = ordered["high"].max()
        low = ordered["low"].min()
        volume = ordered["volume"].fillna(0.0)
        vwap_denominator = float(volume.sum())
        vwap = (ordered["close"] * volume).sum() / vwap_denominator if vwap_denominator > 0 else np.nan
        last_bar_return = ordered["close"].pct_change().iloc[-1] if len(ordered) > 1 else np.nan

        records.append(
            {
                "date": session_date,
                "symbol": symbol,
                "intraday_trend": last_close / first_open - 1 if first_open else np.nan,
                "intraday_last_bar_return": last_bar_return,
                "intraday_range_pct": (high - low) / first_open if first_open else np.nan,
                "intraday_vwap_gap": last_close / vwap - 1 if pd.notna(vwap) and vwap else np.nan,
                "intraday_close_to_high": last_close / high - 1 if high else np.nan,
                "intraday_volume": vwap_denominator,
            }
        )

    aggregated = pd.DataFrame(records).sort_values(["symbol", "date"]).reset_index(drop=True)
    aggregated["intraday_volume_ratio"] = aggregated.groupby("symbol")["intraday_volume"].transform(
        lambda series: series / series.rolling(5).mean().replace(0.0, np.nan)
    )
    return aggregated.drop(columns=["intraday_volume"])


def build_earnings_feature_frame(price_frame: pd.DataFrame, earnings_calendar: pd.DataFrame) -> pd.DataFrame:
    base = price_frame[["date", "symbol"]].drop_duplicates().sort_values(["symbol", "date"]).reset_index(drop=True)
    if earnings_calendar.empty:
        base["days_to_earnings"] = np.nan
        base["days_since_earnings"] = np.nan
        base["earnings_proximity"] = 0.0
        return base

    enriched_frames: list[pd.DataFrame] = []
    earnings_calendar = earnings_calendar.copy()
    earnings_calendar["earnings_date"] = pd.to_datetime(earnings_calendar["earnings_date"]).dt.normalize()

    for symbol, symbol_frame in base.groupby("symbol", sort=False):
        frame = symbol_frame.copy()
        earnings_dates = (
            earnings_calendar.loc[earnings_calendar["symbol"] == symbol, "earnings_date"]
            .dropna()
            .sort_values()
            .drop_duplicates()
            .to_numpy(dtype="datetime64[D]")
        )
        if len(earnings_dates) == 0:
            frame["days_to_earnings"] = np.nan
            frame["days_since_earnings"] = np.nan
            frame["earnings_proximity"] = 0.0
            enriched_frames.append(frame)
            continue

        trade_dates = frame["date"].to_numpy(dtype="datetime64[D]")
        next_index = np.searchsorted(earnings_dates, trade_dates, side="left")
        prev_index = next_index - 1

        days_to: list[float] = []
        days_since: list[float] = []
        for idx, trade_date in enumerate(trade_dates):
            if next_index[idx] < len(earnings_dates):
                next_days = float((earnings_dates[next_index[idx]] - trade_date).astype(int))
            else:
                next_days = np.nan

            if prev_index[idx] >= 0:
                prev_days = float((trade_date - earnings_dates[prev_index[idx]]).astype(int))
            else:
                prev_days = np.nan

            days_to.append(next_days)
            days_since.append(prev_days)

        frame["days_to_earnings"] = days_to
        frame["days_since_earnings"] = days_since
        distance = pd.concat(
            [frame["days_to_earnings"].abs(), frame["days_since_earnings"].abs()],
            axis=1,
        ).min(axis=1)
        frame["earnings_proximity"] = np.exp(-(distance.fillna(30.0) / 7.0))
        enriched_frames.append(frame)

    return pd.concat(enriched_frames, ignore_index=True)


def build_feature_frame(
    price_frame: pd.DataFrame,
    benchmark_symbol: str,
    intraday_frame: pd.DataFrame | None = None,
    earnings_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []

    for symbol, symbol_frame in price_frame.groupby("symbol", sort=False):
        frame = symbol_frame.sort_values("date").copy()
        close = frame["adj_close"].where(frame["adj_close"] > 0, frame["close"])
        close = close.fillna(frame["close"])

        frame["ret_1"] = close.pct_change(1)
        frame["ret_2"] = close.pct_change(2)
        frame["ret_3"] = close.pct_change(3)
        frame["ret_5"] = close.pct_change(5)
        frame["ret_8"] = close.pct_change(8)
        frame["ret_10"] = close.pct_change(10)
        frame["ret_20"] = close.pct_change(20)
        frame["ret_60"] = close.pct_change(60)

        log_volume = np.log1p(frame["volume"])
        volume_mean = log_volume.rolling(20).mean()
        volume_std = log_volume.rolling(20).std().replace(0.0, np.nan)
        frame["volume_z_20"] = (log_volume - volume_mean) / volume_std
        dollar_volume = np.log1p((frame["close"].fillna(0.0) * frame["volume"].fillna(0.0)).clip(lower=0.0))
        dollar_volume_mean = dollar_volume.rolling(20).mean()
        dollar_volume_std = dollar_volume.rolling(20).std().replace(0.0, np.nan)
        frame["dollar_volume_z_20"] = (dollar_volume - dollar_volume_mean) / dollar_volume_std

        frame["volatility_10"] = frame["ret_1"].rolling(10).std()
        frame["volatility_20"] = frame["ret_1"].rolling(20).std()
        frame["volatility_ratio_10_20"] = frame["volatility_10"] / frame["volatility_20"].replace(0.0, np.nan)
        frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan)
        frame["gap_pct"] = (frame["open"] - frame["close"].shift(1)) / frame["close"].shift(1).replace(0.0, np.nan)

        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_12 = _ema(close, 12)
        ema_26 = _ema(close, 26)
        atr_14 = _atr(frame, 14)
        atr_50 = _atr(frame, 50)
        rolling_high_55 = close.rolling(55).max()

        frame["sma_10_gap"] = close / sma_10 - 1
        frame["sma_20_gap"] = close / sma_20 - 1
        frame["sma_50_gap"] = close / sma_50 - 1
        frame["ema_12_gap"] = close / ema_12 - 1
        frame["ema_26_gap"] = close / ema_26 - 1

        frame["macd"] = ema_12 - ema_26
        frame["macd_signal"] = _ema(frame["macd"], 9)
        frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
        frame["rsi_14"] = _rsi(close, 14)
        frame["atr_pct_14"] = atr_14 / close.replace(0.0, np.nan)
        frame["atr_regime_14_50"] = atr_14 / atr_50.replace(0.0, np.nan)

        rolling_min = close.rolling(20).min()
        rolling_max = close.rolling(20).max()
        frame["price_position_20"] = (close - rolling_min) / (rolling_max - rolling_min).replace(0.0, np.nan)
        frame["drawdown_20"] = close / rolling_max - 1
        frame["breakout_distance_55"] = close / rolling_high_55.replace(0.0, np.nan) - 1
        frame["momentum_spread_5_20"] = frame["ret_5"] - frame["ret_20"]
        frame["momentum_spread_20_60"] = frame["ret_20"] - frame["ret_60"]
        frame["mean_reversion_3_10"] = frame["ret_3"] - frame["ret_10"]
        frame["trend_gap_20_50"] = sma_20 / sma_50 - 1
        frame["realized_skew_20"] = frame["ret_1"].rolling(20).skew()
        frame["realized_kurtosis_20"] = frame["ret_1"].rolling(20).kurt()

        feature_frames.append(frame)

    combined = pd.concat(feature_frames, ignore_index=True)
    benchmark_frame = (
        combined.loc[combined["symbol"] == benchmark_symbol, ["date", "ret_5", "ret_20", "volume_z_20"]]
        .rename(
            columns={
                "ret_5": "benchmark_ret_5",
                "ret_20": "benchmark_ret_20",
                "volume_z_20": "benchmark_volume_z_20",
            }
        )
        .drop_duplicates(subset=["date"])
    )

    combined = combined.merge(benchmark_frame, on="date", how="left")
    combined["relative_strength_5"] = combined["ret_5"] - combined["benchmark_ret_5"]
    combined["relative_strength_20"] = combined["ret_20"] - combined["benchmark_ret_20"]
    combined["relative_volume_strength"] = combined["volume_z_20"] - combined["benchmark_volume_z_20"]

    intraday_features = build_intraday_feature_frame(intraday_frame if intraday_frame is not None else pd.DataFrame())
    if not intraday_features.empty:
        combined = combined.merge(intraday_features, on=["date", "symbol"], how="left")
    else:
        for column in [
            "intraday_trend",
            "intraday_last_bar_return",
            "intraday_range_pct",
            "intraday_vwap_gap",
            "intraday_close_to_high",
            "intraday_volume_ratio",
        ]:
            combined[column] = np.nan

    earnings_features = build_earnings_feature_frame(
        combined,
        earnings_calendar if earnings_calendar is not None else pd.DataFrame(columns=["symbol", "earnings_date"]),
    )
    combined = combined.merge(earnings_features, on=["date", "symbol"], how="left")
    return combined.sort_values(["date", "symbol"]).reset_index(drop=True)


def add_multi_horizon_targets(feature_frame: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    labeled_frames: list[pd.DataFrame] = []

    for _, symbol_frame in feature_frame.groupby("symbol", sort=False):
        frame = symbol_frame.sort_values("date").copy()
        current_close = frame["adj_close"].where(frame["adj_close"] > 0, frame["close"])
        for horizon_days in horizons:
            future_close = current_close.shift(-horizon_days)
            target_return_column = f"target_return_{horizon_days}d"
            target_abs_move_column = f"target_abs_move_{horizon_days}d"
            frame[target_return_column] = future_close / current_close - 1
            frame[target_abs_move_column] = frame[target_return_column].abs()
        labeled_frames.append(frame)

    return pd.concat(labeled_frames, ignore_index=True)


def add_targets(feature_frame: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    labeled = add_multi_horizon_targets(feature_frame, horizons=(horizon_days,))
    return labeled.rename(
        columns={
            f"target_return_{horizon_days}d": "target_return",
            f"target_abs_move_{horizon_days}d": "target_abs_move",
        }
    )