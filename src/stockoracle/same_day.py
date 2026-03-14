from __future__ import annotations

import numpy as np
import pandas as pd


SAME_DAY_FEATURE_COLUMNS = [
    "bar_index",
    "bars_remaining",
    "slot_progress",
    "minutes_from_open",
    "minutes_to_close",
    "session_return_so_far",
    "bar_return_1",
    "bar_return_3",
    "bar_return_6",
    "intraday_volatility_6",
    "intraday_range_pct",
    "vwap_gap_live",
    "distance_from_high",
    "distance_from_low",
    "bar_volume_ratio",
    "benchmark_bar_return_1",
    "benchmark_session_return",
    "benchmark_vwap_gap_live",
    "relative_bar_strength",
    "relative_session_strength",
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_20",
    "volume_z_20",
    "volatility_20",
    "gap_pct",
    "rsi_14",
    "atr_pct_14",
    "price_position_20",
    "drawdown_20",
    "relative_strength_5",
    "relative_strength_20",
    "days_to_earnings",
    "days_since_earnings",
    "earnings_proximity",
]


DAILY_CONTEXT_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_20",
    "volume_z_20",
    "volatility_20",
    "gap_pct",
    "rsi_14",
    "atr_pct_14",
    "price_position_20",
    "drawdown_20",
    "relative_strength_5",
    "relative_strength_20",
    "days_to_earnings",
    "days_since_earnings",
    "earnings_proximity",
]


def _rolling_group_stat(series: pd.Series, window: int, op: str) -> pd.Series:
    if op == "std":
        return series.rolling(window).std()
    if op == "mean":
        return series.rolling(window).mean()
    raise ValueError(f"Unsupported rolling operation: {op}")


def build_same_day_dataset(
    intraday_frame: pd.DataFrame,
    daily_feature_frame: pd.DataFrame,
    benchmark_symbol: str,
) -> pd.DataFrame:
    if intraday_frame.empty:
        return pd.DataFrame(columns=["date", "timestamp", "symbol", "target_return", "target_abs_move", *SAME_DAY_FEATURE_COLUMNS])

    frame = intraday_frame.copy()
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_localize(None)
    frame["timestamp"] = timestamps
    minutes_of_day = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame = frame.loc[(minutes_of_day >= 570) & (minutes_of_day <= 960)].copy()
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["trade_price"] = frame["adj_close"].where(frame["adj_close"] > 0, frame["close"]).fillna(frame["close"])
    frame = frame.dropna(subset=["timestamp", "trade_price"]).sort_values(["symbol", "date", "timestamp"]).reset_index(drop=True)

    session_group = frame.groupby(["symbol", "date"], sort=False)
    frame["bar_index"] = session_group.cumcount()
    frame["bars_in_session"] = session_group["timestamp"].transform("size")
    frame["bars_remaining"] = frame["bars_in_session"] - frame["bar_index"] - 1
    frame["slot_progress"] = frame["bar_index"] / frame["bars_in_session"].replace(0, np.nan)

    session_start = session_group["timestamp"].transform("min")
    session_end = session_group["timestamp"].transform("max")
    session_open = session_group["open"].transform("first")
    session_close = session_group["trade_price"].transform("last")

    frame["minutes_from_open"] = (frame["timestamp"] - session_start).dt.total_seconds() / 60
    frame["minutes_to_close"] = (session_end - frame["timestamp"]).dt.total_seconds() / 60
    frame["session_return_so_far"] = frame["trade_price"] / session_open.replace(0.0, np.nan) - 1
    frame["bar_return_1"] = session_group["trade_price"].pct_change(1)
    frame["bar_return_3"] = session_group["trade_price"].pct_change(3)
    frame["bar_return_6"] = session_group["trade_price"].pct_change(6)
    frame["intraday_volatility_6"] = session_group["bar_return_1"].transform(lambda series: _rolling_group_stat(series, 6, "std"))
    frame["intraday_range_pct"] = (frame["high"] - frame["low"]) / frame["trade_price"].replace(0.0, np.nan)

    frame["cum_volume"] = session_group["volume"].cumsum()
    frame["cum_dollar_volume"] = (frame["trade_price"] * frame["volume"]).groupby([frame["symbol"], frame["date"]]).cumsum()
    frame["vwap_live"] = frame["cum_dollar_volume"] / frame["cum_volume"].replace(0.0, np.nan)
    frame["vwap_gap_live"] = frame["trade_price"] / frame["vwap_live"] - 1
    frame["session_high_so_far"] = session_group["high"].cummax()
    frame["session_low_so_far"] = session_group["low"].cummin()
    frame["distance_from_high"] = frame["trade_price"] / frame["session_high_so_far"].replace(0.0, np.nan) - 1
    frame["distance_from_low"] = frame["trade_price"] / frame["session_low_so_far"].replace(0.0, np.nan) - 1
    frame["bar_volume_ratio"] = session_group["volume"].transform(lambda series: series / _rolling_group_stat(series, 6, "mean").replace(0.0, np.nan))

    benchmark_intraday = (
        frame.loc[
            frame["symbol"] == benchmark_symbol,
            ["timestamp", "bar_return_1", "session_return_so_far", "vwap_gap_live"],
        ]
        .rename(
            columns={
                "bar_return_1": "benchmark_bar_return_1",
                "session_return_so_far": "benchmark_session_return",
                "vwap_gap_live": "benchmark_vwap_gap_live",
            }
        )
        .drop_duplicates(subset=["timestamp"])
    )
    frame = frame.merge(benchmark_intraday, on="timestamp", how="left")
    frame["relative_bar_strength"] = frame["bar_return_1"] - frame["benchmark_bar_return_1"]
    frame["relative_session_strength"] = frame["session_return_so_far"] - frame["benchmark_session_return"]

    daily_context = daily_feature_frame[["date", "symbol", *DAILY_CONTEXT_COLUMNS]].sort_values(["symbol", "date"]).copy()
    daily_context[DAILY_CONTEXT_COLUMNS] = daily_context.groupby("symbol", sort=False)[DAILY_CONTEXT_COLUMNS].shift(1)
    frame = frame.merge(daily_context, on=["date", "symbol"], how="left")

    frame["target_return"] = session_close / frame["trade_price"] - 1
    frame["target_abs_move"] = frame["target_return"].abs()
    frame.loc[frame["bars_remaining"] <= 0, ["target_return", "target_abs_move"]] = np.nan

    frame = frame.loc[frame["symbol"] != benchmark_symbol].copy()
    return frame.sort_values(["date", "timestamp", "symbol"]).reset_index(drop=True)


def select_current_same_day_frame(frame: pd.DataFrame, signal_bar_index: int | None = None) -> tuple[pd.DataFrame, int, pd.Timestamp]:
    tradable = frame.loc[frame["bars_remaining"] > 0].copy()
    if tradable.empty:
        raise ValueError("No intraday bars are available before the close. Try again during market hours.")

    if signal_bar_index is not None:
        candidates = tradable.loc[tradable["bar_index"] == signal_bar_index].copy()
        if candidates.empty:
            raise ValueError("The requested intraday decision slot is not available in the current lookback window.")
        current_date = candidates["date"].max()
        current_frame = candidates.loc[candidates["date"] == current_date].copy()
        latest_timestamp = current_frame["timestamp"].max()
        current_frame = current_frame.loc[current_frame["timestamp"] == latest_timestamp].copy()
        return current_frame, signal_bar_index, pd.Timestamp(latest_timestamp)

    latest_timestamp = tradable["timestamp"].max()
    current_frame = tradable.loc[tradable["timestamp"] == latest_timestamp].copy()
    slot = int(current_frame["bar_index"].mode().iloc[0])
    current_frame = tradable.loc[(tradable["date"] == current_frame["date"].max()) & (tradable["bar_index"] == slot)].copy()
    current_frame = current_frame.loc[current_frame["timestamp"] == current_frame["timestamp"].max()].copy()
    return current_frame, slot, pd.Timestamp(current_frame["timestamp"].max())