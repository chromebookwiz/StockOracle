from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

from .runtime import cached_call


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _normalize_download(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"])

    if isinstance(raw.columns, pd.MultiIndex):
        first_level = set(raw.columns.get_level_values(0))
        if first_level.intersection(PRICE_COLUMNS):
            try:
                stacked = raw.stack(level=1, future_stack=True).reset_index()
            except TypeError:
                stacked = raw.stack(level=1).reset_index()
        else:
            try:
                stacked = raw.stack(level=0, future_stack=True).reset_index()
            except TypeError:
                stacked = raw.stack(level=0).reset_index()

        stacked.columns = [str(column).lower().replace(" ", "_") for column in stacked.columns]
        if "level_1" in stacked.columns:
            stacked = stacked.rename(columns={"level_1": "symbol"})
        if "ticker" in stacked.columns:
            stacked = stacked.rename(columns={"ticker": "symbol"})
        if "date" not in stacked.columns:
            stacked = stacked.rename(columns={stacked.columns[0]: "date"})
        return stacked

    single_symbol = raw.reset_index().copy()
    single_symbol.columns = [str(column).lower().replace(" ", "_") for column in single_symbol.columns]
    single_symbol["symbol"] = "UNKNOWN"
    return single_symbol


def _download_batch(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    return cached_call(
        namespace="daily-download",
        payload={"symbols": symbols, "start": start_date, "end": end_date},
        ttl_seconds=900,
        limiter_key="yfinance-download",
        minimum_interval_seconds=0.35,
        loader=lambda: yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        ),
    )


def _download_intraday_batch(symbols: list[str], period_days: int, interval: str) -> pd.DataFrame:
    return cached_call(
        namespace="intraday-download",
        payload={"symbols": symbols, "period_days": period_days, "interval": interval},
        ttl_seconds=120,
        limiter_key="yfinance-download",
        minimum_interval_seconds=0.35,
        loader=lambda: yf.download(
            tickers=symbols,
            period=f"{period_days}d",
            interval=interval,
            prepost=True,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        ),
    )


def download_market_data(symbols: list[str], start_date: str, end_date: str | None = None) -> pd.DataFrame:
    resolved_end = end_date or date.today().isoformat()
    raw = _download_batch(symbols, start_date, resolved_end)
    frame = _normalize_download(raw)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper()

    expected_symbols = {symbol.upper() for symbol in symbols}
    available_symbols = set(frame["symbol"].unique())
    missing_symbols = sorted(expected_symbols - available_symbols)

    if missing_symbols:
        recovered_frames: list[pd.DataFrame] = [frame]
        for symbol in missing_symbols:
            retry_raw = _download_batch([symbol], start_date, resolved_end)
            retry_frame = _normalize_download(retry_raw)
            if retry_frame.empty:
                continue
            retry_frame["date"] = pd.to_datetime(retry_frame["date"])
            retry_frame["symbol"] = symbol
            recovered_frames.append(retry_frame)
        frame = pd.concat(recovered_frames, ignore_index=True)

    required = {
        "open": 0.0,
        "high": 0.0,
        "low": 0.0,
        "close": 0.0,
        "adj_close": 0.0,
        "volume": 0.0,
    }
    for column, default_value in required.items():
        if column not in frame.columns:
            frame[column] = default_value

    if (frame["symbol"] == "UNKNOWN").all() and len(symbols) == 1:
        frame["symbol"] = symbols[0].upper()

    frame = frame[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]
    frame = frame.dropna(subset=["date", "close"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    return frame


def download_intraday_data(symbols: list[str], period_days: int = 7, interval: str = "60m") -> pd.DataFrame:
    raw = _download_intraday_batch(symbols, period_days=period_days, interval=interval)
    frame = _normalize_download(raw)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "adj_close", "volume"])

    time_column = "datetime" if "datetime" in frame.columns else "date"
    frame = frame.rename(columns={time_column: "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()

    required = {
        "open": 0.0,
        "high": 0.0,
        "low": 0.0,
        "close": 0.0,
        "adj_close": 0.0,
        "volume": 0.0,
    }
    for column, default_value in required.items():
        if column not in frame.columns:
            frame[column] = default_value

    if (frame["symbol"] == "UNKNOWN").all() and len(symbols) == 1:
        frame["symbol"] = symbols[0].upper()

    intraday = frame[["timestamp", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]
    intraday = intraday.dropna(subset=["timestamp", "close"])
    intraday = intraday.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return intraday