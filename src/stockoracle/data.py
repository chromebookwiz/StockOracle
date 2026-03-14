from __future__ import annotations

from datetime import date

import pandas as pd

from .runtime import cached_call
from .yahoo_api import fetch_chart


def _normalize_chart(symbol: str, payload: dict, time_column: str) -> pd.DataFrame:
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not result:
        return pd.DataFrame(columns=[time_column, "symbol", "open", "high", "low", "close", "adj_close", "volume"])

    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
    if not timestamps:
        return pd.DataFrame(columns=[time_column, "symbol", "open", "high", "low", "close", "adj_close", "volume"])

    frame = pd.DataFrame(
        {
            time_column: pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            "symbol": symbol.upper(),
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "adj_close": adjclose if len(adjclose) == len(timestamps) else quote.get("close", []),
            "volume": quote.get("volume", []),
        }
    )
    return frame


def _download_symbol_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    payload = cached_call(
        namespace="daily-download",
        payload={"symbol": symbol, "start": start_date, "end": end_date},
        ttl_seconds=900,
        limiter_key="yahoo-chart",
        minimum_interval_seconds=0.2,
        loader=lambda: fetch_chart(symbol, start_date=start_date, end_date=end_date, interval="1d"),
    )
    return _normalize_chart(symbol, payload, "date")


def _download_symbol_intraday(symbol: str, period_days: int, interval: str) -> pd.DataFrame:
    payload = cached_call(
        namespace="intraday-download",
        payload={"symbol": symbol, "period_days": period_days, "interval": interval},
        ttl_seconds=120,
        limiter_key="yahoo-chart",
        minimum_interval_seconds=0.2,
        loader=lambda: fetch_chart(symbol, period_days=period_days, interval=interval, prepost=True),
    )
    return _normalize_chart(symbol, payload, "timestamp")


def download_market_data(symbols: list[str], start_date: str, end_date: str | None = None) -> pd.DataFrame:
    resolved_end = end_date or date.today().isoformat()
    frames = [_download_symbol_daily(symbol, start_date, resolved_end) for symbol in symbols]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
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

    frame = frame[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]
    frame = frame.dropna(subset=["date", "close"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    return frame


def download_intraday_data(symbols: list[str], period_days: int = 7, interval: str = "60m") -> pd.DataFrame:
    frames = [_download_symbol_intraday(symbol, period_days=period_days, interval=interval) for symbol in symbols]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "adj_close", "volume"])
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "adj_close", "volume"])

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

    intraday = frame[["timestamp", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]
    intraday = intraday.dropna(subset=["timestamp", "close"])
    intraday = intraday.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return intraday