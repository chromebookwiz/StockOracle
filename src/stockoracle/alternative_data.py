from __future__ import annotations

from datetime import datetime, timezone
from math import isnan

import numpy as np
import pandas as pd

from .runtime import cached_call
from .yahoo_api import fetch_options_chain, fetch_quote_summary, fetch_search


POSITIVE_WORDS = {
    "beat",
    "breakout",
    "bullish",
    "expands",
    "growth",
    "guidance",
    "higher",
    "launch",
    "momentum",
    "outperform",
    "rally",
    "record",
    "surge",
    "upgrade",
}

NEGATIVE_WORDS = {
    "cuts",
    "delay",
    "downgrade",
    "fall",
    "fraud",
    "investigation",
    "lawsuit",
    "lower",
    "miss",
    "plunge",
    "recall",
    "risk",
    "slowdown",
    "weak",
}


def _extract_nested_text(item: dict, *paths: tuple[str, ...]) -> str:
    for path in paths:
        current = item
        for segment in path:
            if not isinstance(current, dict) or segment not in current:
                current = None
                break
            current = current[segment]
        if isinstance(current, str) and current.strip():
            return current.strip()
    return ""


def _parse_published_at(item: dict) -> datetime | None:
    candidates = [
        item.get("providerPublishTime"),
        item.get("published_at"),
        item.get("pubDate"),
        item.get("published"),
        item.get("content", {}).get("pubDate") if isinstance(item.get("content"), dict) else None,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (int, float)) and not isnan(candidate):
            return datetime.fromtimestamp(candidate, tz=timezone.utc)
        if isinstance(candidate, str):
            normalized = candidate.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(normalized)
            except ValueError:
                continue
    return None


def _sentiment_score(text: str) -> float:
    words = [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()]
    if not words:
        return 0.0
    positive_hits = sum(word in POSITIVE_WORDS for word in words)
    negative_hits = sum(word in NEGATIVE_WORDS for word in words)
    return (positive_hits - negative_hits) / max(len(words), 1)


def _extract_earnings_dates(symbol: str) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []

    try:
        payload = fetch_quote_summary(symbol, ["calendarEvents"])
    except Exception:
        payload = {}

    result = ((payload.get("quoteSummary") or {}).get("result") or [None])[0] or {}
    earnings = ((result.get("calendarEvents") or {}).get("earnings") or {}).get("earningsDate") or []
    for item in earnings:
        raw_value = item.get("raw") if isinstance(item, dict) else None
        if raw_value:
            dates.append(pd.to_datetime(raw_value, unit="s", utc=True).tz_convert(None).normalize())

    unique_dates = sorted({pd.Timestamp(value).normalize() for value in dates if pd.notna(value)})
    return unique_dates


def fetch_earnings_calendar(symbols: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for symbol in symbols:
        try:
            dates = cached_call(
                namespace="earnings-calendar",
                payload={"symbol": symbol},
                ttl_seconds=21600,
                limiter_key="yahoo-quote-summary",
                minimum_interval_seconds=0.2,
                loader=lambda symbol=symbol: _extract_earnings_dates(symbol),
            )
        except Exception:
            dates = []

        for earnings_date in dates:
            records.append({"symbol": symbol, "earnings_date": earnings_date})

    return pd.DataFrame(records, columns=["symbol", "earnings_date"])


def fetch_live_alternative_data(symbols: list[str]) -> pd.DataFrame:
    today = pd.Timestamp.utcnow().normalize()
    records: list[dict[str, object]] = []

    for symbol in symbols:
        record: dict[str, object] = {
            "symbol": symbol,
            "news_sentiment": np.nan,
            "news_buzz": np.nan,
            "recent_news_count": np.nan,
            "days_to_earnings": np.nan,
            "earnings_proximity": np.nan,
            "options_put_call_oi": np.nan,
            "options_call_put_volume": np.nan,
            "options_iv_skew": np.nan,
            "options_open_interest_total": np.nan,
        }

        try:
            news_items = cached_call(
                namespace="live-news",
                payload={"symbol": symbol, "count": 12},
                ttl_seconds=300,
                limiter_key="yahoo-search",
                minimum_interval_seconds=0.25,
                loader=lambda symbol=symbol: (fetch_search(symbol, quotes_count=0, news_count=12).get("news") or []),
            )
        except Exception:
            news_items = []

        scores: list[float] = []
        weights: list[float] = []
        for item in news_items or []:
            title = _extract_nested_text(item, ("title",), ("content", "title"))
            summary = _extract_nested_text(item, ("summary",), ("content", "summary"), ("content", "description"))
            published_at = _parse_published_at(item)
            age_days = 3.0
            if published_at is not None:
                age_delta = pd.Timestamp.now(tz=timezone.utc) - pd.Timestamp(published_at)
                age_days = max(age_delta.total_seconds() / 86400, 0.0)
            weight = 1 / (1 + age_days)
            weights.append(weight)
            scores.append(_sentiment_score(f"{title} {summary}"))

        if weights:
            weighted = np.average(scores, weights=weights)
            record["news_sentiment"] = float(weighted)
            record["news_buzz"] = float(sum(weights))
            record["recent_news_count"] = float(len(weights))

        earnings_dates = cached_call(
            namespace="earnings-live",
            payload={"symbol": symbol},
            ttl_seconds=3600,
            limiter_key="yahoo-quote-summary",
            minimum_interval_seconds=0.2,
            loader=lambda symbol=symbol: _extract_earnings_dates(symbol),
        )
        if earnings_dates:
            deltas = [(earnings_date - today).days for earnings_date in earnings_dates]
            future_deltas = [delta for delta in deltas if delta >= 0]
            if future_deltas:
                nearest = min(future_deltas)
                record["days_to_earnings"] = float(nearest)
                record["earnings_proximity"] = float(np.exp(-(nearest / 7.0)))

        try:
            options_payload = cached_call(
                namespace="options-chain",
                payload={"symbol": symbol},
                ttl_seconds=180,
                limiter_key="yahoo-options",
                minimum_interval_seconds=0.25,
                loader=lambda symbol=symbol: fetch_options_chain(symbol),
            )
        except Exception:
            options_payload = {}

        option_result = ((options_payload.get("optionChain") or {}).get("result") or [None])[0] or {}
        expiries = option_result.get("expirationDates") or []
        options = option_result.get("options") or []
        if expiries and options:
            try:
                chain = options[0]
                calls = pd.DataFrame(chain.get("calls") or [])
                puts = pd.DataFrame(chain.get("puts") or [])

                total_call_oi = float(calls.get("openInterest", pd.Series(dtype=float)).fillna(0.0).sum())
                total_put_oi = float(puts.get("openInterest", pd.Series(dtype=float)).fillna(0.0).sum())
                total_call_volume = float(calls.get("volume", pd.Series(dtype=float)).fillna(0.0).sum())
                total_put_volume = float(puts.get("volume", pd.Series(dtype=float)).fillna(0.0).sum())
                call_iv = calls.get("impliedVolatility", pd.Series(dtype=float)).dropna()
                put_iv = puts.get("impliedVolatility", pd.Series(dtype=float)).dropna()

                record["options_put_call_oi"] = total_put_oi / max(total_call_oi, 1.0)
                record["options_call_put_volume"] = total_call_volume / max(total_put_volume, 1.0)
                record["options_iv_skew"] = float(call_iv.median() - put_iv.median()) if not call_iv.empty and not put_iv.empty else np.nan
                record["options_open_interest_total"] = total_call_oi + total_put_oi
            except Exception:
                pass

        records.append(record)

    return pd.DataFrame(records)