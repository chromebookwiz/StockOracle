from __future__ import annotations

from typing import Any

from .runtime import cached_call
from .yahoo_api import fetch_predefined_screener


DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
    "AVGO",
    "NFLX",
    "CRM",
    "ORCL",
    "PLTR",
    "ADBE",
    "QCOM",
    "MU",
    "INTC",
    "UBER",
    "SHOP",
    "SNOW",
    "PANW",
    "CRWD",
    "ANET",
    "ARM",
    "SMCI",
    "JPM",
    "GS",
    "BAC",
    "MS",
    "V",
    "MA",
    "AXP",
    "LLY",
    "UNH",
    "ABBV",
    "XOM",
    "CVX",
    "CAT",
    "DE",
    "BA",
    "GE",
    "WMT",
    "COST",
    "HD",
    "LOW",
    "DIS",
    "BKNG",
    "SPY",
    "QQQ",
    "IWM",
]


GLOBAL_SCREENER_IDS = [
    "day_gainers",
    "day_losers",
    "most_actives",
    "small_cap_gainers",
]


def _extract_screener_quotes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    finance = payload.get("finance") if isinstance(payload, dict) else None
    result = finance.get("result") if isinstance(finance, dict) else None
    if not isinstance(result, list):
        return []

    quotes: list[dict[str, Any]] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        item_quotes = item.get("quotes") or []
        if not isinstance(item_quotes, list):
            continue
        for quote in item_quotes:
            if isinstance(quote, dict):
                quotes.append(quote)
    return quotes


def discover_global_movers(limit: int = 60) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for scr_id in GLOBAL_SCREENER_IDS:
        try:
            payload = cached_call(
                namespace="global-screener",
                payload={"scr_id": scr_id, "count": limit},
                ttl_seconds=300,
                limiter_key="yahoo-screener",
                minimum_interval_seconds=0.2,
                loader=lambda scr_id=scr_id, limit=limit: fetch_predefined_screener(scr_id, count=limit),
            )
        except Exception:
            continue

        for quote in _extract_screener_quotes(payload):
            symbol = str(quote.get("symbol", "")).strip().upper()
            quote_type = str(quote.get("quoteType", "")).upper()
            exchange = str(quote.get("fullExchangeName", "") or quote.get("exchange", "")).upper()
            if not symbol or symbol in seen:
                continue
            if quote_type not in {"EQUITY", "ETF"}:
                continue
            if any(marker in symbol for marker in ["=F", "-USD", "^", "/"]):
                continue
            if "OTC" in exchange:
                continue
            seen.add(symbol)
            candidates.append(symbol)
            if len(candidates) >= limit:
                return candidates

    return candidates or DEFAULT_UNIVERSE.copy()