from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx


BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
}


def _client() -> httpx.Client:
    return httpx.Client(headers=BASE_HEADERS, timeout=20.0, follow_redirects=True)


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    with _client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def fetch_chart(symbol: str, *, start_date: str | None = None, end_date: str | None = None, period_days: int | None = None, interval: str = "1d", prepost: bool = False) -> dict[str, Any]:
    params: dict[str, Any] = {
        "interval": interval,
        "includePrePost": str(prepost).lower(),
        "events": "div,splits,capitalGains",
        "includeAdjustedClose": "true",
    }

    if period_days is not None:
        params["range"] = f"{period_days}d"
    else:
        start = datetime.fromisoformat(start_date or "2021-01-01").replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(end_date or datetime.now(tz=timezone.utc).date().isoformat()).replace(tzinfo=timezone.utc)
        params["period1"] = int(start.timestamp())
        params["period2"] = int(end.timestamp())

    return _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", params=params)


def fetch_quote_summary(symbol: str, modules: list[str]) -> dict[str, Any]:
    return _get_json(
        f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
        params={"modules": ",".join(modules)},
    )


def fetch_search(query: str, quotes_count: int = 0, news_count: int = 12) -> dict[str, Any]:
    return _get_json(
        "https://query1.finance.yahoo.com/v1/finance/search",
        params={"q": query, "quotesCount": quotes_count, "newsCount": news_count},
    )


def fetch_options_chain(symbol: str, expiration: int | None = None) -> dict[str, Any]:
    params = {"date": expiration} if expiration is not None else None
    return _get_json(f"https://query1.finance.yahoo.com/v7/finance/options/{symbol}", params=params)


def fetch_predefined_screener(scr_id: str, count: int = 25) -> dict[str, Any]:
    return _get_json(
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
        params={"scrIds": scr_id, "count": count},
    )