from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from . import AppConfig, run_stock_oracle
from .data import download_intraday_data
from .execution import ExecutionPlan, flatten_positions, get_broker
from .storage import get_binary_store
from .universe import DEFAULT_UNIVERSE


STATE_KEY = "autopilot/global.json"
NEW_YORK = ZoneInfo("America/New_York")


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_symbols(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_UNIVERSE.copy()
    return [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]


def _parse_hhmm(value: str, fallback_hour: int, fallback_minute: int) -> tuple[int, int]:
    try:
        hour_text, minute_text = value.split(":", maxsplit=1)
        return int(hour_text), int(minute_text)
    except Exception:
        return fallback_hour, fallback_minute


def _load_state() -> dict[str, Any]:
    payload = get_binary_store().get_bytes(STATE_KEY)
    if payload is None:
        return {
            "lastRunDate": None,
            "lastCloseDate": None,
            "runs": [],
            "closeouts": [],
            "paperEquity": None,
        }
    try:
        state = json.loads(payload.decode("utf-8"))
    except Exception:
        state = {}
    state.setdefault("lastRunDate", None)
    state.setdefault("lastCloseDate", None)
    state.setdefault("runs", [])
    state.setdefault("closeouts", [])
    state.setdefault("paperEquity", None)
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["runs"] = state.get("runs", [])[-30:]
    state["closeouts"] = state.get("closeouts", [])[-30:]
    get_binary_store().set_bytes(STATE_KEY, json.dumps(state, indent=2).encode("utf-8"))


@dataclass(slots=True)
class AutopilotSettings:
    enabled: bool
    mode: str
    daily_budget: float
    benchmark: str
    universe: list[str]
    start_date: str
    holdout_days: int
    top_k: int
    intraday_period_days: int
    intraday_interval: str
    enable_live_news: bool
    enable_live_options: bool
    enable_earnings_features: bool
    transaction_cost_bps: float
    slippage_bps: float
    max_position_weight: float
    max_notional_per_trade: float
    run_window: tuple[int, int]
    close_window: tuple[int, int]

    @classmethod
    def from_env(cls) -> "AutopilotSettings":
        daily_budget = float(os.getenv("STOCKORACLE_AUTOPILOT_DAILY_BUDGET", "10000"))
        top_k = int(os.getenv("STOCKORACLE_AUTOPILOT_TOP_K", "4"))
        max_weight = float(os.getenv("STOCKORACLE_AUTOPILOT_MAX_POSITION_WEIGHT", "0.25"))
        default_notional = daily_budget / max(top_k, 1)
        run_window = _parse_hhmm(os.getenv("STOCKORACLE_AUTOPILOT_RUN_TIME", "15:45"), 15, 45)
        close_window = _parse_hhmm(os.getenv("STOCKORACLE_AUTOPILOT_CLOSE_TIME", "15:58"), 15, 58)
        return cls(
            enabled=_parse_bool(os.getenv("STOCKORACLE_AUTOPILOT_ENABLED"), False),
            mode=os.getenv("STOCKORACLE_AUTOPILOT_MODE", os.getenv("STOCKORACLE_DEFAULT_EXECUTION_MODE", "paper")).strip().lower(),
            daily_budget=daily_budget,
            benchmark=os.getenv("STOCKORACLE_AUTOPILOT_BENCHMARK", "SPY").strip().upper() or "SPY",
            universe=_parse_csv_symbols(os.getenv("STOCKORACLE_AUTOPILOT_UNIVERSE")),
            start_date=os.getenv("STOCKORACLE_AUTOPILOT_START_DATE", "2021-01-01"),
            holdout_days=int(os.getenv("STOCKORACLE_AUTOPILOT_HOLDOUT_DAYS", "45")),
            top_k=top_k,
            intraday_period_days=int(os.getenv("STOCKORACLE_AUTOPILOT_INTRADAY_PERIOD_DAYS", "45")),
            intraday_interval=os.getenv("STOCKORACLE_AUTOPILOT_INTRADAY_INTERVAL", "15m"),
            enable_live_news=_parse_bool(os.getenv("STOCKORACLE_AUTOPILOT_ENABLE_LIVE_NEWS"), True),
            enable_live_options=_parse_bool(os.getenv("STOCKORACLE_AUTOPILOT_ENABLE_LIVE_OPTIONS"), True),
            enable_earnings_features=_parse_bool(os.getenv("STOCKORACLE_AUTOPILOT_ENABLE_EARNINGS"), True),
            transaction_cost_bps=float(os.getenv("STOCKORACLE_AUTOPILOT_TRANSACTION_COST_BPS", "5")),
            slippage_bps=float(os.getenv("STOCKORACLE_AUTOPILOT_SLIPPAGE_BPS", "5")),
            max_position_weight=max_weight,
            max_notional_per_trade=float(os.getenv("STOCKORACLE_AUTOPILOT_MAX_NOTIONAL_PER_TRADE", f"{default_notional:.2f}")),
            run_window=run_window,
            close_window=close_window,
        )


def _market_now(now: datetime | None = None) -> datetime:
    current = now or datetime.now(tz=UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(NEW_YORK)


def _within_window(now: datetime, target_hour: int, target_minute: int, tolerance_minutes: int = 7) -> bool:
    target_minutes = target_hour * 60 + target_minute
    current_minutes = now.hour * 60 + now.minute
    return abs(current_minutes - target_minutes) <= tolerance_minutes


def _latest_price_map(symbols: list[str], interval: str) -> dict[str, float]:
    if not symbols:
        return {}
    intraday = download_intraday_data(symbols=symbols, period_days=2, interval=interval)
    if intraday.empty:
        return {}
    ordered = intraday.sort_values(["symbol", "timestamp"]).copy()
    ordered["trade_price"] = ordered["adj_close"].where(ordered["adj_close"] > 0, ordered["close"]).fillna(ordered["close"])
    latest = ordered.groupby("symbol", as_index=False).tail(1)
    return {str(row["symbol"]): float(row["trade_price"] or 0.0) for _, row in latest.iterrows() if float(row["trade_price"] or 0.0) > 0}


def _adaptive_controls(state: dict[str, Any], settings: AutopilotSettings) -> tuple[int, float]:
    recent_closeouts = state.get("closeouts", [])[-5:]
    base_top_k = settings.top_k
    base_weight = settings.max_position_weight
    if not recent_closeouts:
        return base_top_k, base_weight

    average_return = mean(float(item.get("returnPct", 0.0) or 0.0) for item in recent_closeouts)
    average_hit_rate = mean(float(item.get("directionalAccuracy", 0.0) or 0.0) for item in recent_closeouts)

    adjusted_top_k = base_top_k
    adjusted_weight = base_weight
    if average_return > 0.003 and average_hit_rate > 0.52:
        adjusted_top_k = max(3, base_top_k - 1)
        adjusted_weight = min(0.35, base_weight + 0.03)
    elif average_return < -0.002 or average_hit_rate < 0.45:
        adjusted_top_k = min(8, base_top_k + 1)
        adjusted_weight = max(0.10, base_weight - 0.05)

    return adjusted_top_k, adjusted_weight


def _app_config_from_settings(settings: AutopilotSettings, state: dict[str, Any]) -> AppConfig:
    adaptive_top_k, adaptive_weight = _adaptive_controls(state, settings)
    per_trade_cap = min(settings.max_notional_per_trade, settings.daily_budget / max(adaptive_top_k, 1))
    return AppConfig(
        universe=settings.universe,
        benchmark=settings.benchmark,
        start_date=settings.start_date,
        holdout_days=settings.holdout_days,
        top_k=adaptive_top_k,
        intraday_period_days=settings.intraday_period_days,
        intraday_interval=settings.intraday_interval,
        enable_live_news=settings.enable_live_news,
        enable_live_options=settings.enable_live_options,
        enable_earnings_features=settings.enable_earnings_features,
        starting_capital=settings.daily_budget,
        execution_mode=settings.mode,
        max_notional_per_trade=per_trade_cap,
        transaction_cost_bps=settings.transaction_cost_bps,
        slippage_bps=settings.slippage_bps,
        max_position_weight=adaptive_weight,
    )


def autopilot_status(now: datetime | None = None) -> dict[str, Any]:
    settings = AutopilotSettings.from_env()
    state = _load_state()
    broker = get_broker(settings.mode)
    market_now = _market_now(now)
    return {
        "enabled": settings.enabled,
        "mode": settings.mode,
        "dailyBudget": settings.daily_budget,
        "marketTimestamp": market_now.isoformat(),
        "lastRunDate": state.get("lastRunDate"),
        "lastCloseDate": state.get("lastCloseDate"),
        "paperEquity": state.get("paperEquity"),
        "openPositions": broker.positions(),
        "recentRuns": state.get("runs", [])[-10:],
        "recentCloseouts": state.get("closeouts", [])[-10:],
    }


def run_autopilot(now: datetime | None = None, force: bool = False) -> dict[str, Any]:
    settings = AutopilotSettings.from_env()
    if not settings.enabled:
        raise ValueError("STOCKORACLE_AUTOPILOT_ENABLED must be set to true.")

    market_now = _market_now(now)
    if not force and market_now.weekday() >= 5:
        return {"status": "skipped", "reason": "weekend", "marketTimestamp": market_now.isoformat()}

    state = _load_state()
    trade_date = market_now.date().isoformat()
    if not force and state.get("lastRunDate") == trade_date:
        return {"status": "skipped", "reason": "already-ran", "tradeDate": trade_date}
    if not force and not _within_window(market_now, *settings.run_window):
        return {"status": "skipped", "reason": "outside-run-window", "marketTimestamp": market_now.isoformat()}

    stale_closeout = None
    existing_positions = get_broker(settings.mode).positions()
    if existing_positions and state.get("lastCloseDate") != trade_date:
        price_lookup = _latest_price_map([str(item["symbol"]) for item in existing_positions], settings.intraday_interval)
        stale_closeout = flatten_positions(settings.mode, price_lookup=price_lookup)

    config = _app_config_from_settings(settings, state)
    output = run_stock_oracle(config)
    broker = get_broker(settings.mode)
    plans = output.execution_plan.head(config.top_k)
    execution_result = broker.place_orders([ExecutionPlan(**row) for row in plans.to_dict(orient="records")])

    run_record = {
        "tradeDate": trade_date,
        "timestamp": market_now.isoformat(),
        "mode": settings.mode,
        "dailyBudget": settings.daily_budget,
        "topK": config.top_k,
        "maxPositionWeight": config.max_position_weight,
        "submitted": len(execution_result.get("orders", [])),
        "symbols": plans.get("symbol", pd.Series(dtype=str)).tolist(),
        "sides": plans.get("side", pd.Series(dtype=str)).tolist(),
        "avgPredictedReturn": float(plans.get("predicted_return", pd.Series(dtype=float)).mean() or 0.0) if not plans.empty else 0.0,
        "directionalAccuracy": float(output.metrics.get("directional_accuracy", 0.0) or 0.0),
        "holdoutReturn": float(output.metrics.get("avg_top_k_return", 0.0) or 0.0),
        "staleCloseout": stale_closeout,
    }

    state["lastRunDate"] = trade_date
    state.setdefault("runs", []).append(run_record)
    _save_state(state)

    return {
        "status": "executed",
        "run": run_record,
        "positions": execution_result.get("positions", []),
        "metrics": output.metrics,
    }


def close_autopilot(now: datetime | None = None, force: bool = False) -> dict[str, Any]:
    settings = AutopilotSettings.from_env()
    if not settings.enabled:
        raise ValueError("STOCKORACLE_AUTOPILOT_ENABLED must be set to true.")

    market_now = _market_now(now)
    if not force and market_now.weekday() >= 5:
        return {"status": "skipped", "reason": "weekend", "marketTimestamp": market_now.isoformat()}

    state = _load_state()
    trade_date = market_now.date().isoformat()
    if not force and state.get("lastCloseDate") == trade_date:
        return {"status": "skipped", "reason": "already-closed", "tradeDate": trade_date}
    if not force and not _within_window(market_now, *settings.close_window, tolerance_minutes=4):
        return {"status": "skipped", "reason": "outside-close-window", "marketTimestamp": market_now.isoformat()}

    positions = get_broker(settings.mode).positions()
    if not positions:
        state["lastCloseDate"] = trade_date
        _save_state(state)
        return {"status": "skipped", "reason": "no-open-positions", "tradeDate": trade_date}

    price_lookup = _latest_price_map([str(item["symbol"]) for item in positions], settings.intraday_interval)
    close_result = flatten_positions(settings.mode, price_lookup=price_lookup)
    realized_pnl = float(close_result.get("realizedPnl", 0.0) or 0.0)
    closeout_record = {
        "tradeDate": trade_date,
        "timestamp": market_now.isoformat(),
        "mode": settings.mode,
        "flattened": int(close_result.get("flattened", 0) or 0),
        "realizedPnl": realized_pnl,
        "returnPct": realized_pnl / settings.daily_budget if settings.daily_budget else 0.0,
        "directionalAccuracy": float(state.get("runs", [{}])[-1].get("directionalAccuracy", 0.0) or 0.0),
    }

    state["lastCloseDate"] = trade_date
    if settings.mode == "paper":
        prior_equity = float(state.get("paperEquity") or settings.daily_budget)
        state["paperEquity"] = prior_equity + realized_pnl
    state.setdefault("closeouts", []).append(closeout_record)
    _save_state(state)

    return {
        "status": "closed",
        "closeout": closeout_record,
        "positions": close_result.get("positions", []),
        "orders": close_result.get("orders", []),
    }