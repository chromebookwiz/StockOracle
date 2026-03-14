from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stockoracle import AppConfig, run_stock_oracle  # noqa: E402
from stockoracle.execution import ExecutionPlan, get_broker  # noqa: E402


app = FastAPI(title="StockOracle API")


class RankingRequest(BaseModel):
    universe: list[str] = Field(default_factory=list)
    benchmark: str = "SPY"
    startDate: str = "2021-01-01"
    holdoutDays: int = 45
    topK: int = 10
    intradayPeriodDays: int = 45
    intradayInterval: str = "15m"
    signalBarIndex: int | None = None
    enableLiveNews: bool = True
    enableLiveOptions: bool = True
    enableEarningsFeatures: bool = True
    startingCapital: float = 25_000.0
    executionMode: str = "paper"
    maxNotionalPerTrade: float = 5_000.0
    transactionCostBps: float = 5.0
    slippageBps: float = 5.0


class ExecuteRequest(RankingRequest):
    submitTopK: int | None = None


def _records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    serializable = frame.copy()
    if limit is not None:
        serializable = serializable.head(limit)

    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].dt.strftime("%Y-%m-%dT%H:%M:%S")

    serializable = serializable.replace({np.nan: None})
    return serializable.to_dict(orient="records")


@app.get("/")
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/")
@app.post("/api/rank")
def rank(payload: RankingRequest) -> dict[str, Any]:
    universe = payload.universe or []
    if not universe:
        raise HTTPException(status_code=400, detail="Universe must contain at least one symbol.")

    try:
        output = run_stock_oracle(
            AppConfig(
                universe=universe,
                benchmark=payload.benchmark.strip().upper() or "SPY",
                start_date=payload.startDate,
                holdout_days=payload.holdoutDays,
                top_k=payload.topK,
                intraday_period_days=payload.intradayPeriodDays,
                intraday_interval=payload.intradayInterval,
                signal_bar_index=payload.signalBarIndex,
                enable_live_news=payload.enableLiveNews,
                enable_live_options=payload.enableLiveOptions,
                enable_earnings_features=payload.enableEarningsFeatures,
                starting_capital=payload.startingCapital,
                execution_mode=payload.executionMode,
                max_notional_per_trade=payload.maxNotionalPerTrade,
                transaction_cost_bps=payload.transactionCostBps,
                slippage_bps=payload.slippageBps,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ranking": _records(output.ranking, limit=25),
        "metrics": output.metrics,
        "featureImportance": _records(output.feature_importance, limit=20),
        "holdoutPredictions": _records(output.holdout_predictions, limit=250),
        "backtestCurve": _records(output.backtest_curve, limit=250),
        "executionPlan": _records(output.execution_plan, limit=25),
    }


@app.post("/api/execute")
def execute(payload: ExecuteRequest) -> dict[str, Any]:
    ranked = rank(payload)
    execution_plan = pd.DataFrame(ranked["executionPlan"])
    if execution_plan.empty:
        raise HTTPException(status_code=400, detail="No execution plan was generated for the current ranking.")

    broker = get_broker(payload.executionMode)
    orders = [ExecutionPlan(**row) for row in execution_plan.head(payload.submitTopK or payload.topK).to_dict(orient="records")]
    result = broker.place_orders(orders)
    return {"submitted": len(result["orders"]), "orders": result["orders"], "positions": result["positions"]}


@app.get("/api/positions")
def positions(mode: str = "paper") -> dict[str, Any]:
    broker = get_broker(mode)
    return {"positions": broker.positions(), "orders": broker.orders()[:25]}