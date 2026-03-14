from __future__ import annotations

import math

import numpy as np
import pandas as pd


def run_backtest(
    holdout_predictions: pd.DataFrame,
    top_k: int,
    transaction_cost_bps: float,
    slippage_bps: float,
    max_position_weight: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    if holdout_predictions.empty:
        return {}, pd.DataFrame(columns=["date", "gross_return", "net_return", "turnover", "equity", "positions"])

    total_cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    previous_weights: dict[str, float] = {}
    equity = 1.0
    rows: list[dict[str, object]] = []

    for trade_date, day_frame in holdout_predictions.sort_values(["date", "score"], ascending=[True, False]).groupby("date"):
        selected = day_frame.head(top_k).copy()
        if selected.empty:
            continue

        target_weight = min(1 / len(selected), max_position_weight)
        weights = {symbol: target_weight for symbol in selected["symbol"]}
        turnover = 0.0
        all_symbols = set(previous_weights) | set(weights)
        for symbol in all_symbols:
            turnover += abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))

        gross_return = float((selected["target_return"] * target_weight).sum())
        net_return = gross_return - turnover * total_cost_rate
        equity *= 1 + net_return

        rows.append(
            {
                "date": pd.Timestamp(trade_date),
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "equity": equity,
                "positions": ", ".join(selected["symbol"].tolist()),
            }
        )
        previous_weights = weights

    curve = pd.DataFrame(rows)
    if curve.empty:
        return {}, curve

    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1
    net_returns = curve["net_return"]
    annualized_return = math.pow(curve["equity"].iloc[-1], 252 / max(len(curve), 1)) - 1
    volatility = float(net_returns.std(ddof=0) * math.sqrt(252)) if len(curve) > 1 else 0.0
    sharpe = float((net_returns.mean() / net_returns.std(ddof=0)) * math.sqrt(252)) if len(curve) > 1 and net_returns.std(ddof=0) > 0 else 0.0

    metrics = {
        "backtest_total_return": float(curve["equity"].iloc[-1] - 1),
        "backtest_annualized_return": float(annualized_return),
        "backtest_annualized_volatility": volatility,
        "backtest_sharpe": sharpe,
        "backtest_max_drawdown": float(curve["drawdown"].min()),
        "backtest_win_rate": float((curve["net_return"] > 0).mean()),
        "backtest_avg_turnover": float(curve["turnover"].mean()),
    }
    return metrics, curve