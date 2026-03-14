from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import pandas as pd


def _storage_path() -> Path:
    root = os.getenv("STOCKORACLE_EXECUTION_DIR")
    base = Path(root) if root else Path(".stockoracle")
    base.mkdir(parents=True, exist_ok=True)
    return base / "paper_broker.json"


def _load_state() -> dict:
    path = _storage_path()
    if not path.exists():
        return {"orders": [], "positions": {}, "fills": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"orders": [], "positions": {}, "fills": []}


def _save_state(state: dict) -> None:
    _storage_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


@dataclass(slots=True)
class ExecutionPlan:
    symbol: str
    side: str
    quantity: int
    notional: float
    reference_price: float
    predicted_return: float
    confidence: float
    final_score: float


class Broker(Protocol):
    def place_orders(self, orders: list[ExecutionPlan]) -> dict: ...

    def positions(self) -> list[dict]: ...

    def orders(self) -> list[dict]: ...


class PaperBroker:
    def place_orders(self, orders: list[ExecutionPlan]) -> dict:
        state = _load_state()
        now = datetime.now(tz=timezone.utc).isoformat()
        placed_orders: list[dict] = []

        for order in orders:
            if order.quantity <= 0:
                continue

            order_id = str(uuid4())
            signed_quantity = order.quantity if order.side == "buy" else -order.quantity
            position = state["positions"].get(order.symbol, {"symbol": order.symbol, "quantity": 0, "avgPrice": 0.0})
            current_quantity = int(position["quantity"])
            new_quantity = current_quantity + signed_quantity

            saved_order = {
                "orderId": order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "notional": order.notional,
                "referencePrice": order.reference_price,
                "predictedReturn": order.predicted_return,
                "confidence": order.confidence,
                "finalScore": order.final_score,
                "status": "filled",
                "timestamp": now,
            }

            state["fills"].append(
                {
                    "fillId": str(uuid4()),
                    "orderId": order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.reference_price,
                    "timestamp": now,
                }
            )

            if new_quantity == 0:
                state["positions"].pop(order.symbol, None)
            else:
                if current_quantity == 0 or (current_quantity > 0) != (signed_quantity > 0):
                    avg_price = order.reference_price
                else:
                    avg_price = ((abs(current_quantity) * float(position["avgPrice"])) + (order.quantity * order.reference_price)) / max(abs(new_quantity), 1)
                state["positions"][order.symbol] = {
                    "symbol": order.symbol,
                    "quantity": new_quantity,
                    "avgPrice": avg_price,
                    "updatedAt": now,
                }

            state["orders"].append(saved_order)
            placed_orders.append(saved_order)

        _save_state(state)
        return {"orders": placed_orders, "positions": self.positions()}

    def positions(self) -> list[dict]:
        state = _load_state()
        return sorted(state["positions"].values(), key=lambda item: item["symbol"])

    def orders(self) -> list[dict]:
        state = _load_state()
        return list(reversed(state["orders"]))


def build_execution_plan(
    ranking: pd.DataFrame,
    top_k: int,
    capital: float,
    max_position_weight: float,
    max_notional_per_trade: float,
) -> list[ExecutionPlan]:
    selected = ranking.head(top_k).copy()
    if selected.empty:
        return []

    position_budget = min(capital * max_position_weight, max_notional_per_trade)
    plans: list[ExecutionPlan] = []
    for _, row in selected.iterrows():
        price = float(row.get("close", 0.0) or 0.0)
        if price <= 0:
            continue
        quantity = int(math.floor(position_budget / price))
        if quantity <= 0:
            continue
        plans.append(
            ExecutionPlan(
                symbol=str(row["symbol"]),
                side="buy" if float(row.get("predicted_return", 0.0) or 0.0) >= 0 else "sell",
                quantity=quantity,
                notional=quantity * price,
                reference_price=price,
                predicted_return=float(row.get("predicted_return", 0.0) or 0.0),
                confidence=float(row.get("confidence", 0.0) or 0.0),
                final_score=float(row.get("final_score", 0.0) or 0.0),
            )
        )
    return plans


def execution_plan_frame(plans: list[ExecutionPlan]) -> pd.DataFrame:
    return pd.DataFrame([asdict(plan) for plan in plans])


def get_broker(mode: str) -> Broker:
    normalized = mode.lower().strip()
    if normalized == "paper":
        return PaperBroker()
    raise ValueError(f"Unsupported execution mode: {mode}")