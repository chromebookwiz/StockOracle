from __future__ import annotations

import hmac
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from hmac import compare_digest
from typing import Protocol
from uuid import uuid4

import httpx
import pandas as pd

from .storage import get_binary_store


def _load_json_state(key: str, default: dict) -> dict:
    payload = get_binary_store().get_bytes(key)
    if payload is None:
        return default
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return default


def _save_json_state(key: str, state: dict) -> None:
    get_binary_store().set_bytes(key, json.dumps(state, indent=2).encode("utf-8"))


def _execution_state_key(mode: str) -> str:
    return f"execution/{mode.lower().strip()}.json"


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
        state = _load_json_state(_execution_state_key("paper"), {"orders": [], "positions": {}, "fills": []})
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

        _save_json_state(_execution_state_key("paper"), state)
        return {"orders": placed_orders, "positions": self.positions()}

    def positions(self) -> list[dict]:
        state = _load_json_state(_execution_state_key("paper"), {"orders": [], "positions": {}, "fills": []})
        return sorted(state["positions"].values(), key=lambda item: item["symbol"])

    def orders(self) -> list[dict]:
        state = _load_json_state(_execution_state_key("paper"), {"orders": [], "positions": {}, "fills": []})
        return list(reversed(state["orders"]))


class AlpacaBroker:
    def __init__(self) -> None:
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca credentials are missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json",
            },
            timeout=20.0,
        )

    def _request(self, method: str, path: str, *, json_payload: dict | None = None, params: dict | None = None) -> dict | list:
        response = self.client.request(method, path, json=json_payload, params=params)
        response.raise_for_status()
        return response.json()

    def place_orders(self, orders: list[ExecutionPlan]) -> dict:
        placed_orders: list[dict] = []
        for order in orders:
            payload = {
                "symbol": order.symbol,
                "qty": str(order.quantity),
                "side": order.side,
                "type": "market",
                "time_in_force": "day",
            }
            response = self._request("POST", "/v2/orders", json_payload=payload)
            placed_orders.append(response)
        return {"orders": placed_orders, "positions": self.positions()}

    def positions(self) -> list[dict]:
        response = self._request("GET", "/v2/positions")
        return response if isinstance(response, list) else []

    def orders(self) -> list[dict]:
        response = self._request("GET", "/v2/orders", params={"status": "all", "limit": 25, "direction": "desc"})
        return response if isinstance(response, list) else []


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


def requires_execution_auth() -> bool:
    return bool(os.getenv("STOCKORACLE_EXECUTION_TOKEN"))


def require_execution_auth_configured() -> None:
    if not os.getenv("STOCKORACLE_EXECUTION_TOKEN"):
        raise ValueError("STOCKORACLE_EXECUTION_TOKEN must be configured for execution endpoints.")


def validate_execution_auth(token: str | None) -> None:
    expected = os.getenv("STOCKORACLE_EXECUTION_TOKEN")
    if not expected:
        raise ValueError("STOCKORACLE_EXECUTION_TOKEN must be configured for execution endpoints.")
    if not token or not compare_digest(token, expected):
        raise ValueError("Execution auth token is missing or invalid.")


def build_confirmation_token(execution_plan: pd.DataFrame) -> str:
    secret = os.getenv("STOCKORACLE_CONFIRMATION_SECRET")
    if not secret:
        raise ValueError("STOCKORACLE_CONFIRMATION_SECRET must be configured for execution confirmation.")
    columns = ["symbol", "side", "quantity", "reference_price"]
    payload = execution_plan.loc[:, [column for column in columns if column in execution_plan.columns]].copy()
    payload["reference_price"] = payload.get("reference_price", pd.Series(dtype=float)).round(4) if "reference_price" in payload.columns else pd.Series(dtype=float)
    message = payload.to_json(orient="records")
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), sha256).hexdigest()


def get_broker(mode: str) -> Broker:
    normalized = mode.lower().strip()
    if normalized == "paper":
        return PaperBroker()
    if normalized == "alpaca":
        return AlpacaBroker()
    raise ValueError(f"Unsupported execution mode: {mode}")