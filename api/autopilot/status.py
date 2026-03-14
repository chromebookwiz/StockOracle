from __future__ import annotations

import os
import sys
from hmac import compare_digest
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stockoracle.autopilot import autopilot_status  # noqa: E402


app = FastAPI(title="StockOracle Autopilot Status API")


def _validate_autopilot_access(request_token: str | None, authorization: str | None) -> None:
    expected = os.getenv("STOCKORACLE_AUTOPILOT_TOKEN") or os.getenv("CRON_SECRET") or os.getenv("STOCKORACLE_EXECUTION_TOKEN")
    if not expected:
        raise ValueError("Configure STOCKORACLE_AUTOPILOT_TOKEN, CRON_SECRET, or STOCKORACLE_EXECUTION_TOKEN for autopilot endpoints.")

    bearer = authorization.removeprefix("Bearer ").strip() if authorization and authorization.startswith("Bearer ") else None
    candidate = request_token or bearer
    if not candidate or not compare_digest(candidate, expected):
        raise ValueError("Autopilot token is missing or invalid.")


@app.get("/")
@app.get("/api/autopilot/status")
def status(autopilotToken: str | None = None, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        _validate_autopilot_access(autopilotToken, authorization)
        return autopilot_status()
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc