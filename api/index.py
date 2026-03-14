from __future__ import annotations

from fastapi import FastAPI


app = FastAPI(title="StockOracle")


@app.get("/")
@app.get("/api")
def root() -> dict[str, str]:
    return {
        "name": "StockOracle",
        "status": "ready",
        "rankEndpoint": "/api/rank",
        "healthEndpoint": "/api/health",
    }