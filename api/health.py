from __future__ import annotations

from fastapi import FastAPI


app = FastAPI(title="StockOracle Health")


@app.get("/")
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}